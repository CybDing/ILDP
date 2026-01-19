from typing import Dict
import genesis as gs
import wandb
import numpy as np
import torch
import collections
from pathlib import Path
import tqdm
import dill
import math
import time
import shutil
import wandb.sdk.data_types.video as wv
from genesis_ILDP.env.pushT_env import PushTEnv
from genesis_ILDP.gym_util.multistep_wrapper_parallel import MultiStepWrapper

from genesis_ILDP.utils.pytorch_util import dict_apply
from genesis_ILDP.env_runner.pusht_image_runner import PushTImageRunner
from genesis_ILDP.env_runner.eval_utils import VideoManager, EvalLogManager

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from collections import defaultdict


class PushTMemoryRunner(PushTImageRunner):
    """
    Extended runner that supports stateful rollout using MemoryNet.
    During rollout, uses state evolution instead of image encoding.
    """
    def __init__(self, *args, use_memory_rollout=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_memory_rollout = use_memory_rollout
        self.env_states = {}  # Per-environment state tracking

    def run(self, policy, generator=None, wandb_run=None):
        """
        Run evaluation with memory-based rollout if enabled.
        """
        if not self.use_memory_rollout:
            return super().run(policy, generator, wandb_run)

        self._prepare_run()
        self._clean_buffer()

        obs = self.env.reset()

        # Initialize states for all environments from first observation
        self._initialize_env_states(policy, obs)

        self.past_action = None
        done = False

        pbar = tqdm.tqdm(total=self.max_steps//self.n_action_steps, desc=f"Eval PushTMemoryRunner",
                             leave=False)

        if self.enable_render:
            self.env.start_recording()

        try:
            active_envs_idx = list(range(self.n_envs))
            is_done = False

            while not is_done:
                obs_dict = dict(obs)

                if 'envs_idx' in obs_dict:
                    del obs_dict['envs_idx']

                with torch.no_grad():
                    # Use state-based prediction instead of image-based
                    action = self._predict_action_from_state(policy, obs_dict, active_envs_idx, generator)

                if self.episode_recording:
                    for local_idx, env_idx in enumerate(active_envs_idx):
                        self.env_obs_buffer[env_idx].append({
                            'image': obs['image'][local_idx],
                            'agent_pos': obs['agent_pos'][local_idx]
                        })

                envs_count = action.shape[0]
                if envs_count != len(active_envs_idx):
                    raise ValueError("Inconsistent envs number for active envs when predicting actions")
                else:
                    Active_action = {
                        'envs_idx': active_envs_idx,
                        'action': action
                    }

                obs, reward, done, info = self.env.step(Active_action)

                # Update states using executed actions
                self._update_env_states(policy, Active_action, active_envs_idx)

                if self.episode_recording:
                    for local_idx, env_idx in enumerate(active_envs_idx):
                        self.env_reward_buffer[env_idx].append(reward[env_idx])

                self.global_timestep += 1

                if self.enable_past_action:
                    self._update_past_action(Active_action)

                obs, active_envs_idx = self._process_info(obs, reward, info, done)

                if 2 not in done:
                    is_done = True
                pbar.update(1)

        except Exception as e:
            print(f"Error during policy execution: {e}")
            raise e

        finally:
            pbar.close()
            if self.enable_render:
                self.env.stop_recording()

        return self._collect_results(wandb_run)

    def _initialize_env_states(self, policy, obs):
        """Initialize state for all environments in parallel"""
        all_states = policy.compute_initial_state(obs['image'], obs['agent_pos'])

        for env_idx in range(self.n_envs):
            self.env_states[env_idx] = all_states[env_idx:env_idx+1]

    def _predict_action_from_state(self, policy, obs_dict, active_envs_idx, generator=None):
        """Predict actions using current states instead of images"""
        states = torch.cat([self.env_states[env_idx] for env_idx in active_envs_idx], dim=0)

        agent_pos = obs_dict['agent_pos']

        action = policy.predict_action_from_state(states, agent_pos, generator=generator)

        return action

    def _update_env_states(self, policy, Active_action, active_envs_idx):
        """Update environment states using executed actions"""
        actions = Active_action['action']

        for local_idx, env_idx in enumerate(active_envs_idx):
            action = actions[local_idx:local_idx+1]

            prev_state = self.env_states[env_idx]

            next_state = policy.compute_next_state(prev_state, action)

            self.env_states[env_idx] = next_state

    def reset_env_state(self, env_idx, policy, obs):
        """Reset state for a specific environment when episode ends"""
        img_tensor = obs['image'][env_idx:env_idx+1]
        agent_pos = obs['agent_pos'][env_idx:env_idx+1]

        state = policy.compute_initial_state(img_tensor, agent_pos)
        self.env_states[env_idx] = state
