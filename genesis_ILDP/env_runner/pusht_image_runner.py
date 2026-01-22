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
from genesis_ILDP.gym_util.multistep_wrapper_parallel import MultiStepWrapper # new wrapper

# from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.utils.pytorch_util import dict_apply
from genesis_ILDP.env_runner.base_image_runner import BaseImageRunner
from genesis_ILDP.env_runner.eval_utils import VideoManager, EvalLogManager

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from collections import defaultdict
from genesis_ILDP.policy.dummy_policy import DummyPolicy

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
                 output_dir,
                 n_train = 50,
                 n_train_vis = 0,
                 n_test = 10,
                 n_test_vis = 10,
                 n_obs_steps = 2,
                 n_action_steps = 8,
                 diff_steps = 100,
                 max_steps=200,
                 image_shape=(96, 96),
                 tqdm_interval_sec=1.0,
                 n_envs = None,
                 fps = 30,
                 device = 'cuda:0',
                 enable_past_action=True,
                 train_start_seed = 0,
                 test_start_seed = 5000,
                 enable_render = False,
                 max_envs_running = 3,
                 done_ratio = 0.85,
                 episode_recording = False,
                 video_dir = None,
                 spawn_mode = 'uniform',
                 uniform_sampler_config = None,
                 circular_sampler_config = None,
                 show_interactive_viewer=False,
                 ):
        super().__init__(output_dir)
        if n_envs is None:
            self.n_envs = n_train + n_test
            print(f"Using computed n_envs: {self.n_envs} (train: {n_train}, test: {n_test})")
        else:
            self.n_envs = n_envs
            print(f"Using provided n_envs: {self.n_envs}")

        self.parallel_envs_counts = min(max_envs_running, self.n_envs) 
        
        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = min(self.n_train, n_train_vis)
        self.n_test_vis = min(self.n_test, n_test_vis)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.diff_steps = diff_steps
        self.device = device
        self.max_steps = max_steps
        self.enable_past_action = enable_past_action
        self.seed_train = train_start_seed
        self.seed_test = test_start_seed
        self.enable_render = enable_render
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env_seeds = None
        self.info = None
        self.done_ratio = done_ratio
        self.show_interactive_viewer = show_interactive_viewer

        # Initialize video manager with configurable directory
        if video_dir is None:
            video_dir = output_dir  # Fallback to default


        self.video_manager = VideoManager(
            base_video_path=video_dir,
            n_train=n_train,
            n_test=n_test,
            n_train_vis=n_train_vis,
            n_test_vis=n_test_vis,
            train_start_seed=train_start_seed,
            test_start_seed=test_start_seed
        )

        # Initialize log manager
        self.log_manager = EvalLogManager(
            n_train=n_train,
            n_test=n_test,
            n_train_vis=n_train_vis,
            n_test_vis=n_test_vis
        )

        # Track whether video directories have been initialized
        self._video_dirs_initialized = False

        self.episode_obs = None
        self.episode_reward = [[] for _ in range(self.n_envs)]
        self.episode_info = dict()
        self.max_reward = [None for _ in range(self.n_envs)]

        self.completion_timesteps = [None] * self.n_envs  # Record completion timesteps
        self.global_timestep = 0  # Global step counter across all envs

        self.episode_recording = episode_recording # whether to enable recording the trajectories details
        # for fine-tuning usage

        # Per-environment temporary buffers (store data during episode rollout)
        self.diffusion_action_buffer = [[] for _ in range(self.n_envs)]  # per-env diffusion actions
        self.env_obs_buffer = [[] for _ in range(self.n_envs)]  # per-env observations (unnormalized)
        self.env_reward_buffer = [[] for _ in range(self.n_envs)]  # per-env rewards

        # Episode buffer: stores all collected trajectories
        # All buffers use torch.empty for consistent format and efficiency
        self.episode_buffer = {
            'obs': {
                'img': torch.empty(size=(0, self.n_obs_steps, 3, *image_shape), dtype=torch.float32),  # UNNORMALIZED (global_steps, n_obs_steps, C, H, W)
                'agent_pos': torch.empty(size=(0, self.n_obs_steps, 2), dtype=torch.float32)  # UNNORMALIZED (global_steps, n_obs_steps, Dp)
            },
            'last_obs': {  # For bootstrapping in RL training
                'img': torch.empty(size=(0, self.n_obs_steps, 3, *image_shape), dtype=torch.float32),  # UNNORMALIZED (N_episodes, n_obs_steps, C, H, W)
                'agent_pos': torch.empty(size=(0, self.n_obs_steps, 2), dtype=torch.float32)  # UNNORMALIZED (N_episodes, n_obs_steps, Dp)
            },
            'action': torch.empty(size=(0, self.diff_steps+1, self.n_action_steps, 2), dtype=torch.float32),  # NORMALIZED diffusion actions (global_steps, diff_steps+1, horizons, Da)
            'reward': torch.empty(size=(0,), dtype=torch.float32),  # (global_steps,)
            'episode_ends': torch.empty(size=(0,), dtype=torch.int64),  # (N_episodes,) cumulative indices where episodes end
            'is_truncate': torch.empty(size=(0,), dtype=torch.bool)  # (N_episodes,) whether episode was truncated by max_steps
        }

        self.env = MultiStepWrapper(
                    PushTEnv(
                        render_size=image_shape,
                        fps = fps,
                        show_fps=False,
                        device=device,
                        done_ratio=done_ratio,
                        spawn_mode = spawn_mode,
                        uniform_sampler_config = uniform_sampler_config,
                        circular_sampler_config = circular_sampler_config,
                    ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                n_envs = self.n_envs, 
            )
    
        self._setup_envs()

    def _clean_buffer(self):
        """Reset all buffers to empty state using torch.empty for consistency."""
        image_shape = self.env.env.render_size
        self.episode_buffer = {
            'obs': {
                'img': torch.empty(size=(0, self.n_obs_steps, 3, *image_shape), dtype=torch.float32),  # UNNORMALIZED
                'agent_pos': torch.empty(size=(0, self.n_obs_steps, 2), dtype=torch.float32)  # UNNORMALIZED
            },
            'last_obs': {
                'img': torch.empty(size=(0, self.n_obs_steps, 3, *image_shape), dtype=torch.float32),  # UNNORMALIZED
                'agent_pos': torch.empty(size=(0, self.n_obs_steps, 2), dtype=torch.float32)  # UNNORMALIZED
            },
            'action': torch.empty(size=(0, self.diff_steps+1, self.n_action_steps, 2), dtype=torch.float32),  # NORMALIZED
            'reward': torch.empty(size=(0,), dtype=torch.float32),
            'episode_ends': torch.empty(size=(0,), dtype=torch.int64),
            'is_truncate': torch.empty(size=(0,), dtype=torch.bool)
        }
        self.diffusion_action_buffer = [[] for _ in range(self.n_envs)]
        self.env_obs_buffer = [[] for _ in range(self.n_envs)]
        self.env_reward_buffer = [[] for _ in range(self.n_envs)]  

    def get_saved_trajectories(self, ):
        return self.episode_buffer

    def run(self, policy, generator=None, wandb_run=None):
        """
        Run evaluation with the given policy and return logging data.

        Args:
            policy: Policy to evaluate
            generator: Optional torch.Generator for reproducible action sampling
            wandb_run: Wandb run object (or None if wandb disabled)

        Returns:
            dict: JSON-serializable logging data (videos excluded)
        """

        self._prepare_run()
        self._clean_buffer() # clean the episode buffer when running a series of new collection under new seeds(if not given specific seed, the 
        # seed is being inferred from the last used seed + n_test)

        obs = self.env.reset()

        self.past_action = None
        done = False

        pbar = tqdm.tqdm(total=self.max_steps//self.n_action_steps, desc=f"Eval PushTImageRunner",
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
                    result = policy.predict_action(obs_dict, recording_diffusion=self.episode_recording, generator=generator)

                if isinstance(result, dict):
                    action = result['action']
                    current_action_diffusion_buffer = result.get('action_diffusion_buffer', None)
                else:
                    action = result
                    current_action_diffusion_buffer = None

                # Store per-environment observations and diffusion actions before stepping
                if self.episode_recording and current_action_diffusion_buffer is not None:
                    for local_idx, env_idx in enumerate(active_envs_idx):
                        # Store UNNORMALIZED observations (policy will handle the normalization according to dataset normalizer)
                        self.env_obs_buffer[env_idx].append({
                            'image': obs['image'][local_idx],  
                            'agent_pos': obs['agent_pos'][local_idx]
                        })
                        # Store NORMALIZED actions (diff_steps+1, horizons, Da) on GPU
                        self.diffusion_action_buffer[env_idx].append(current_action_diffusion_buffer[local_idx])

                # record the last step observation and current action(note that we should pad for
                # terminated envs their observations and actions)

                envs_count = action.shape[0]
                if envs_count != len(active_envs_idx):
                    raise ValueError("Inconsistent envs number for active envs when predicting actions")
                else:
                    Active_action = {
                        'envs_idx': active_envs_idx,
                        'action': action
                    }

                obs, reward, done, info = self.env.step(Active_action['action'])

                if self.episode_recording:
                    for local_idx, env_idx in enumerate(active_envs_idx):
                        self.env_reward_buffer[env_idx].append(reward[env_idx]) 

                self.global_timestep += 1
            
                if self.enable_past_action:
                    self._update_past_action(Active_action)

                obs, active_envs_idx = self._process_info(obs, reward, info, done)

                # the global ending condition for the rolling out
                if 2 not in done:
                    is_done = True
                pbar.update(1)  # Update by 1 step since all envs run together
                
        except Exception as e:
            print(f"Error during policy execution: {e}")
            raise e
        finally:
            pbar.close()

        if self.enable_render:
            try:
                self.env.stop_recording(self.video_manager.base_generate_path)
                self._process_generated_videos()
                self._cleanup_temp_files()
            except Exception as e:
                print(f"Warning: Video processing failed: {e}")

        # Update seeds for next run
        self.seed_train = self.seed_train + self.n_train
        self.seed_test = self.seed_test + self.n_test

        # Create and return logging data using EvalLogManager (JSON-safe, no videos)
        log_data = self._create_log_data()

        # Print summary statistics
        EvalLogManager.print_summary(log_data, self.n_train, self.n_test, self.completion_timesteps)

        # If wandb is enabled, create and log videos separately
        if wandb_run is not None:
            wandb_videos = self.log_manager.create_wandb_videos(
                env_seeds=self.env_seeds,
                file_paths=self.video_manager.file_paths,
                wandb_enabled=True
            )
            # Log videos to wandb immediately (not returned in log_data)
            if wandb_videos:
                wandb_run.log(wandb_videos, step=wandb_run.step)
                print(f"✓ Logged {len(wandb_videos)} videos to wandb")

        return log_data
    

    def _update_past_action(self, action_dict):
        if self.past_action is None:
            self.past_action = [[] for _ in range(self.n_envs)]
        assert 'envs_idx' in action_dict.keys()
        local_idx = 0
        for i in range(self.n_envs):
            if i in action_dict['envs_idx']:
                self.past_action[i].append(action_dict['action'][local_idx])
                local_idx = local_idx + 1
            else:
                self.past_action[i].append(None)

    def _setup_envs(self,):
        if self.device == 'cuda:0' or self.device == 'cuda':
            self.env.start(n_envs=self.n_envs, env_separate=True, show_interact_viewer = False)
        else:
            self.env.start(n_envs=self.n_envs, env_separate=False, show_interact_viewer = self.show_interactive_viewer) 
            # The mac does not support separate the env for rendering

        print(f"------PUSHT ENV RUNNER SETUP COMPLETE!------\
              \n Configuration:  n_test={self.n_test},  n_train={self.n_train}, \
              n_envs={self.n_envs}\n max_steps={self.max_steps}")

    def _update_seeds(self, seed_train=None, seed_test=None):
        self.seed_train = seed_train if seed_train is not None else self.seed_train
        self.seed_test = seed_test if seed_test is not None else self.seed_test

    def _prepare_run(self, enable_render=True):
        """
        Prepare per-run state (called before each run()).
        """

        # Update env_seeds based on current seed values (which increment between runs during training)
        # For collecting RL replay buffer, the collecting behaviours are the same, except '_update_seeds_' will be called 
        # before all the seeds are collected in one turn. 

        self.env_seeds = np.concatenate((
            np.arange(start=self.seed_train, stop=self.seed_train + self.n_train),
            np.arange(start=self.seed_test, stop=self.seed_test + self.n_test)
        ), axis=0)

        self.env.seed(self.env_seeds)

        if self.enable_render:
            if not self._video_dirs_initialized:
                self.video_manager.setup_directories()
                self._video_dirs_initialized = True
                print(f"✓ Video dirs: {self.video_manager.train_video_dir.parent.parent}") # timestep -> train -> parent

            self.video_manager.update_file_paths(self.seed_train, self.seed_test)

        # Reset per-run state (used for replay buffer saving)
        self.info = None
        self.episode_obs = None
        self.episode_reward = [[] for _ in range(self.n_envs)]
        self.episode_info = dict()
        self.max_reward = [None for _ in range(self.n_envs)]
        self.completion_timesteps = [None] * self.n_envs  # CRITICAL: Reset completion timesteps!
        self.global_timestep = 0  # Reset global timestep counter


    def _process_generated_videos(self):
        """Process generated videos using VideoManager."""
        return self.video_manager.process_generated_videos()

    def _cleanup_temp_files(self):
        """Clean up temporary video files using VideoManager."""
        self.video_manager.cleanup_temp_files()

    def _process_info(self, obs, reward, info, env_status, final_saving=False):
        """Process environment info and rewards.

        env_status codes from multistep_wrapper_parallel.py:
        0: Already finished in previous iterations
        1: Just finished (truncated by max_steps)
        2: Still active
        3: Just finished (natural success - reached done condition)
        """
        new_obs = defaultdict(list)
        if self.info is None:
            self.info = info

        active_env_indices = []
        for i in range(self.n_envs):
            # check here if the rewards in terminated envs should not be saved into the buffer 
            if reward[i] is None:
                raise ValueError("Reward saving error!")
            self.episode_reward[i].append(reward[i].cpu().numpy())

            if env_status[i] == 2:  # Still active
                active_env_indices.append(i)

            elif env_status[i] == 3:  # Natural success
                self.completion_timesteps[i] = self.global_timestep * self.n_action_steps

                for key in self.info.keys():
                    if key == 'envs_idx': continue
                    self.info[key][i] = info[key][i]
    
                if self.episode_recording:
                    self._transfer_env_to_episode_buffer(i, obs, is_truncate=False)

            elif env_status[i] == 1:  # Truncated by max_steps, then this last obs should be used as bootstrapping
                if self.episode_recording:
                    self._transfer_env_to_episode_buffer(i, obs, is_truncate=True)

            # env_status[i] == 0 (already done): do nothing, already transferred and terminated
            # (without sending making any new actions sending via env api)

        if active_env_indices:
            if isinstance(obs, dict)==False:
                raise TypeError('Obs Type Error!')

            for key in obs.keys():
                active_obs = []
                for i in active_env_indices:
                    if obs[key][i] is None:
                        raise ValueError("Obs Saving Error!")
                    active_obs.append(obs[key][i])

                if active_obs:
                    new_obs[key] = torch.stack(active_obs, dim=0)

        return new_obs, active_env_indices

    def _transfer_env_to_episode_buffer(self, env_idx, last_obs=None, is_truncate=True):
        """
        Transfer completed environment data to the global episode_buffer.

        Args:
            env_idx: Index of the environment that just finished
            last_obs: Terminal observation for bootstrapping (dict with 'image' and 'agent_pos' keys)
            is_truncate: Whether episode was truncated by max_steps (True) or naturally terminated (False)
        """
        if len(self.env_obs_buffer[env_idx]) == 0:
            return

        # Stack observations, actions, rewards for this episode (UNNORMALIZED obs, NORMALIZED actions)
        episode_images = torch.stack([obs['image'] for obs in self.env_obs_buffer[env_idx]], dim=0)
        episode_agent_pos = torch.stack([obs['agent_pos'] for obs in self.env_obs_buffer[env_idx]], dim=0)
        episode_actions = torch.stack(self.diffusion_action_buffer[env_idx], dim=0)
        episode_rewards = torch.stack(self.env_reward_buffer[env_idx], dim=0)

        # Append to global buffer using torch.cat (unified logic, no if-else branching)
        self.episode_buffer['obs']['img'] = torch.cat([self.episode_buffer['obs']['img'], episode_images], dim=0)
        self.episode_buffer['obs']['agent_pos'] = torch.cat([self.episode_buffer['obs']['agent_pos'], episode_agent_pos], dim=0)
        self.episode_buffer['action'] = torch.cat([self.episode_buffer['action'], episode_actions], dim=0)
        self.episode_buffer['reward'] = torch.cat([self.episode_buffer['reward'], episode_rewards], dim=0)

        # Store last observation for bootstrapping (if provided)
        if last_obs is not None:
            # UNNORMALIZED last observation: (n_obs_steps, C, H, W) and (n_obs_steps, Dp)
            last_img = last_obs['image'].unsqueeze(0)  # Add batch dimension -> (1, n_obs_steps, C, H, W)
            last_pos = last_obs['agent_pos'].unsqueeze(0)  # Add batch dimension -> (1, n_obs_steps, Dp)
            self.episode_buffer['last_obs']['img'] = torch.cat([self.episode_buffer['last_obs']['img'], last_img], dim=0)
            self.episode_buffer['last_obs']['agent_pos'] = torch.cat([self.episode_buffer['last_obs']['agent_pos'], last_pos], dim=0)

        # Record episode metadata
        current_global_steps = self.episode_buffer['obs']['img'].shape[0]
        episode_end_idx = torch.tensor([current_global_steps], dtype=torch.int64)
        truncate_flag = torch.tensor([is_truncate], dtype=torch.bool)

        self.episode_buffer['episode_ends'] = torch.cat([self.episode_buffer['episode_ends'], episode_end_idx], dim=0)
        self.episode_buffer['is_truncate'] = torch.cat([self.episode_buffer['is_truncate'], truncate_flag], dim=0)

        # Clear per-env buffers
        self.env_obs_buffer[env_idx] = []
        self.diffusion_action_buffer[env_idx] = []
        self.env_reward_buffer[env_idx] = []

    def _create_log_data(self):
        """Create logging data using EvalLogManager."""
        return self.log_manager.create_log_data(
            episode_rewards=self.episode_reward,
            completion_timesteps=self.completion_timesteps,
            env_seeds=self.env_seeds,
            file_paths=self.video_manager.file_paths
        )

if __name__ == "__main__":
        # use collect_data policy to check availability
        output_dir = ''
        Runner = PushTImageRunner(output_dir)
        policy = DummyPolicy(Runner.n_action_steps)
        Runner.run(policy)
        