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
                 n_train = 50, # Using train seed 
                 n_train_vis = 0,
                 n_test = 10, # Using test seed
                 n_test_vis = 10,
                 n_obs_steps = 2,
                 n_action_steps = 8,
                 diff_steps = 100,  # Number of diffusion steps
                 max_steps=200,
                 image_shape=(96, 96),
                 tqdm_interval_sec=1.0,
                 n_envs = None,
                 fps = 30,
                 device = 'cuda:0',
                #  crf = 22, # video quality
                 enable_past_action=True,
                 train_start_seed = 0,
                 test_start_seed = 5000, 
                 # does not reach 100000 episodes
                 enable_render = False,
                 max_envs_running = 3,
                 done_ratio = 0.85, 
                 episode_recording = False,
                 video_dir = None,  
                 spawn_center = (-0.3, 0.3),
                 spawn_range_scale = 0.65,
                 # Manual assistant parameters
                 enable_manual_assistant = True,
                 stagnation_steps = 10,  # N: consecutive steps for detection
                 agent_pos_threshold = 0.01,  # Movement threshold for agent (easily editable)
                 object_pos_threshold = 0.005,  # Movement threshold for object XY (easily editable)
                 ):
        super().__init__(output_dir)

        # Manual assistant configuration
        self.enable_manual_assistant = enable_manual_assistant
        self.stagnation_steps = stagnation_steps
        self.agent_pos_threshold = agent_pos_threshold
        self.object_pos_threshold = object_pos_threshold
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

        if video_dir is None:
            video_dir = output_dir  
                    
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

        # Timestep tracking for completion analysis
        # self.episode_timesteps = [0] * self.n_envs  # Current timesteps per environment
        self.completion_timesteps = [None] * self.n_envs  # Record completion timesteps
        self.global_timestep = 0  # Global step counter across all envs

        self.episode_recording = episode_recording # whether to enable recording the trajectories details
        # for fine-tuning usage

        # Per-environment temporary buffers (store data during episode rollout)
        self.diffusion_action_buffer = [[] for _ in range(self.n_envs)]  # per-env diffusion actions
        self.env_obs_buffer = [[] for _ in range(self.n_envs)]  # per-env observations (unnormalized)
        self.env_reward_buffer = [[] for _ in range(self.n_envs)]  # per-env rewards

        # Global episode buffer (accumulates completed episodes)
        self.episode_buffer = {
            'obs': {
                'img': [],  # Will store unnormalized or normalized obs depending on convenience
                'agent_pos': [],
            },
            'action': [],  # Stores NORMALIZED diffusion actions (diff_steps+1, horizons, Da)
            'reward': [],
            'episode_ends': []  # Records global_step index where each episode ends
        }

        # Manual assistant buffers for stagnation detection
        # Each env has a deque storing recent agent_pos and object_pose for comparison
        self.agent_pos_history = [collections.deque(maxlen=stagnation_steps) for _ in range(self.n_envs)]
        self.object_pose_history = [collections.deque(maxlen=stagnation_steps) for _ in range(self.n_envs)]
        """
        Shape meta for the episode buffer:

        [global_steps are concatenating the episodes together along the step axis, and use the
        episode_ends to indicate at which global_step each episode has ended]

        Data format details:
            obs: (condition for diffusion process)
                img: (global_steps, n_obs_steps, C, H, W) - UNNORMALIZED observations
                agent_pos: (global_steps, n_obs_steps, Dp) - UNNORMALIZED agent positions

            action: (global_steps, diff_steps+1, horizons, Da)
                    - NORMALIZED diffusion actions (normalized in policy during training)
                    - Includes the initial random noise step (index 0) through all denoising steps
                    - Can be directly fed into diffusion model for computing action mean prediction
                    [See action_diffusion_image_policy.py for diffusion process details]

            reward: (global_steps,) - Raw reward values from environment

            episode_ends: List[int] - Indices in global_steps where each episode ends
                          e.g., [150, 320, 500] means:
                          - Episode 0: steps 0-149
                          - Episode 1: steps 150-319
                          - Episode 2: steps 320-499
                          Length equals number of completed episodes

        Note: All data is stored on GPU for efficient computation during training/fine-tuning.
        """

        self.env = MultiStepWrapper(
                    PushTEnv(
                        render_size=image_shape,
                        fps = fps,
                        show_fps=False,
                        device=device, 
                        done_ratio=done_ratio, 
                        spawn_center = spawn_center, 
                        spawn_range_scale = spawn_range_scale, 
                    ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                n_envs = self.n_envs, 
            )
    
        self._setup_envs()

    def _clean_buffer(self, ):
        self.episode_buffer = {
                'obs': {
                    'img': [],
                    'agent_pos': [],
                },
                'action': [], 
                'reward': [],
                'episode_ends': []  
        } 
        self.diffusion_action_buffer = [[] for _ in range(self.n_envs)]  #
        self.env_obs_buffer = [[] for _ in range(self.n_envs)]  
        self.env_reward_buffer = [[] for _ in range(self.n_envs)]  

    def get_saved_trajectories(self, ):
        return self.episode_buffer

    def run(self, policy, wandb_run=None):
        """
        Run evaluation with the given policy and return logging data.

        Args:
            policy: Policy to evaluate
            wandb_run: Wandb run object (or None if wandb disabled)

        Returns:
            dict: JSON-serializable logging data (videos excluded)
        """

        self._prepare_run()  # Prepare per-run state (not global state!)
        self._clean_buffer()

        obs = self.env.reset() # return (n_envs, obs_dict)

        self.past_action = None
        # policy.reset()  # Reset states for stateful policy
        done = False

        pbar = tqdm.tqdm(total=self.max_steps//self.n_action_steps, desc=f"Eval PushTImageRunner",
                             leave=False)
        
        if self.enable_render: 
            self.env.start_recording()

        # envs_remained = self.n_envs
            
        try:
            active_envs_idx = list(range(self.n_envs)) # current set up for full active envs for predicting the action
            is_done = False
            while not is_done:
                obs_dict = dict(obs)

                if 'envs_idx' in obs_dict:
                    del obs_dict['envs_idx']

                with torch.no_grad():
                    result = policy.predict_action(obs_dict, recording_diffusion=self.episode_recording)

                    if isinstance(result, dict):
                        action_dict = result
                        current_diffusion_buffer = result.get('action_diffusion_buffer', None)
                    else:
                        # Backward compatibility: handle old tuple return format
                        action_dict = result
                        current_diffusion_buffer = None

                if isinstance(action_dict, dict):
                    action = action_dict['action']
                else:
                    action = action_dict

                # Store per-environment observations and diffusion actions before stepping
                if self.episode_recording and current_diffusion_buffer is not None:
                    for local_idx, env_idx in enumerate(active_envs_idx):
                        # Store unnormalized observations (original from obs_dict before normalization in policy)
                        self.env_obs_buffer[env_idx].append({
                            'image': obs['image'][local_idx],  # Keep on GPU
                            'agent_pos': obs['agent_pos'][local_idx]
                        })
                        # Store normalized diffusion actions (diff_steps+1, horizons, Da) on GPU
                        self.diffusion_action_buffer[env_idx].append(current_diffusion_buffer[local_idx])

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

                obs, reward, done, info = self.env.step(Active_action)

                # Store rewards for active environments after stepping
                if self.episode_recording:
                    for local_idx, env_idx in enumerate(active_envs_idx):
                        self.env_reward_buffer[env_idx].append(reward[env_idx])  # Keep on GPU

                # Update timestep tracking
                self.global_timestep += 1
                # for env_idx in active_envs_idx:
                #     self.episode_timesteps[env_idx] += 1

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
            self.env.start(n_envs=self.n_envs, env_separate=False, show_interact_viewer = False) # The mac does not support separate the env for rendering

        print(f"------SETUP COMPLETE!------\
              \n Configuration:  n_test={self.n_test},  n_train={self.n_train}, \
              n_envs={self.n_envs}\n max_steps={self.max_steps}")

    def _prepare_run(self):
        """
        Prepare per-run state (called before each run()).
        Does NOT modify global state like seeds or video directories.
        """
        # Update env_seeds based on current seed values (which increment between runs)
        self.env_seeds = np.concatenate((
            np.arange(start=self.seed_train, stop=self.seed_train + self.n_train),
            np.arange(start=self.seed_test, stop=self.seed_test + self.n_test)
        ), axis=0)

        # Seed the environments
        self.env.seed(self.env_seeds)

        # Setup video directories and paths (only if rendering enabled)
        if self.enable_render:
            if not self._video_dirs_initialized:
                # One-time: Create directory structure
                self.video_manager.setup_directories()
                self._video_dirs_initialized = True
                print(f"✓ Video dirs: {self.video_manager.train_video_dir.parent}")

            # Every run: Update paths with current seeds
            self.video_manager.update_file_paths(self.seed_train, self.seed_test)

        # Reset per-run state
        self.info = None
        self.episode_obs = None
        self.episode_reward = [[] for _ in range(self.n_envs)]
        self.episode_info = dict()
        self.max_reward = [None for _ in range(self.n_envs)]
        self.completion_timesteps = [None] * self.n_envs  # CRITICAL: Reset completion timesteps!
        self.global_timestep = 0  # Reset global timestep counter

        # Reset manual assistant buffers
        if self.enable_manual_assistant:
            self.agent_pos_history = [collections.deque(maxlen=self.stagnation_steps) for _ in range(self.n_envs)]
            self.object_pose_history = [collections.deque(maxlen=self.stagnation_steps) for _ in range(self.n_envs)]


    def _process_generated_videos(self):
        """Process generated videos using VideoManager."""
        return self.video_manager.process_generated_videos()

    def _cleanup_temp_files(self):
        """Clean up temporary video files using VideoManager."""
        self.video_manager.cleanup_temp_files()

    def _detect_stagnation(self, env_idx, current_agent_pos, current_obj_pose):
        """
        Detect if an environment is stagnant (robot or object not moving).

        Args:
            env_idx: Environment index
            current_agent_pos: Current agent position tensor (2D)
            current_obj_pose: Current object pose tensor (x, y, z, roll, pitch, yaw)

        Returns:
            bool: True if stagnation detected
        """
        if not self.enable_manual_assistant:
            return False

        # Extract XY position and yaw for object (ignoring z, roll, pitch for 2D scenario)
        obj_xy_yaw = torch.tensor([
            current_obj_pose[0].item(),  # x
            current_obj_pose[1].item(),  # y
            current_obj_pose[5].item(),  # yaw
        ], device=current_obj_pose.device)

        # Update histories
        self.agent_pos_history[env_idx].append(current_agent_pos.clone())
        self.object_pose_history[env_idx].append(obj_xy_yaw)

        # Need full history buffer to detect stagnation
        if len(self.agent_pos_history[env_idx]) < self.stagnation_steps:
            return False

        # Check agent stagnation: max movement over N steps < threshold
        agent_positions = torch.stack(list(self.agent_pos_history[env_idx]), dim=0)  # (N, 2)
        agent_displacements = torch.diff(agent_positions, dim=0)  # (N-1, 2)
        agent_distances = torch.norm(agent_displacements, dim=1)  # (N-1,)
        max_agent_movement = agent_distances.max().item()

        # Check object stagnation: max movement over N steps < threshold (XY + yaw)
        obj_poses = torch.stack(list(self.object_pose_history[env_idx]), dim=0)  # (N, 3)
        obj_xy_displacements = torch.diff(obj_poses[:, :2], dim=0)  # (N-1, 2) - only XY
        obj_distances = torch.norm(obj_xy_displacements, dim=1)  # (N-1,)
        max_obj_movement = obj_distances.max().item()

        # Stagnation detected if either condition is met
        agent_stagnant = max_agent_movement < self.agent_pos_threshold
        object_stagnant = max_obj_movement < self.object_pos_threshold

        return agent_stagnant or object_stagnant

    def _manual_reset_env(self, envs_idx, info):
        """
        Manually reset the T-pose for a stagnant environment, and return the corresponding new obs 
        when resetting the specific env. 
        """

        # set them back to homepos which might be far away from the cur_pos(or change into four corners version with better ? check )  
        self.env.robot.set_dofs_position(
            envs_idx = envs_idx, 
            position=self.env.home_pos,
            dofs_idx_local=self.env.robot_dofs_idx[0:7],
        )

        self.env.robot.control_dofs_position(
            envs_idx = envs_idx, 
            position=self.env.home_pos,
            dofs_idx_local=self.env.robot_dofs_idx[0:7],
        )

        _, reset_image, reset_agent_pos = self.env._get_cur_obs(envs_idx=envs_idx)
        self.agent_pos_history[envs_idx].clear()
        self.object_pose_history[envs_idx].clear()

        print(f"[Manual Assistant] Env {envs_idx} reset due to stagnation")
        return reset_image, reset_agent_pos

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
        manual_reset_indices = []  # Track envs needing manual reset

        for i in range(self.n_envs):
            if reward[i] is None:
                raise ValueError("Reward saving error!")
            self.episode_reward[i].append(reward[i].cpu().numpy())

            if env_status[i] == 2:  # Still active
                # Check for stagnation before adding to active list
                if self.enable_manual_assistant and 'agent_pos' in info and 'cur_Tpose' in info:
                    is_stagnant = self._detect_stagnation(
                        i,
                        info['agent_pos'][i],
                        info['cur_Tpose'][i]
                    )

                    if is_stagnant:
                        # Mark for manual reset, still add to active envs 
                        # reset the original obs_dict observation after resetting them to the home_pos
                        manual_reset_indices.append(i)
                        reset_img, reset_agent_pos = self._manual_reset_env(i, info)
                        obs['image'] = reset_img
                        obs['agent_pos'] = reset_agent_pos

                    active_env_indices.append(i)
                else:
                    active_env_indices.append(i)

            elif env_status[i] == 3:  # Natural success
                # Record per-env timestep when this environment succeeded
                # Convert action steps to actual environment steps
                self.completion_timesteps[i] = self.global_timestep * self.n_action_steps

                for key in self.info.keys():
                    if key == 'envs_idx': continue
                    self.info[key][i] = info[key][i]

                # Transfer completed environment data to episode_buffer
                if self.episode_recording:
                    self._transfer_env_to_episode_buffer(i)

            elif env_status[i] == 1:  # Truncated by max_steps
                # Also transfer truncated episode data to episode_buffer
                if self.episode_recording:
                    self._transfer_env_to_episode_buffer(i)

            # env_status[i] == 0 (already done): do nothing, already transferred

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

    def _transfer_env_to_episode_buffer(self, env_idx):
        """
        Transfer completed environment data to the global episode_buffer.

        Args:
            env_idx: Index of the environment that just finished
        """
        if len(self.env_obs_buffer[env_idx]) == 0:
            # No data to transfer (shouldn't happen, but safeguard)
            return

        # Stack observations for this episode
        # env_obs_buffer[env_idx] is a list of dicts, each containing 'image' and 'agent_pos'
        episode_images = torch.stack([obs['image'] for obs in self.env_obs_buffer[env_idx]], dim=0)
        episode_agent_pos = torch.stack([obs['agent_pos'] for obs in self.env_obs_buffer[env_idx]], dim=0)

        # Stack diffusion actions for this episode (already normalized, from policy)
        # Shape per step: (diff_steps+1, horizons, Da)
        episode_actions = torch.stack(self.diffusion_action_buffer[env_idx], dim=0)

        # Stack rewards for this episode
        episode_rewards = torch.stack(self.env_reward_buffer[env_idx], dim=0)

        # Append to global episode_buffer (keep on GPU)
        # Note: These lists will accumulate data from all completed episodes
        if len(self.episode_buffer['obs']['img']) == 0:
            # First episode - directly assign
            self.episode_buffer['obs']['img'] = episode_images
            self.episode_buffer['obs']['agent_pos'] = episode_agent_pos
            self.episode_buffer['action'] = episode_actions
            self.episode_buffer['reward'] = episode_rewards
        else:
            # Concatenate along the step dimension (global_steps)
            self.episode_buffer['obs']['img'] = torch.cat([
                self.episode_buffer['obs']['img'], episode_images
            ], dim=0)
            self.episode_buffer['obs']['agent_pos'] = torch.cat([
                self.episode_buffer['obs']['agent_pos'], episode_agent_pos
            ], dim=0)
            self.episode_buffer['action'] = torch.cat([
                self.episode_buffer['action'], episode_actions
            ], dim=0)
            self.episode_buffer['reward'] = torch.cat([
                self.episode_buffer['reward'], episode_rewards
            ], dim=0)

        # Record the cumulative global end index for this episode
        # This is the total number of steps accumulated so far across all episodes
        current_global_steps = self.episode_buffer['obs']['img'].shape[0]
        self.episode_buffer['episode_ends'].append(current_global_steps)

        # Clear this environment's temporary buffers for next episode
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
        