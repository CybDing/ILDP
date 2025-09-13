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
from genesis_ILDP.gym_util.async_vector_env import AsyncVectorEnv
from genesis_ILDP.gym_util.multistep_wrapper_parallel import MultiStepWrapper

from genesis_ILDP.utils.pytorch_util import dict_apply
from genesis_ILDP.env_runner.base_image_runner import BaseImageRunner

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from collections import defaultdict
from genesis_ILDP.policy.test_policy import TestPolicy

base_video_path = target_folder1


class PushTImageRunnerBatched(BaseImageRunner):
    """
    Batched runner with auto-reset functionality for continuous training.
    
    Key Features:
    - Auto-resets terminated environments to maintain constant batch size
    - Tracks episode completion and collects episode statistics
    - Handles environment lifecycle with proper information storage
    - Supports continuous video recording across episode boundaries
    """
    
    def __init__(self, 
                 output_dir,
                 n_train = 0,
                 n_train_vis = 0,
                 n_test = 50,
                 n_test_vis = 50,
                 n_obs_steps = 2,
                 n_action_steps = 8,
                 max_steps=200,
                 image_shape=(96, 96),
                 tqdm_interval_sec=1.0,
                 n_envs = None,
                 fps = 20,
                 past_action=False,
                 train_start_seed = 0,
                 test_start_seed = 100000,
                 enable_render = True,
                 max_episodes_per_env = 10,  # New: episodes per environment
                 ):
        super().__init__(output_dir)
        
        # Environment configuration
        if n_envs is None:
            self.n_envs = n_train + n_test
            print(f"Using computed n_envs: {self.n_envs} (train: {n_train}, test: {n_test})")
        else:
            self.n_envs = n_envs
            print(f"Using provided n_envs: {self.n_envs}")

        # MultiStepWrapper setup - all environments run in parallel
        self.env = MultiStepWrapper(
            PushTEnv(
                render_size=image_shape,
                fps=fps,
                show_fps=False
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            n_envs=self.n_envs  # Use ALL environments
        )
        
        # Configuration
        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = min(self.n_train, n_train_vis)
        self.n_test_vis = min(self.n_test, n_test_vis)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.past_action = past_action
        self.seed_train = train_start_seed
        self.seed_test = test_start_seed
        self.enable_render = enable_render
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_episodes_per_env = max_episodes_per_env
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Episode tracking for auto-reset
        self.completed_episodes = [[] for _ in range(self.n_envs)]  # Store completed episode data
        self.current_episode_rewards = [[] for _ in range(self.n_envs)]  # Current episode rewards
        self.current_episode_steps = [0] * self.n_envs  # Steps in current episode
        self.total_episodes_completed = [0] * self.n_envs  # Total episodes per env
        
        # Video recording
        self.file_path = []
        self.base_generate_path = None
        
        self._setup_envs()

    def run(self, policy, max_total_episodes=None):
        """
        Run evaluation with auto-reset functionality.
        
        Args:
            policy: Policy to evaluate
            max_total_episodes: Maximum total episodes across all environments
        """
        self._prepare_env()
        obs = self.env.reset()
        past_action = None
        policy.reset()
        
        # Episode counting
        total_episodes_run = 0
        max_total_episodes = max_total_episodes or (self.max_episodes_per_env * self.n_envs)
        
        pbar = tqdm.tqdm(total=max_total_episodes, desc="Batched Episodes",
                         leave=False, mininterval=self.tqdm_interval_sec)
        
        if self.enable_render:
            self.env.start_recording()
            
        try:
            while total_episodes_run < max_total_episodes:
                # Prepare observation dictionary
                obs_dict = dict(obs)
                if 'envs_idx' in obs_dict:
                    del obs_dict['envs_idx']

                # Add past action if enabled
                if self.past_action and (past_action is not None):
                    obs_dict['past_action'] = past_action[
                                            :, -(self.n_obs_steps - 1):
                                            ].astype(np.float32)

                # Get policy prediction
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    
                if isinstance(action_dict, dict):
                    action = action_dict['action']
                else:
                    action = action_dict

                # Execute action in environment
                obs, reward, done, info, env_status = self.env.step(action)
                
                # Update past action
                past_action = action
                
                # Process environment information and handle episode completion
                completed_episodes_this_step = self._process_env_lifecycle(
                    obs, reward, info, env_status)
                
                # Update progress
                total_episodes_run += completed_episodes_this_step
                pbar.update(completed_episodes_this_step)
                
                if total_episodes_run >= max_total_episodes:
                    break
                    
        except Exception as e:
            print(f"Error during policy execution: {e}")
            raise e
        finally:
            pbar.close()

        # Handle video recording cleanup
        if self.enable_render:
            try:
                self.env.stop_recording(self.base_generate_path)
                self._process_generated_videos()
                self._cleanup_temp_files()
            except Exception as e:
                print(f"Warning: Video processing failed: {e}")

        # Create and return logging data
        log_data = self._create_log_data()
        return log_data
    
    def _process_env_lifecycle(self, obs, reward, info, env_status):
        """
        Handle environment lifecycle: episode completion, data storage, and auto-reset.
        
        Args:
            obs: Observations from environments
            reward: Rewards from environments  
            info: Info from environments
            env_status: Status array (0=inactive, 1=terminated, 2=active)
            
        Returns:
            Number of episodes completed this step
        """
        completed_episodes_this_step = 0
        terminated_envs = []
        
        # Process each environment
        for env_idx in range(self.n_envs):
            # Update current episode data
            if env_status[env_idx] > 0:  # Active or terminated
                if len(reward) > env_idx and reward[env_idx] is not None:
                    self.current_episode_rewards[env_idx].append(reward[env_idx])
                    self.current_episode_steps[env_idx] += 1
            
            # Handle terminated environments
            if env_status[env_idx] == 1:  # Terminated
                terminated_envs.append(env_idx)
                
                # Store episode completion data BEFORE reset
                episode_data = {
                    'env_idx': env_idx,
                    'episode_num': self.total_episodes_completed[env_idx],
                    'total_reward': sum(self.current_episode_rewards[env_idx]),
                    'max_reward': max(self.current_episode_rewards[env_idx]) if self.current_episode_rewards[env_idx] else 0.0,
                    'episode_length': self.current_episode_steps[env_idx],
                    'final_info': info[env_idx] if hasattr(info, '__getitem__') else info,
                    'seed': self._get_env_seed(env_idx),
                    'completion_time': time.time()
                }
                
                self.completed_episodes[env_idx].append(episode_data)
                self.total_episodes_completed[env_idx] += 1
                completed_episodes_this_step += 1
                
                print(f"Episode completed - Env {env_idx}: "
                      f"Reward={episode_data['total_reward']:.3f}, "
                      f"Steps={episode_data['episode_length']}")
                
                # Reset episode tracking for this environment
                self.current_episode_rewards[env_idx] = []
                self.current_episode_steps[env_idx] = 0
        
        # Auto-reset terminated environments
        if terminated_envs:
            self._auto_reset_environments(terminated_envs)
        
        # Process active observations for next iteration
        active_obs = self._extract_active_observations(obs, env_status)
        
        return completed_episodes_this_step
    
    def _auto_reset_environments(self, terminated_env_indices):
        """
        Automatically reset terminated environments to maintain continuous operation.
        
        Args:
            terminated_env_indices: List of environment indices to reset
        """
        if not terminated_env_indices:
            return
            
        print(f"Auto-resetting environments: {terminated_env_indices}")
        
        # Convert to tensor for Genesis API
        reset_envs_tensor = to_torch(np.array(terminated_env_indices))
        
        # Reset specific environments using Genesis reset_idx
        try:
            reset_obs = self.env.env.reset_idx(reset_envs_tensor)
            
            # Update MultiStepWrapper observation buffers for reset environments
            for i, env_idx in enumerate(terminated_env_indices):
                # Clear old observations
                self.env.obs[env_idx].clear()
                
                # Add initial observation from reset
                reset_env_obs = self.env._extract_env_data(reset_obs, i)
                self.env.obs[env_idx].append(reset_env_obs)
                
                # Reset step counts and other tracking
                self.env.step_counts[env_idx] = 0
                
            print(f"Successfully reset {len(terminated_env_indices)} environments")
            
        except Exception as e:
            print(f"Error during auto-reset: {e}")
            raise e
    
    def _extract_active_observations(self, obs, env_status):
        """
        Extract observations from active environments for next policy iteration.
        
        Args:
            obs: Raw observations 
            env_status: Environment status array
            
        Returns:
            Processed observations for active environments
        """
        active_env_indices = [i for i in range(len(env_status)) if env_status[i] == 2]
        
        if not active_env_indices:
            return {}
            
        new_obs = {}
        for key in obs.keys():
            if key == 'envs_idx':
                continue
                
            active_obs_list = []
            for i in active_env_indices:
                if obs[key][i] is not None:
                    active_obs_list.append(obs[key][i])
            
            if active_obs_list:
                try:
                    new_obs[key] = torch.stack(active_obs_list, dim=0)
                except Exception as e:
                    print(f"Warning: Could not stack observations for key {key}: {e}")
        
        return new_obs
    
    def _get_env_seed(self, env_idx):
        """Get seed for specific environment index."""
        if hasattr(self, 'env_seeds') and env_idx < len(self.env_seeds):
            return self.env_seeds[env_idx]
        return env_idx  # Fallback

    def _setup_envs(self):
        """Setup environments."""
        self.env.start(n_envs=self.n_envs, env_separate=True)
        print(f"------BATCHED SETUP COMPLETE!------"
              f"\n Configuration: n_test={self.n_test}, n_train={self.n_train}, "
              f"n_envs={self.n_envs}\n max_steps={self.max_steps}")

    def _prepare_env(self, enable_render=True):
        """Prepare environment with seeds and video recording."""
        # Setup seeds
        self.env_seeds = np.concatenate((
            np.arange(start=self.seed_train, stop=self.seed_train + self.n_train),
            np.arange(start=self.seed_test, stop=self.seed_test + self.n_test)
        ), axis=0)
        
        self.env.seed(self.env_seeds)
        
        # Setup video recording paths
        if enable_render:
            timestamp = time.strftime("%Y%m%d-%H-%M")
            train_video_dir = Path(base_video_path) / 'batched_train' / timestamp
            test_video_dir = Path(base_video_path) / 'batched_test' / timestamp
            raw_video_dir = Path(base_video_path) / 'batched_raw' / timestamp
            
            for directory in [train_video_dir, test_video_dir, raw_video_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            self.base_generate_path = str(raw_video_dir / 'PushT_Batched.mp4')
            
            # Setup video file paths for each environment
            self.file_path = []
            for i in range(self.n_train):
                target_path = train_video_dir / f'PushT-seed{self.seed_train + i}.mp4'
                self.file_path.append(str(target_path))
                
            for i in range(self.n_test):
                target_path = test_video_dir / f'PushT-seed{self.seed_test + i}.mp4'
                self.file_path.append(str(target_path))

    def _process_generated_videos(self):
        """Process and organize generated video files."""
        if self.base_generate_path is None:
            print("Warning: No base video path configured")
            return []
        
        base_path = Path(self.base_generate_path)
        base_dir = base_path.parent
        base_name = base_path.stem
        
        moved_files = []
        errors = []
        
        for i in range(self.n_envs):
            generated_file = base_dir / f'{base_name}_{i}.mp4'
            
            if generated_file.exists() and i < len(self.file_path):
                target_file = Path(self.file_path[i])
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(generated_file), str(target_file))
                    moved_files.append(str(target_file))
                    print(f"Moved video: {generated_file.name} -> {target_file}")
                except Exception as e:
                    error_msg = f"Failed to move {generated_file} to {target_file}: {e}"
                    print(f"Warning: {error_msg}")
                    errors.append(error_msg)
            else:
                if not generated_file.exists():
                    print(f"Warning: Generated video file not found: {generated_file}")
                elif i >= len(self.file_path):
                    print(f"Warning: No target path configured for env {i}")
        
        if not errors:
            print(f"Successfully processed {len(moved_files)} video files")
        
        return moved_files

    def _cleanup_temp_files(self):
        """Clean up temporary video files after processing."""
        if self.base_generate_path is None:
            return
            
        base_path = Path(self.base_generate_path)
        base_dir = base_path.parent
        base_name = base_path.stem
        
        temp_files = list(base_dir.glob(f'{base_name}_*.mp4'))
        if temp_files:
            cleaned_count = 0
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_file}: {e}")
            
            if cleaned_count > 0:
                print(f"Cleaned up {cleaned_count} temporary video files")

    def _create_log_data(self):
        """Create comprehensive logging data from all completed episodes."""
        log_data = {}
        all_train_rewards = []
        all_test_rewards = []
        
        # Process completed episodes
        total_episodes = sum(len(episodes) for episodes in self.completed_episodes)
        print(f"Creating log data from {total_episodes} completed episodes")
        
        for env_idx in range(self.n_envs):
            episodes = self.completed_episodes[env_idx]
            if not episodes:
                continue
                
            # Determine if train or test environment
            if env_idx < self.n_train:
                prefix = 'train/'
                reward_list = all_train_rewards
                should_upload_video = env_idx < self.n_train_vis
            else:
                prefix = 'test/'
                reward_list = all_test_rewards
                test_idx = env_idx - self.n_train
                should_upload_video = test_idx < self.n_test_vis
            
            # Log all episodes for this environment
            for episode_data in episodes:
                seed = episode_data['seed']
                reward = episode_data['max_reward']
                reward_list.append(reward)
                
                # Log individual episode
                log_data[f"{prefix}episode_reward_{env_idx}_{episode_data['episode_num']}"] = reward
                log_data[f"{prefix}episode_length_{env_idx}_{episode_data['episode_num']}"] = episode_data['episode_length']
            
            # Add video if available
            if should_upload_video and self.file_path and env_idx < len(self.file_path):
                video_path = self.file_path[env_idx]
                if Path(video_path).exists():
                    try:
                        sim_video = wandb.Video(video_path)
                        log_data[f"{prefix}video_env_{env_idx}"] = sim_video
                        print(f"Added video log: {prefix}video_env_{env_idx}")
                    except Exception as e:
                        print(f"Warning: Failed to create video log for {video_path}: {e}")
        
        # Calculate summary statistics
        if all_train_rewards:
            log_data['train/mean_score'] = np.mean(all_train_rewards)
            log_data['train/std_score'] = np.std(all_train_rewards)
            log_data['train/total_episodes'] = len(all_train_rewards)
            print(f"Train performance: {np.mean(all_train_rewards):.3f} ± {np.std(all_train_rewards):.3f} ({len(all_train_rewards)} episodes)")
        
        if all_test_rewards:
            log_data['test/mean_score'] = np.mean(all_test_rewards)
            log_data['test/std_score'] = np.std(all_test_rewards)
            log_data['test/total_episodes'] = len(all_test_rewards)
            print(f"Test performance: {np.mean(all_test_rewards):.3f} ± {np.std(all_test_rewards):.3f} ({len(all_test_rewards)} episodes)")
        
        # Overall statistics
        all_rewards = all_train_rewards + all_test_rewards
        if all_rewards:
            log_data['overall/mean_score'] = np.mean(all_rewards)
            log_data['overall/total_episodes'] = len(all_rewards)
        
        return log_data


if __name__ == "__main__":
    # Test the batched runner
    output_dir = ''
    runner = PushTImageRunnerBatched(
        output_dir=output_dir,
        n_test=4,
        n_train=0, 
        max_episodes_per_env=3
    )
    
    policy = TestPolicy(runner.n_action_steps)
    log_data = runner.run(policy, max_total_episodes=12)
    
    print("Batched runner completed successfully!")
    print(f"Log data keys: {list(log_data.keys())}")