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

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from collections import defaultdict
from genesis_ILDP.policy.dummy_policy import DummyPolicy

base_video_path = target_folder1

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
        self.file_path = list()
        self.base_generate_path = None
        self.enable_render = enable_render
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env_seeds = None
        self.info = None
        self.done_ratio = done_ratio

        self._setup_envs()

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
                        done_ratio=done_ratio
                    ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                n_envs = self.n_envs, 
                episode_recording = self.episode_recording
            )

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

    def run(self, policy):
        """Run evaluation with the given policy and return logging data."""
        
        self._prepare_env()
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
                    if not self.episode_recording:
                        action_dict = policy.predict_action(obs_dict)
                        current_diffusion_buffer = None
                    else:
                        action_dict, current_diffusion_buffer = policy.predict_action(obs_dict, recording_diffusion = True)

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
                self.env.stop_recording(self.base_generate_path)
                self._process_generated_videos()
                self._cleanup_temp_files()
            except Exception as e:
                print(f"Warning: Video processing failed: {e}")

        # Update seeds for next run
        self.seed_train = self.seed_train + self.n_train
        self.seed_test = self.seed_test + self.n_test

        # Create and return logging data
        log_data = self._create_log_data()

        # Print summary statistics (aligned with eval.py format)
        print("\n=== Evaluation Results Summary ===")
        for prefix in ['test/', 'train/']:
            prefix_name = prefix.rstrip('/')
            if f'{prefix}reward_mean' in log_data:
                # Calculate episode counts for debugging
                if prefix == 'train/':
                    total = self.n_train
                    successful = sum(1 for i in range(self.n_train) if self.completion_timesteps[i] is not None)
                else:
                    total = self.n_test
                    successful = sum(1 for i in range(self.n_train, self.n_envs) if self.completion_timesteps[i] is not None)

                print(f"{prefix_name.capitalize()}: {successful}/{total} succeeded")
                print(f"  Reward: {log_data[f'{prefix}reward_mean']:.3f} ± {log_data.get(f'{prefix}reward_std', 0):.3f}")
                if f'{prefix}completion_mean' in log_data:
                    print(f"  Completion: {log_data[f'{prefix}completion_mean']:.1f} ± {log_data.get(f'{prefix}completion_std', 0):.1f} steps")
                if f'{prefix}success_rate' in log_data:
                    print(f"  Success Rate: {log_data[f'{prefix}success_rate']:.1%}")

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
        # TODO could add an api function integrating the full envs control inside the wrapper without referring back to the original env function 
        self.env.start(n_envs=self.n_envs, env_separate=False, show_interact_viewer = False)
        print(f"------SETUP COMPLETE!------\
              \n Configuration:  n_test={self.n_test},  n_train={self.n_train}, \
              n_envs={self.n_envs}\n max_steps={self.max_steps}")

    def _prepare_env(self, enable_render=True):

        self.env_seeds = np.concatenate((
            np.arange(start=self.seed_train, stop=self.seed_train + self.n_train),
            np.arange(start=self.seed_test, stop=self.seed_test + self.n_test)
        ), axis=0)
        
        self.env.seed(self.env_seeds)
        
        if enable_render:
            timestamp = time.strftime("%Y%m%d-%H-%M")
            train_video_dir = Path(base_video_path) / 'train' / timestamp
            test_video_dir = Path(base_video_path) / 'test' / timestamp
            raw_video_dir = Path(base_video_path) / 'raw' / timestamp
            
            train_video_dir.mkdir(parents=True, exist_ok=True)
            test_video_dir.mkdir(parents=True, exist_ok=True)
            raw_video_dir.mkdir(parents=True, exist_ok=True)
            
            self.base_generate_path = str(raw_video_dir / 'PushT.mp4')
            
            self.file_path = []
            for i in range(self.n_train):
                target_path = train_video_dir / f'PushT-seed{self.seed_train + i}.mp4'
                self.file_path.append(str(target_path))
                
            for i in range(self.n_test):
                target_path = test_video_dir / f'PushT-seed{self.seed_test + i}.mp4'
                self.file_path.append(str(target_path))

            self.info = None
            self.episode_obs = None
            self.episode_reward = [[] for _ in range(self.n_envs)]
            self.episode_info = dict()
            self.max_reward = [None for _ in range(self.n_envs)]


    def _process_generated_videos(self):
    # Only work if render videos
        if self.base_generate_path is None:
            raise ValueError('Video Generated Error!')
        
        base_path = Path(self.base_generate_path)
        base_dir = base_path.parent
        base_name = base_path.stem 
        
        moved_files = []
        
        for i in range(self.n_envs):
            generated_file = base_dir / f'{base_name}_{i}.mp4'
            
            if generated_file.exists() and i < len(self.file_path):
                target_file = Path(self.file_path[i])
                try:
                    
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(generated_file), str(target_file))
                    moved_files.append(str(target_file))
                    print(f"Moved: {generated_file} -> {target_file}")
                    
                except Exception as e:
                    print(f"Error moving {generated_file} to {target_file}: {e}")
                    raise ValueError("Video Generation error!") 
            else:
                print(f"Generated file not found: {generated_file}")
                raise ValueError("Video Generation error!") 
        
        return moved_files

    def _cleanup_temp_files(self):
        if self.base_generate_path is None:
            return
        base_path = Path(self.base_generate_path)
        base_dir = base_path.parent
        base_name = base_path.stem
        
        for temp_file in base_dir.glob(f'{base_name}_*.mp4'):
            try:
                temp_file.unlink()
                print(f"Cleaned up: {temp_file}")
            except Exception as e:
                print(f"Error cleaning {temp_file}: {e}")

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
            if reward[i] is None:
                raise ValueError("Reward saving error!")
            self.episode_reward[i].append(reward[i].cpu().numpy())

            if env_status[i] == 2:  # Still active
                active_env_indices.append(i)

            elif env_status[i] == 3:  # Natural success
                # Record per-env timestep when this environment succeeded
                self.completion_timesteps[i] = self.global_timestep

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
        """Create logging data from collected episode information.
        Aligned with eval.py format for consistency.
        """
        max_rewards = collections.defaultdict(list)
        completion_times = collections.defaultdict(list)
        log_data = dict()

        # Collect per-environment data
        for i in range(self.n_envs):
            seed = self.env_seeds[i]
            if i < self.n_train:
                prefix = 'train/'
                should_upload_video = i < self.n_train_vis
            else:
                prefix = 'test/'
                test_idx = i - self.n_train
                should_upload_video = test_idx < self.n_test_vis

            # Calculate max reward for this episode
            if len(self.episode_reward[i]) > 0:
                max_reward = np.max(np.array(self.episode_reward[i]))
            else:
                max_reward = 0.0

            max_rewards[prefix].append(max_reward)

            # Track completion timesteps only for successful episodes
            if self.completion_timesteps[i] is not None:
                completion_times[prefix].append(self.completion_timesteps[i])

            # Add video logs if available
            if should_upload_video and self.file_path is not None and i < len(self.file_path):
                video_path = self.file_path[i]
                if Path(video_path).exists():
                    try:
                        sim_video = wandb.Video(video_path)
                        log_data[prefix + f'sim_video_{seed}'] = sim_video
                    except Exception as e:
                        print(f"Warning: Failed to create video log for {video_path}: {e}")

        # Calculate aggregate statistics per prefix (train/test)
        for prefix in ['train/', 'test/']:
            if prefix not in max_rewards:
                continue

            rewards = np.array(max_rewards[prefix])
            times = np.array(completion_times[prefix]) if prefix in completion_times else np.array([])

            # Reward statistics (aligned with eval.py)
            if len(rewards) > 0:
                reward_mean = float(np.mean(rewards))
                reward_std = float(np.std(rewards))
                reward_min = float(np.min(rewards))
                reward_max = float(np.max(rewards))
                reward_median = float(np.median(rewards))

                # Format with / for runner compatibility
                log_data[prefix + 'reward_mean'] = reward_mean
                log_data[prefix + 'reward_std'] = reward_std
                log_data[prefix + 'reward_min'] = reward_min
                log_data[prefix + 'reward_max'] = reward_max
                log_data[prefix + 'reward_median'] = reward_median
                log_data[prefix + 'mean_score'] = reward_mean  # Backward compatibility

                # Also add format without / for eval.py compatibility
                prefix_name = prefix.rstrip('/')
                log_data[f'{prefix_name}_reward_mean'] = reward_mean
                log_data[f'{prefix_name}_reward_std'] = reward_std
                log_data[f'{prefix_name}_reward_min'] = reward_min
                log_data[f'{prefix_name}_reward_max'] = reward_max
                log_data[f'{prefix_name}_reward_median'] = reward_median

            # Completion time statistics (aligned with eval.py)
            if len(times) > 0:
                completion_mean = float(np.mean(times))
                completion_std = float(np.std(times))
                completion_min = float(np.min(times))
                completion_max = float(np.max(times))
                completion_median = float(np.median(times))

                # Format with / for runner compatibility
                log_data[prefix + 'completion_mean'] = completion_mean
                log_data[prefix + 'completion_std'] = completion_std
                log_data[prefix + 'completion_min'] = completion_min
                log_data[prefix + 'completion_max'] = completion_max
                log_data[prefix + 'completion_median'] = completion_median

                # Also add format without / for eval.py compatibility
                prefix_name = prefix.rstrip('/')
                log_data[f'{prefix_name}_completion_mean'] = completion_mean
                log_data[f'{prefix_name}_completion_std'] = completion_std
                log_data[f'{prefix_name}_completion_min'] = completion_min
                log_data[f'{prefix_name}_completion_max'] = completion_max
                log_data[f'{prefix_name}_completion_median'] = completion_median

            # Success rate (aligned with eval.py)
            total_episodes = len(rewards)
            successful_episodes = len(times)
            success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0

            # Format with / for runner compatibility
            log_data[prefix + 'success_rate'] = float(success_rate)

            # Also add format without / for eval.py compatibility
            prefix_name = prefix.rstrip('/')
            log_data[f'{prefix_name}_success_rate'] = float(success_rate)

        return log_data

if __name__ == "__main__":
        # use collect_data policy to check availability
        output_dir = ''
        Runner = PushTImageRunner(output_dir)
        policy = DummyPolicy(Runner.n_action_steps)
        Runner.run(policy)
        