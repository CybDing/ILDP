"""Reusable utilities for evaluation video processing and logging."""

from pathlib import Path
import shutil
import time
import numpy as np
import collections
import wandb
import json


class VideoManager:
    """Manages video recording paths and processing for evaluation."""

    def __init__(self, base_video_path, n_train, n_test, n_train_vis, n_test_vis,
                 train_start_seed, test_start_seed):
        
        self.base_video_path = Path(base_video_path)
        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = min(n_train, n_train_vis)
        self.n_test_vis = min(n_test, n_test_vis)
        self.train_start_seed = train_start_seed
        self.test_start_seed = test_start_seed

        self.train_video_dir = None
        self.test_video_dir = None
        self.raw_video_dir = None
        self.base_generate_path = None
        self.file_paths = []

    def setup_directories(self, timestamp=None, train_seed_base=None, test_seed_base=None):
        """Create video directories with timestamp and prepare file paths."""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d-%H-%M")

        if train_seed_base is not None:
            self.train_start_seed = train_seed_base
        if test_seed_base is not None:
            self.test_start_seed = test_seed_base

        self.train_video_dir = self.base_video_path / 'train' / timestamp
        self.test_video_dir = self.base_video_path / 'test' / timestamp
        self.raw_video_dir = self.base_video_path / 'raw' / timestamp

        self.train_video_dir.mkdir(parents=True, exist_ok=True)
        self.test_video_dir.mkdir(parents=True, exist_ok=True)
        self.raw_video_dir.mkdir(parents=True, exist_ok=True)

        # Base path for generated videos (will have _0, _1, etc. appended)
        self.base_generate_path = str(self.raw_video_dir / 'PushT.mp4')

        # Prepare target file paths
        self.file_paths = []
        for i in range(self.n_train):
            target_path = self.train_video_dir / f'PushT-seed{self.train_start_seed + i}.mp4'
            self.file_paths.append(str(target_path))

        for i in range(self.n_test):
            target_path = self.test_video_dir / f'PushT-seed{self.test_start_seed + i}.mp4'
            self.file_paths.append(str(target_path))

        return self.base_generate_path

    def update_file_paths(self, train_seed_base, test_seed_base):
        """
        Update file paths with new seed bases (called before each run).
        Keeps same directories, just updates the seed numbers in filenames.
        """
        self.train_start_seed = train_seed_base
        self.test_start_seed = test_seed_base

        # Regenerate file paths with new seeds
        self.file_paths = []
        for i in range(self.n_train):
            self.file_paths.append(str(self.train_video_dir / f'PushT-seed{train_seed_base + i}.mp4'))
        for i in range(self.n_test):
            self.file_paths.append(str(self.test_video_dir / f'PushT-seed{test_seed_base + i}.mp4'))

    def process_generated_videos(self):
        """Move videos from raw folder to train/test folders and delete unwanted ones."""
        if self.base_generate_path is None:
            raise ValueError('Video directories not set up! Call setup_directories() first.')

        base_path = Path(self.base_generate_path)
        base_dir = base_path.parent
        base_name = base_path.stem

        moved_files = []
        n_envs = self.n_train + self.n_test

        for i in range(n_envs):
            generated_file = base_dir / f'{base_name}_{i}.mp4'

            # Determine if this video should be kept
            should_keep = False
            if i < self.n_train:
                should_keep = (i < self.n_train_vis)
            else:
                test_idx = i - self.n_train
                should_keep = (test_idx < self.n_test_vis)

            if generated_file.exists():
                if should_keep and i < len(self.file_paths):
                    # Move video to target folder
                    target_file = Path(self.file_paths[i])
                    try:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(generated_file), str(target_file))
                        moved_files.append(str(target_file))
                        print(f"Moved: {generated_file} -> {target_file}")
                    except Exception as e:
                        print(f"Error moving {generated_file} to {target_file}: {e}")
                        raise ValueError("Video Generation error!")
                else:
                    # Delete unwanted video
                    try:
                        generated_file.unlink()
                        print(f"Deleted: {generated_file}")
                    except Exception as e:
                        print(f"Warning: Could not delete {generated_file}: {e}")
            else:
                if should_keep:
                    print(f"Warning: Expected video not found: {generated_file}")

        return moved_files

    def cleanup_temp_files(self):
        """Clean up any remaining temporary video files in raw directory."""
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


class EvalLogManager:
    """Manages logging data creation for evaluation results."""

    def __init__(self, n_train, n_test, n_train_vis, n_test_vis):
        """
        Args:
            n_train: Number of training environments
            n_test: Number of test environments
            n_train_vis: Number of training videos to log to wandb
            n_test_vis: Number of test videos to log to wandb
        """
        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = n_train_vis
        self.n_test_vis = n_test_vis

    def create_log_data(self, episode_rewards, completion_timesteps, env_seeds, file_paths,):
        """
        Create logging data from evaluation results.

        Args:
            episode_rewards: List[List[float]] - rewards per step for each environment
            completion_timesteps: List[int or None] - timestep when each env completed (None if failed)
            env_seeds: np.ndarray - seed for each environment
            file_paths: List[str] - video file paths for each environment
        
        Returns:
            dict: Logging data compatible with wandb (or JSON if include_videos=False)
        """
        max_rewards = collections.defaultdict(list)
        completion_times = collections.defaultdict(list)
        log_data = dict()

        n_envs = self.n_train + self.n_test

        # Collect per-environment data
        for i in range(n_envs):
            seed = env_seeds[i]

            # Determine prefix and whether to upload video
            if i < self.n_train:
                prefix = 'train'
                env_id = i
                should_upload_video = i < self.n_train_vis
            else:
                prefix = 'test'
                env_id = i - self.n_train
                should_upload_video = env_id < self.n_test_vis

            # Calculate max reward for this episode
            if len(episode_rewards[i]) > 0:
                max_reward = float(np.max(np.array(episode_rewards[i])))
            else:
                max_reward = 0.0

            max_rewards[prefix].append(max_reward)

            # Store reward directly at root level with seed in key name
            log_data[f'{prefix}/sim_max_reward_{seed}'] = max_reward

            # Track completion timesteps only for successful episodes
            if completion_timesteps[i] is not None:
                completion_times[prefix].append(completion_timesteps[i])

            # Validate video files exist (but don't add to log_data for JSON compatibility)
            # Videos should be logged to wandb separately through wandb.log() in your training script
            if should_upload_video and file_paths is not None and i < len(file_paths):
                video_path = file_paths[i]
                video_file = Path(video_path)

                if video_file.exists() and video_file.is_file() and video_file.suffix == '.mp4':
                    print(f"✓ Video ready: {video_path}")
                elif not video_file.exists():
                    print(f"✗ Video file not found: {video_path}")
                else:
                    print(f"✗ Invalid video file: {video_path}")

        # Calculate aggregate statistics per prefix (train/test)
        for prefix in ['train', 'test']:
            if prefix not in max_rewards:
                continue

            rewards = np.array(max_rewards[prefix])
            times = np.array(completion_times[prefix]) if prefix in completion_times else np.array([])

            # Reward statistics
            if len(rewards) > 0:
                log_data.update(self._create_reward_stats(prefix, rewards))

            # Completion time statistics
            if len(times) > 0:
                log_data.update(self._create_completion_stats(prefix, times))

            # Success rate
            total_episodes = len(rewards)
            successful_episodes = len(times)
            success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
            log_data[f'{prefix}/success_rate'] = float(success_rate)

        return log_data

    def _create_reward_stats(self, prefix, rewards):
        """Create reward statistics for logging."""
        stats = {}

        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        reward_min = float(np.min(rewards))
        reward_max = float(np.max(rewards))
        reward_median = float(np.median(rewards))

        stats[f'{prefix}/reward_mean'] = reward_mean
        stats[f'{prefix}/reward_std'] = reward_std
        stats[f'{prefix}/reward_min'] = reward_min
        stats[f'{prefix}/reward_max'] = reward_max
        stats[f'{prefix}/reward_median'] = reward_median
        stats[f'{prefix}/mean_score'] = reward_mean 

        return stats

    def _create_completion_stats(self, prefix, times):
        """Create completion time statistics for logging."""
        stats = {}

        completion_mean = float(np.mean(times))
        completion_std = float(np.std(times))
        completion_min = float(np.min(times))
        completion_max = float(np.max(times))
        completion_median = float(np.median(times))

        stats[f'{prefix}/completion_mean'] = completion_mean
        stats[f'{prefix}/completion_std'] = completion_std
        stats[f'{prefix}/completion_min'] = completion_min
        stats[f'{prefix}/completion_max'] = completion_max
        stats[f'{prefix}/completion_median'] = completion_median

        return stats

    def create_wandb_videos(self, env_seeds, file_paths, wandb_enabled=True):
        """
        Create wandb.Video objects for logging (separate from JSON data).

        Args:
            env_seeds: np.ndarray - seed for each environment
            file_paths: List[str] - video file paths for each environment
            wandb_enabled: bool - whether wandb logging is enabled (set False when mode='disabled')

        Returns:
            dict: Dictionary of wandb.Video objects keyed by 'prefix/sim_video_seed'
                  Returns empty dict if wandb_enabled=False
        """
        video_dict = {}

        # Skip video creation if wandb is disabled
        if not wandb_enabled:
            print("⊘ Wandb disabled - skipping video object creation")
            return video_dict

        n_envs = self.n_train + self.n_test

        for i in range(n_envs):
            seed = env_seeds[i]

            # Determine prefix and whether to upload video
            if i < self.n_train:
                prefix = 'train'
                should_upload_video = i < self.n_train_vis
            else:
                prefix = 'test'
                test_idx = i - self.n_train
                should_upload_video = test_idx < self.n_test_vis

            # Create wandb.Video object if conditions are met
            if should_upload_video and file_paths is not None and i < len(file_paths):
                video_path = file_paths[i]
                video_file = Path(video_path)

                if video_file.exists() and video_file.is_file() and video_file.suffix == '.mp4':
                    try:
                        # Preserve original video dimensions (don't let wandb resize)
                        # format="mp4" ensures proper encoding
                        sim_video = wandb.Video(video_path, format="mp4")
                        video_dict[f'{prefix}/sim_video_{seed}'] = sim_video
                        print(f"✓ Video added to wandb log: {video_path}")
                    except Exception as e:
                        print(f"✗ Failed to create wandb.Video for {video_path}: {e}")

        return video_dict

    @staticmethod
    def print_summary(log_data, n_train, n_test, completion_timesteps):
        """Print evaluation results summary."""
        print("\n=== Evaluation Results Summary ===")
        for prefix in ['test/', 'train/']:
            prefix_name = prefix.rstrip('/')
            if f'{prefix}reward_mean' in log_data:
                # Calculate episode counts
                if prefix == 'train/':
                    total = n_train
                    successful = sum(1 for i in range(n_train) if completion_timesteps[i] is not None)
                else:
                    total = n_test
                    successful = sum(1 for i in range(n_train, n_train + n_test)
                                   if completion_timesteps[i] is not None)

                print(f"{prefix_name.capitalize()}: {successful}/{total} succeeded")
                print(f"  Reward: {log_data[f'{prefix}reward_mean']:.3f} ± {log_data.get(f'{prefix}reward_std', 0):.3f}")
                if f'{prefix}completion_mean' in log_data:
                    print(f"  Completion: {log_data[f'{prefix}completion_mean']:.1f} ± {log_data.get(f'{prefix}completion_std', 0):.1f} steps")
                if f'{prefix}success_rate' in log_data:
                    print(f"  Success Rate: {log_data[f'{prefix}success_rate']:.1%}")
