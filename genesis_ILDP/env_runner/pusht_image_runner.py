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
from genesis_ILDP.gym_util.multistep_wrapper_parallel import MultiStepWrapper # new wrapper

# from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.utils.pytorch_util import dict_apply
from genesis_ILDP.env_runner.base_image_runner import BaseImageRunner

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from collections import defaultdict
from genesis_ILDP.policy.test_policy import TestPolicy

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
                 max_steps=200,
                 image_shape=(96, 96),
                 tqdm_interval_sec=1.0,
                 n_envs = None,
                 fps = 20,
                #  crf = 22, # video quality
                 past_action=False,
                 train_start_seed = 0,
                 test_start_seed = 5000, 
                 # does not reach 100000 episodes
                 enable_render = True,
                 max_envs_running = 3,
                 ):
        super().__init__(output_dir)
        if n_envs is None:
            self.n_envs = n_train + n_test
            print(f"Using computed n_envs: {self.n_envs} (train: {n_train}, test: {n_test})")
        else:
            self.n_envs = n_envs
            print(f"Using provided n_envs: {self.n_envs}")

        steps_per_render = 1 # double check!
        
        # set the max_envs for parallel envs running together inside the genesis engine for less gpu memory
        self.parallel_envs_counts = min(max_envs_running, self.n_envs) 
        self.env = MultiStepWrapper(
                    PushTEnv(
                        render_size=image_shape,
                        fps = fps,
                        show_fps=False
                    ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                n_envs = self.n_envs
            )
        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = min(self.n_train, n_train_vis)
        self.n_test_vis = min(self.n_test, n_test_vis)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_steps = max_steps
        self.past_action = past_action
        self.seed_train = train_start_seed
        self.seed_test = test_start_seed
        self.file_path = list()
        self.base_generate_path = None
        self.enable_render = enable_render
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env_seeds = None
        self.info = None

        self._setup_envs()

        self.episode_obs = None
        self.episode_reward = [[] for _ in range(self.n_envs)]
        self.episode_info = dict()
        self.max_reward = [None for _ in range(self.n_envs)]

    def run(self, policy):
        """Run evaluation with the given policy and return logging data."""
        
        self._prepare_env()
        obs = self.env.reset() # return (parallel_envs_counts, obs_dict)
        past_action = None
        # policy.reset()  # Reset states for stateful policy
        done = False

        pbar = tqdm.tqdm(total=self.max_steps * self.n_envs, desc=f"Eval PushTImageRunner",
                             leave=False, mininterval=self.tqdm_interval_sec)
        
        if self.enable_render: 
            self.env.start_recording()

        # envs_remained = self.n_envs
            
        try:
            while not done:
                obs_dict = dict(obs)
                
                if 'envs_idx' in obs_dict:
                    del obs_dict['envs_idx']

                # Add past action if enabled, and note that the past action is counted according to the n_obs_steps number
                # if n_obs_steps is larger than 2, say 3, than the past action should only terminate at one step before the full execution
                # since it start planning earlier ? 
                if self.past_action and (past_action is not None):
                    obs_dict['past_action'] = past_action[
                                            :, -(self.n_obs_steps - 1):
                                            ].astype(np.float32)

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    
                if isinstance(action_dict, dict):
                    action = action_dict['action']
                else:
                    action = action_dict

                obs, reward, done, info, env_status = self.env.step(action)
                
                past_action = action
                
                obs = self._process_info(obs, reward, info, env_status)

                pbar.update(1)  # Update by 1 step since all envs run together
                
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

        # Update seeds for next run
        self.seed_train = self.seed_train + self.n_train
        self.seed_test = self.seed_test + self.n_test

        # Create and return logging data
        log_data = self._create_log_data()
        return log_data

    def _setup_envs(self,):
        # TODO should refine this initializing process inside the wrapper function without directly callout the start function from api env
        # which might be inconsistent with current configuration 
        self.env.start(n_envs=self.n_envs, env_separate=True, show_interact_viewer = False)
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
        # save info, reward(max), process obs(number: 2) 
        # prepare obs
        new_obs = defaultdict(list)
        if self.info is None:
            self.info = info
        
        active_env_indices = []
        for i in range(self.n_envs):
            if env_status[i] == 0 and final_saving==True: # save for truncating envs
                # copy latest info
                for key in self.info.keys():
                    if key == 'envs_idx': continue
                    if reward[i] is None: raise ValueError("Reward saving error!")
                    self.info[key][i] = info[key][i]
            
            if reward[i] is None: raise ValueError("Reward saving error!")
            self.episode_reward[i].append(reward[i])

            if env_status[i] == 2: # 活跃环境
                active_env_indices.append(i)
            
            if env_status[i] == 1: # save for terminated envs
                for key in self.info.keys():
                    if key == 'envs_idx': continue
                    if reward[i] is None: raise ValueError("Reward saving error!")
                    self.info[key][i] = info[key][i]
        
        if active_env_indices:
            if isinstance(obs, dict)==False: 
                raise TypeError('Obs Type Error!')
            
            for key in obs.keys():
                if key == 'envs_idx':
                    continue

                active_obs = []
                for i in active_env_indices:
                    if obs[key][i] is None: 
                        raise ValueError("Obs Saving Error!")
                    active_obs.append(obs[key][i])
                
                if active_obs:
                    new_obs[key] = torch.stack(active_obs, dim=0)
        
        # print(f"Active envs: {active_env_indices}, new_obs keys: {list(new_obs.keys())}")
        return new_obs


    def _create_log_data(self):
        """Create logging data from collected episode information."""
        max_rewards = collections.defaultdict(list)
        log_data = dict()
    
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
                print(f"Warning: No rewards collected for env {i}")
            
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
    
            # Add video logs if available
            if should_upload_video and self.file_path is not None and i < len(self.file_path):
                video_path = self.file_path[i]
                if Path(video_path).exists(): 
                    try:
                        sim_video = wandb.Video(video_path)
                        log_data[prefix + f'sim_video_{seed}'] = sim_video
                        print(f"Added video log: {prefix}sim_video_{seed} from {video_path}")
                    except Exception as e:
                        print(f"Warning: Failed to create video log for {video_path}: {e}")
                else:
                    print(f"Warning: Video file not found: {video_path}")
    
        # Calculate mean scores
        for prefix, rewards in max_rewards.items():
            if len(rewards) > 0:
                mean_score = np.mean(rewards)
                log_data[prefix + 'mean_score'] = mean_score
                print(f"Mean score for {prefix}: {mean_score:.3f}")
            else:
                log_data[prefix + 'mean_score'] = 0.0
                print(f"Warning: No rewards for {prefix}")
    
        return log_data

if __name__ == "__main__":
        # use collect_data policy to check availability
        output_dir = ''
        Runner = PushTImageRunner(output_dir)
        policy = TestPolicy(Runner.n_action_steps)
        Runner.run(policy)






