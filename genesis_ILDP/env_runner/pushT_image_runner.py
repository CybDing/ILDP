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
                 n_train = 1, # Using train seed 
                 n_test = 2, # Using test seed
                 n_obs_steps = 8,
                 n_action_steps = 8,
                 max_steps=200,
                 render_size=128,
                 tqdm_interval_sec=1.0,
                 n_envs = None,
                 fps = 20,
                 crf = 22, # video quality
                 past_action=False,
                 seed_train = 0,
                 seed_test = 10000, # assert data collected 
                 # does not reach 10000 episodes
                 enable_render = True
                 ):
        super().__init__(output_dir)
        if n_envs is None:
            self.n_envs = n_train + n_test
        else: raise NotImplementedError('n_envs None Not implemented!')

        steps_per_render = 1 # double check!

        self.env = MultiStepWrapper(
                    PushTEnv(
                        render_size=render_size,
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
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        # self.device = gs.device
        self.max_steps = max_steps
        self.past_action = False
        self.seed_train = seed_train
        self.seed_test = seed_test
        self.file_path = list()
        self.base_generate_path = None
        self.enable_render = enable_render
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env_seeds = None
        self.info = None

        self._setup_envs()

        self.episode_obs = None
        self.episode_reward = [[] for _ in range(self.env.n_envs)]
        self.episode_info = dict()
        self.max_reward = [None for _ in range(self.env.n_envs)]

    def run(self, policy):
        
        self._prepare_env()
        obs = self.env.reset()
        past_action = None
        # policy.reset() # reset states for stateful policy
        done = False

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushTImageRunner",
                             leave=False, mininterval=self.tqdm_interval_sec)
        
        if self.enable_render: self.env.start_recording()
        while not done: # Run for one epoch

            ## TODO clear out "envs_idx" 
            obs_dict = dict(obs)

            # Not implemented yet! TODO Change past_action
            if self.past_action and (past_action is not None):
                obs_dict['past_action'] = past_action[
                                        :, -(self.n_obs_steps - 1):
                                        ].astype(np.float32)

            with torch.no_grad():
                action = policy.predict_action(obs_dict)

            # action = action_dict['action']

            obs, reward, done, info, env_status = self.env.step(action)
            print(f"Step: done={done}, env_status={env_status}, active_envs={len([i for i in range(len(env_status)) if env_status[i] == 2])}")
            # TODO add past action into process_info
            past_action = action
            
            obs = self._process_info(obs, reward, info, env_status)

            pbar.update(action.shape[1])
        pbar.close()

        if self.enable_render: 
            self.env.stop_recording(self.base_generate_path)
            self._process_generated_videos()
            self._cleanup_temp_files()

        self.seed_train = self.seed_train + self.n_train
        self.seed_test = self.seed_test + self.n_test

        self._create_log_data()

    def _setup_envs(self,):

        self.env.start(n_envs=self.n_envs, env_separate=True)
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
            self.episode_reward = [[] for _ in range(self.env.n_envs)]
            self.episode_info = dict()
            self.max_reward = [None for _ in range(self.env.n_envs)]


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


    def _create_log_data(self, ):
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(self.n_envs):
            seed = self.env_seeds[i]
            if i < self.n_train:
                prefix = 'train/'
            else: prefix = 'test/'
            max_reward = np.max(np.array(self.episode_reward[i]))
            
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = self.file_path
            if video_path is not None:
                sim_video = wandb.Video(video_path[i])
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

if __name__ == "__main__":
        # use collect_data policy to check availability
        output_dir = ''
        Runner = PushTImageRunner(output_dir)
        policy = TestPolicy(Runner.n_action_steps)
        Runner.run(policy)






