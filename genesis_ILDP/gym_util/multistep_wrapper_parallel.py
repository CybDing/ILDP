import gym
from gym import spaces
import numpy as np
from collections import deque
import torch
from genesis_ILDP.utils.cuda import * 

class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            n_envs,
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        
        # 获取环境数量
        self.n_envs = n_envs
        if n_envs is None:
            raise ValueError("Environment must have n_envs attribute")
        
        self._initialize_buffers(self.n_envs)
    
    def _initialize_buffers(self, n_envs):
        """obs, current rewards, latest info, step counts, active_envs"""
        self.obs = [deque(maxlen=self.n_obs_steps+1) for _ in range(n_envs)]
        
        self.current_step_rewards = [[] for _ in range(n_envs)]

        self.info = [{} for _ in range(n_envs)]
        
        self.step_counts = [0] * n_envs
        self.active_envs = list(range(n_envs))  
    
    def reset(self):
        
        obs = super().reset()
        self._reset_buffers()
        
        # 为每个环境添加初始观察
        for env_idx in range(self.n_envs):
            env_obs = self._extract_env_obs(obs, env_idx)
            self.obs[env_idx].append(env_obs)
        
        return self._get_obs(self.n_obs_steps)
    
    def _reset_buffers(self):
        for env_idx in range(self.n_envs):
            self.obs[env_idx].clear()
            self.current_step_rewards[env_idx].clear()
            self.info[env_idx].clear()  # 简单清空字典
            self.step_counts[env_idx] = 0
        
        self.active_envs = list(range(self.n_envs))
    
    def _extract_env_obs(self, obs, array_idx):
        """确保提取的观察保持原始tensor格式"""
        if isinstance(obs, dict):
            env_obs = {}
            for key, value in obs.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    extracted = value[array_idx]
                    # 🔑 保持torch tensor格式，不转换
                    env_obs[key] = extracted
                else:
                    env_obs[key] = value
            return env_obs
        else:
            raise ValueError("Obs should be dict type!")
        
    def _extract_env_info(self, info, array_idx):
        """从返回的info数组中提取指定索引的info"""
        if isinstance(info, dict):
            env_info = {}
            for key, value in info.items():
                if isinstance(value, (np.ndarray, torch.Tensor, list)):
                    try:
                        env_info[key] = value[array_idx]
                    except (IndexError, TypeError):
                        env_info[key] = value
                else:
                    env_info[key] = value
            return env_info
        else:
            return info

    def step(self, action):
        """执行多步动作"""
        
        # Action 检查
        if isinstance(action, (np.ndarray, torch.Tensor)):
            if len(action.shape) != 3:
                raise ValueError(f"Action dim should be 3D: (n_active_envs, n_action_steps, action_dim)")
        else:
            raise ValueError("Action should be numpy array or torch tensor")
        
        # 检查 action batch size
        if action.shape[0] != len(self.active_envs): 
            raise ValueError(f'Action batch_size ({action.shape[0]}) != active_envs ({len(self.active_envs)})')
        
        for env_idx in self.active_envs:
            self.current_step_rewards[env_idx].clear()
        
        done_list = []
        
        # 执行 n_action_steps 步
        for step_idx in range(self.n_action_steps):
            current_action = action[:, step_idx, :]  # (n_active_envs, action_dim)
            
            # 传递活跃环境的 tensor
            active_envs_tensor = to_torch(np.array(self.active_envs))
            observation, reward, done, info = self.env.step(action=current_action, 
                                                           envs_idx=active_envs_tensor)
            
            for i, env_idx in enumerate(self.active_envs):
                
                env_obs = self._extract_env_obs(observation, i)
                env_reward = reward[i] if isinstance(reward, (list, np.ndarray, torch.Tensor)) else reward
                env_done = done[i] if isinstance(done, (list, np.ndarray, torch.Tensor)) else done
                env_info = self._extract_env_info(info, i)
                
                self.obs[env_idx].append(env_obs)
                self.current_step_rewards[env_idx].append(env_reward)
                
                self.info[env_idx] = env_info  # 🔑 简化：直接覆盖最新的 info
                
                self.step_counts[env_idx] += 1
                if (self.max_episode_steps is not None) and \
                   (self.step_counts[env_idx] >= self.max_episode_steps):
                    env_done = True
                
                if env_done and env_idx not in done_list:
                    done_list.append(env_idx)
        
        
        observation = self._get_obs(self.n_obs_steps)
        reward = self._aggregate_current_rewards()
        info = self._aggregate_current_info()

        env_status = [0 for _ in range(self.env.n_envs)]

        # CHECK
        for idx in self.active_envs:
            if idx in done_list: 
                env_status[idx] = 1
            else:
                env_status[idx] = 2
        
        for idx in done_list:
            if idx in self.active_envs:
                self.active_envs.remove(idx)

        done = self._aggregate_done()

        return observation, reward, done, info, env_status
    
    def _aggregate_current_rewards(self):
        """聚合当前步骤的奖励"""
        aggregated_rewards = []
        for env_idx in range(self.n_envs):
            if len(self.current_step_rewards[env_idx]) > 0:
                env_reward = aggregate(self.current_step_rewards[env_idx], self.reward_agg_method)
            else:
                env_reward = 0.0
            aggregated_rewards.append(env_reward)
        return np.array(aggregated_rewards)
    
    def _aggregate_done(self):
        """检查是否所有环境都完成"""
        return len(self.active_envs) == 0
    
    def _aggregate_current_info(self):
        """简化的 info 聚合：直接返回最新的 info"""
        aggregated_info = {}
        
        all_keys = set()
        for env_idx in range(self.n_envs):
            all_keys.update(self.info[env_idx].keys())
        
        for key in all_keys:
            values = []
            for env_idx in range(self.n_envs):
                if key in self.info[env_idx]:
                    values.append(self.info[env_idx][key])
                else:
                    values.append(None)
            aggregated_info[key] = values
        
        return aggregated_info

    def _get_obs(self, n_steps=1):
        if self.n_envs == 0 or not all(len(obs_deque) > 0 for obs_deque in self.obs):
            raise RuntimeError("没有可用的观察数据")
        
        sample_obs = list(self.obs[0])[-1]
        
        if isinstance(sample_obs, dict):
            result = {}
            for key in sample_obs.keys():
                env_observations = []
                for env_idx in range(self.n_envs):
                    key_obs_history = [obs[key] for obs in self.obs[env_idx]]
                    stacked_obs = stack_last_n_obs(key_obs_history, n_steps)
                    env_observations.append(stacked_obs)
                
                if isinstance(env_observations[0], torch.Tensor):
                    result[key] = torch.stack(env_observations, dim=0)
                else:
                    result[key] = np.stack(env_observations, axis=0)
            
            return result
        else:
            raise TypeError("Observation Space should be dict type!")

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def aggregate(data, method='max'):
    if method == 'max':
        return np.max(data)
    elif method == 'min':
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):

    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    sample_obs = all_obs[-1]
    
    if isinstance(sample_obs, torch.Tensor):

        result = torch.zeros((n_steps,) + sample_obs.shape, 
                           dtype=sample_obs.dtype, 
                           device=sample_obs.device)
        
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:], dim=0)
        
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
        
        return result
    else:
        # numpy array处理
        result = np.zeros((n_steps,) + sample_obs.shape, 
                         dtype=sample_obs.dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
        return result