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
        """Reset all environments and buffers."""
        obs = super().reset()
        self._reset_buffers()
        
        # Process all environment observations in batch
        for env_idx in range(self.n_envs):
            env_obs = self._extract_env_data(obs, env_idx)
            self.obs[env_idx].append(env_obs)
        
        return self._get_obs(self.n_obs_steps)
    
    def _reset_buffers(self):
        """Reset all buffers efficiently."""
        for env_idx in range(self.n_envs):
            self.obs[env_idx].clear()
            self.current_step_rewards[env_idx].clear()
            self.info[env_idx].clear()
        
        # Reset counters in batch
        self.step_counts = [0] * self.n_envs
        self.active_envs = list(range(self.n_envs))
    
    def _extract_env_data(self, data, array_idx):
        """Extract data for specific environment index, with proper bounds checking."""
        if not isinstance(data, dict):
            return data
            
        extracted = {}
        for key, value in data.items():
            if isinstance(value, (np.ndarray, torch.Tensor, list)):
                try:
                    if array_idx >= len(value):
                        raise IndexError(f"Environment index {array_idx} out of bounds for key '{key}' with length {len(value)}")
                    extracted[key] = value[array_idx]
                except (IndexError, TypeError) as e:
                    raise ValueError(f"Failed to extract environment {array_idx} data for key '{key}': {e}")
            else:
                # Scalar values are shared across environments
                extracted[key] = value
        return extracted

    def step(self, action):
        """Execute multi-step actions efficiently."""
        
        # Validate action format once
        self._validate_action(action)
        
        # Clear rewards for all active environments in one operation
        for env_idx in self.active_envs:
            self.current_step_rewards[env_idx].clear()
        
        done_list = []
        
        # Pre-convert active_envs to tensor (avoid repeated conversion)
        active_envs_tensor = to_torch(np.array(self.active_envs))
        
        # Determine data types once
        reward_is_array = None
        done_is_array = None
        
        # Execute n_action_steps
        for step_idx in range(self.n_action_steps):
            current_action = action[:, step_idx, :]
            
            observation, reward, done, info = self.env.step(
                action=current_action, envs_idx=active_envs_tensor)
            
            # Determine array types on first iteration only
            if step_idx == 0:
                reward_is_array = isinstance(reward, (list, np.ndarray, torch.Tensor))
                done_is_array = isinstance(done, (list, np.ndarray, torch.Tensor))
            
            # Process all environments in batch
            for i, env_idx in enumerate(self.active_envs):
                # Extract data using unified function
                env_obs = self._extract_env_data(observation, i)
                env_reward = reward[i] if reward_is_array else reward
                env_done = done[i] if done_is_array else done
                env_info = self._extract_env_data(info, i)
                
                # Update environment state
                self.obs[env_idx].append(env_obs)
                self.current_step_rewards[env_idx].append(env_reward)
                self.info[env_idx] = env_info
                
                # Check termination conditions
                self.step_counts[env_idx] += 1
                if (self.max_episode_steps is not None and 
                    self.step_counts[env_idx] >= self.max_episode_steps):
                    env_done = True
                
                # Track done environments (avoid duplicates)
                if env_done and env_idx not in done_list:
                    done_list.append(env_idx)
        
        # Aggregate results efficiently
        observation = self._get_obs(self.n_obs_steps)
        reward = self._aggregate_current_rewards()
        info = self._aggregate_current_info()
        
        # Create env_status array efficiently
        env_status = self._create_env_status(done_list)
        
        # Remove terminated environments from active list
        self.active_envs = [idx for idx in self.active_envs if idx not in done_list]
        
        done = len(self.active_envs) == 0
        
        return observation, reward, done, info, env_status
    
    def _aggregate_current_rewards(self):
        """Aggregate rewards with proper error handling."""
        aggregated_rewards = []
        for env_idx in range(self.n_envs):
            env_rewards = self.current_step_rewards[env_idx]
            if env_rewards:
                try:
                    reward = aggregate(env_rewards, self.reward_agg_method)
                    aggregated_rewards.append(reward)
                except ValueError as e:
                    raise RuntimeError(f"Failed to aggregate rewards for environment {env_idx}: {e}")
            else:
                # Empty reward list indicates no steps were executed for this env
                # This should only happen for inactive environments
                if env_idx in self.active_envs:
                    raise RuntimeError(f"Active environment {env_idx} has no reward data")
                aggregated_rewards.append(0.0)
        
        return np.array(aggregated_rewards)
    
    def _validate_action(self, action):
        """Validate action format once."""
        if not isinstance(action, (np.ndarray, torch.Tensor)):
            raise ValueError("Action should be numpy array or torch tensor")
        if len(action.shape) != 3:
            raise ValueError(f"Action dim should be 3D: (n_active_envs, n_action_steps, action_dim)")
        if action.shape[0] != len(self.active_envs):
            raise ValueError(f'Action batch_size ({action.shape[0]}) != active_envs ({len(self.active_envs)})')
    
    def _create_env_status(self, done_list):
        """Create environment status array efficiently."""
        env_status = [0] * self.env.n_envs  # More efficient than list comprehension
        
        # Set active environments to status 2
        for idx in self.active_envs:
            env_status[idx] = 2
        
        # Set terminated environments to status 1
        for idx in done_list:
            env_status[idx] = 1
            
        return env_status
    
    def _aggregate_current_info(self):
        """Aggregate info from all environments efficiently."""
        if not any(self.info):  # Early exit if no info
            return {}
            
        # Collect all keys efficiently
        all_keys = set()
        for info_dict in self.info:
            all_keys.update(info_dict.keys())
        
        # Build aggregated info with list comprehension
        aggregated_info = {
            key: [self.info[env_idx].get(key, None) for env_idx in range(self.n_envs)]
            for key in all_keys
        }
        
        return aggregated_info

    def _get_obs(self, n_steps=1):
        """Get stacked observations efficiently."""
        if self.n_envs == 0 or not all(len(obs_deque) > 0 for obs_deque in self.obs):
            raise RuntimeError("No observation data available")
        
        # Get sample observation using direct deque access (no list conversion)
        sample_obs = self.obs[0][-1]
        
        if not isinstance(sample_obs, dict):
            raise TypeError("Observation Space should be dict type!")
        
        result = {}
        
        # Process each observation key
        for key in sample_obs.keys():
            # Pre-allocate list for better performance
            env_observations = []
            
            # Extract observations for all environments
            for env_idx in range(self.n_envs):
                # Direct deque access instead of list comprehension
                key_obs_history = [obs[key] for obs in self.obs[env_idx]]
                stacked_obs = stack_last_n_obs(key_obs_history, n_steps)
                env_observations.append(stacked_obs)
            
            # Stack observations (type determined once per key)
            if isinstance(env_observations[0], torch.Tensor):
                result[key] = torch.stack(env_observations, dim=0)
            else:
                result[key] = np.stack(env_observations, axis=0)
        
        return result

# Utility functions for space creation
def repeated_space(space, n):
    """Create repeated gym space for multi-step actions/observations."""
    if isinstance(space, spaces.Box):
        return spaces.Box(
            low=np.repeat(np.expand_dims(space.low, axis=0), n, axis=0),
            high=np.repeat(np.expand_dims(space.high, axis=0), n, axis=0),
            shape=(n,) + space.shape,
            dtype=space.dtype
        )
    elif isinstance(space, spaces.Dict):
        return spaces.Dict({
            key: repeated_space(value, n) 
            for key, value in space.items()
        })
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def aggregate(data, method='max'):
    """Aggregate reward data using specified method."""
    if not data:
        raise ValueError(f"Cannot aggregate empty data with method '{method}'")
        
    # Use dict for cleaner lookup
    aggregation_methods = {
        'max': np.max,
        'min': np.min, 
        'mean': np.mean,
        'sum': np.sum
    }
    
    if method not in aggregation_methods:
        raise ValueError(f"Unsupported aggregation method: {method}. Choose from {list(aggregation_methods.keys())}")
    
    try:
        return float(aggregation_methods[method](data))
    except Exception as e:
        raise ValueError(f"Failed to aggregate data with method '{method}': {e}")

def stack_last_n_obs(all_obs, n_steps):
    """Stack last n observations, duplicating if insufficient history."""
    assert len(all_obs) > 0, "No observations available for stacking"
    
    # Convert to list only if necessary
    if not isinstance(all_obs, list):
        all_obs = list(all_obs)
        
    sample_obs = all_obs[-1]
    n_available = len(all_obs)
    start_idx = -min(n_steps, n_available)
    
    if isinstance(sample_obs, torch.Tensor):
        result = torch.zeros((n_steps,) + sample_obs.shape, 
                           dtype=sample_obs.dtype, device=sample_obs.device)
        
        # Fill with available observations
        result[start_idx:] = torch.stack(all_obs[start_idx:], dim=0)
        
        # Duplicate first available observation to fill missing history
        if n_steps > n_available:
            result[:start_idx] = result[start_idx]
    else:
        result = np.zeros((n_steps,) + sample_obs.shape, dtype=sample_obs.dtype)
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > n_available:
            result[:start_idx] = result[start_idx]
            
    return result