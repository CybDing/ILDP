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
            reward_agg_method='max',
            debug = False
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.debug = debug
        
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
        
        for env_idx in range(self.n_envs):
            env_obs = self._extract_env_data(obs, env_idx)
            self.obs[env_idx].append(env_obs)
        
        return self._get_obs(self.n_obs_steps)
    
    def _reset_buffers(self):
        """Reset all buffers."""
        for env_idx in range(self.n_envs):
            self.obs[env_idx].clear()
            self.current_step_rewards[env_idx].clear()
            self.info[env_idx].clear()
        
        self.step_counts = [0] * self.n_envs
        self.active_envs = list(range(self.n_envs))
    
    def _extract_env_data(self, data, array_idx):
        """Extract data for specific environment index, with proper shape checking."""
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
                print('<INFO> [multistep_wrapper_parallel.py] Fall back to list instead of env dict!')
                extracted[key] = value
        return extracted
    
    def step(self, action):
        """Execute multi-step actions"""

        # Handle action format from runner (can be dict or tensor)
        if isinstance(action, dict):
            if 'action' in action:
                action = action['action']  
                assert isinstance(action, torch.Tensor)
                assert action.shape[0] == len(self.active_envs) # do not check for the idx if the len is correct
                assert len(action.shape) == 3
            else:
                raise ValueError("Action dict must contain 'action' key")

        for env_idx in self.active_envs:
            self.current_step_rewards[env_idx].clear()

        done_list = []
        natural_done_list = []  # Track which envs reached natural done (not truncated)

        active_envs_tensor = to_torch(np.array(self.active_envs))

        # print("[multistep_wrapper_parallel.py]: action shape", action.shape)
        if self.debug:
            print("<Debug> [multistep_wrapper_parallel.py]: action in env_idx = 0", action[0, :])

        for step_idx in range(self.n_action_steps):
            current_action = action[:, step_idx, :]

            observation, reward, done, info = self.env.step(
                action=current_action, envs_idx=active_envs_tensor)
            # print("[env_observation]", observation)
            # print("[env_info]", info)

            # if step_idx == 0:
            #     reward_is_array = isinstance(reward, (list, np.ndarray, torch.Tensor))
            #     done_is_array = isinstance(done, (list, np.ndarray, torch.Tensor))

            # Process all environments in batch
            for env_idx in self.active_envs:
                # Extract data using unified function - use env_idx instead of i for correct indexing
                env_obs = self._extract_env_data(observation, env_idx)
                env_reward = reward[env_idx]
                env_done = done[env_idx]
                env_info = self._extract_env_data(info, env_idx)

                # Update environment state
                self.obs[env_idx].append(env_obs)
                self.current_step_rewards[env_idx].append(env_reward)
                self.info[env_idx] = env_info

                # Check termination conditions
                self.step_counts[env_idx] += 1
                natural_done = env_done  # Store natural done before overwriting

                if (self.max_episode_steps is not None and
                    self.step_counts[env_idx] >= self.max_episode_steps):
                    env_done = True

                if env_done and env_idx not in done_list:
                    done_list.append(env_idx)
                    if natural_done:  # Only add to natural_done if env reached done naturally
                        natural_done_list.append(env_idx)
    
        observation = self._get_obs(self.n_obs_steps)
        reward = self._aggregate_current_rewards()
        info = self._aggregate_current_info()

        done = self._create_env_status(done_list, natural_done_list)

        # Remove terminated environments from active list
        self.active_envs = [idx for idx in self.active_envs if idx not in done_list]

        return observation, reward, done, info
    
    def _aggregate_current_rewards(self):
        """Aggregate rewards. Stack the max reward thoughout the whole list with rewards, the inactive envs will be replaced by 0
            
            self.current_step_rewards: list including n_envs sublist for appending the current action step reward 
            
            output:
                torch.Tensor: shape(self.n_envs, )                
        """
        aggregated_rewards = []
        for env_idx in range(self.n_envs):
            env_rewards = self.current_step_rewards[env_idx]
            if env_rewards:
                try:
                    reward = aggregate(torch.stack(env_rewards), self.reward_agg_method)
                    aggregated_rewards.append(reward)
                except ValueError as e:
                    raise RuntimeError(f"Failed to aggregate rewards for environment {env_idx}: {e}")
            else:
                # Empty reward list indicates no steps were executed for this env
                # This should only happen for inactive environments
                if env_idx in self.active_envs:
                    raise RuntimeError(f"Active environment {env_idx} has no reward data")
                aggregated_rewards.append(0.0)
        
        return torch.stack(aggregated_rewards, dim=0)
    
    
    def _create_env_status(self, done_list, natural_done_list):
        """Create environment status list.
        Status codes:
        0: Already finished in previous iterations
        1: Just finished (truncated by max_steps)
        2: Still active
        3: Just finished (natural success - reached done condition)
        """
        env_status = [0] * self.env.n_envs  # Already finished env in previous round

        # Set active environments to status 2
        for idx in self.active_envs:
            env_status[idx] = 2

        # Set terminated environments: 3 for natural success, 1 for truncation
        for idx in done_list:
            if idx in natural_done_list:
                env_status[idx] = 3  # Natural success
            else:
                env_status[idx] = 1  # Truncated

        return env_status
    
    def _aggregate_current_info(self):
        """Aggregate info from all environments.
        The input of the aggregate_current_info from the outer self.info[idx] shoule be 
        simple dict without any subdict inside
        """
        if not any(self.info): 
            return {}
            
        all_keys = self.info[0].keys()
        
        aggregated_info = {
            key: [self.info[env_idx].get(key, None) for env_idx in range(self.n_envs)]
            for key in all_keys
        }
        
        return aggregated_info

    def _get_obs(self, n_steps=1):
        """Get stacked observations for all envs from the deque saving recent observations
            meta_data of observation got from pushT_env.py:
            {
                'envs_idx': torch.Tensor, 
                'image': torch.Tensor(envs_idx * 3 * (*render_size)), 
                'agent_pos': torch.Tensor(envs_idx * 2)
            }
            Former meta shape is one element inside the self.obs[envs_idx][idx] 
            which represent the observation for <envs_idx> at the <idx> action time step 

            output:{
                'envs_idx': torch.Tensor, (envs_idx * n_steps)
                'image': torch.Tensor(envs_idx * n_steps * 3 * (*render_size)), 
                'agent_pos': torch.Tensor(envs_idx * n_steps * 2)
            } 
        """

        if self.n_envs == 0 or not all(len(obs_deque) > 0 for obs_deque in self.obs):
            raise RuntimeError("No observation data available")
                
        # Get an sample observation from the deque for determining the key to iterate
        sample_obs = self.obs[0][0]
        
        if not isinstance(sample_obs, dict):
            raise TypeError("Observation Space should be dict type!")
        
        result = {}
        
        # Process each observation key
        for key in sample_obs.keys():

            env_observations = []
            
            # Extract observations for all environments
            for env_idx in range(self.n_envs):
                # Direct deque access instead of list comprehension
                key_obs_history = [obs[key] for obs in self.obs[env_idx]]
                stacked_obs = stack_last_n_obs(key_obs_history, n_steps)
                # ensure the observations stacked for different env is separated
                # inside different list could be stacked to turn into a batch
                env_observations.append(stacked_obs) 
            
            if isinstance(env_observations[0], torch.Tensor):
                result[key] = torch.stack(env_observations, dim=0)
            else:
                result[key] = np.stack(env_observations, axis=0)
        
        return result

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

    # Use dict for cleaner lookup - aggregate across dimension 0 (steps)
    aggregation_methods = {
        'max': lambda x: torch.max(x, dim=0)[0], # use [0] to extract the value not the index
        'min': lambda x: torch.min(x, dim=0)[0],
        'mean': lambda x: torch.mean(x, dim=0),
        'sum': lambda x: torch.sum(x, dim=0)
    }

    if method not in aggregation_methods:
        raise ValueError(f"Unsupported aggregation method: {method}. Choose from {list(aggregation_methods.keys())}")

    try:
        return aggregation_methods[method](data)
    except Exception as e:
        raise ValueError(f"Failed to aggregate data with method '{method}': {e}")

def stack_last_n_obs(all_obs, n_steps):
    """Stack last n observations, duplicating if insufficient history.
        args: 
            all_obs: list - including all the observation data in one type extracted from self.obs deque
    """

    assert len(all_obs) > 0, "No observations available for stacking"
        
    sample_obs = all_obs[-1]
    n_available = len(all_obs)
    start_idx = -min(n_steps, n_available)
    
    if isinstance(sample_obs, torch.Tensor):
        result = torch.zeros((n_steps,) + sample_obs.shape, 
                           dtype=sample_obs.dtype, device=sample_obs.device)
        
        result[start_idx:] = torch.stack(all_obs[start_idx:], dim=0)
        
        # fill the observation using the last observation available now
        if n_steps > n_available:
            last_obs = result[start_idx]
            for i in range(0, n_steps + start_idx):  # start_idx is negative
                result[i] = last_obs
    else:
        raise TypeError("<Error!> [multistep_wrapper_parallel.py] Check env observation type inside each key is torch.Tensor " \
        "for simpler transformation.")
            
    return result