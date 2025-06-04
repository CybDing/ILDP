from typing import Dict
import torch
import torch.nn as nn
# from genesis_ILDP.policy.base_image_policy import BaseImagePolicy


class TestPolicy():
    def __init__(self, n_action_steps):
        super().__init__()
        self.n_action_steps = n_action_steps

    def predict_action(self, obs_dict):
        for key in obs_dict.keys():
            
            n_envs = obs_dict[key].shape[0]
            break
        
        return torch.ones(size=(n_envs, self.n_action_steps, 2)) * 0.6
    
