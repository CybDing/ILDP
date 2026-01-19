### Modified from diffusion policy repository 

from typing import Dict
import torch
import torch.nn as nn
# from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseImagePolicy:
    """
    Base imagePolicy for robot to manipulate using imgs as input as well as other perceivable states
    and useful information to predict future actions. 
    """

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
