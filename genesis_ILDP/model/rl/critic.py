import torch.nn as nn 
import torch

from genesis_ILDP.model.common.mlp import MLP, ResidualMLP
from genesis_ILDP.model.common.miniblock import MiniBlock
from genesis_ILDP.model.encoding.base_encoding import BaseEncoding
from genesis_ILDP.utils.module_utils import ModuleDict
from typing import List, Type, Optional, Dict, Union


ModuleType = Type[nn.Module]

class BaseCritic(nn.Module):
    '''
    BaseCritic: a MLP based simple base critic used for future wrapping with different kind out input types 
    '''
    def __init__(self, in_dim: int, 
                 arch: List[int], 
                 activation: Optional[ModuleType], 
                 normalisation: Optional[ModuleType]):

        super().__init__()
        dims = [in_dim] + arch
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += MiniBlock(in_dim, out_dim, norm_layer=normalisation, activation=activation)
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.model(state)
    
class CriticObs(BaseCritic):
    """State-only critic network.
    
    Obs: one type simple state inputs(usually feasible for training inputs from gym environments
      where the accepted inputs are only smaller than 60 arguments)
    """

    def __init__(
        self,
        critic_config: Dict[List|int],
    ):
        in_dim = critic_config['obs_dim']
        arch = critic_config['arch']
        activation = ModuleDict.get(critic_config['activation'].lower())
        normalisation = ModuleDict.get(critic_config['normalisation'].lower())

        super().__init__(in_dim=in_dim,
                         arch=arch,
                         activation=activation,
                         normalisation=normalisation)

    def forward(self, obs):
        return super().forward(state=obs)

class CriticImageObs(BaseCritic):
    """
    CriticImageObs: Critic network with obs input which include multiple types, not only state based variables, but also for imgs
    (which need additional configuration like whether to use Vit or simple cnn or so on)
    """

    def __init__(
        self,
        critic_config: Dict[int|list],
        img_encoding: BaseEncoding,
        state_encoding: BaseEncoding,
        residual_style = False,
    ):
        in_dim = len(img_encoding) + len(state_encoding)
        arch = critic_config['arch']
        activation = ModuleDict.get(critic_config['activation'].lower())
        normalisation = ModuleDict.get(critic_config['normalisation'].lower())

        super().__init__(in_dim=in_dim,
                         arch=arch,
                         activation=activation,
                         normalisation=normalisation)

        self.img_encoding = img_encoding
        self.state_encoding = state_encoding

    def forward(self, obs):
        img_vec = self.img_encoding(obs['imgs'])
        state_vec = self.state_encoding(obs['agent_pos'])
        feature_vec = torch.cat([img_vec, state_vec], dim=-1)

        return super().forward(state=feature_vec)
        
