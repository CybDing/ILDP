import torch 
import torch.nn as nn
import torch.optim as optim
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.model.rl.critic import CriticImageObs
from genesis_ILDP.model.rl.critic import BaseCritic
from genesis_ILDP.utils.module_utils import get_optimizer
from typing import Dict


class BaseAgent:
    def __init__(self, 
                 critic: BaseCritic, 
                 actor: BaseImagePolicy, 
                 optimizer_config: Dict[float|str]):
        
        self.critic = critic
        self.actor = actor
        self.optimizer_config = optimizer_config
        self.p_optimizer = None # actor optimizer used for p_loss 
        self.v_optimizer = None # critic optimizer used for v_loss
        
        self._build_optimizer()

    def _build_optimizer(self, ):
        critic_config = self.optimizer_config['critic']
        actor_config = self.optimizer_config['actor']

        self.p_optimizer = get_optimizer(type=actor_config['type'], 
                                         params=self.actor.parameters(), 
                                         lr=self.actor_config['lr'], 
                                         weight_decay=self.actor_config['weight_decay']
                                         )
        
        self.v_optimizer = get_optimizer(type=critic_config['type'], 
                                         params=self.critic.parameters(), 
                                         lr=self.critic_config['lr'], 
                                         weight_decay=self.critic_config['weight_decay']
                                         )
    def train_step(self, *args):
        raise NotImplementedError

    def _loss(self, *args, ): 
        raise NotImplementedError

    def _ploss(self, *args, ):
        raise NotImplementedError


    def _vloss(self, *args, ):
        raise NotImplementedError