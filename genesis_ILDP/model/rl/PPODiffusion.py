import torch
import torch.nn as nn

# The ppo for reinforcement learning for diffusion model is not implemented yet!
# this is the class where we will calculate the ppo loss (policy gradient)
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
"""
Intermediate Param: 
    new_logprob (B, Ta) calculated using the same obs, chain_before and chain next
                        with the new policy
    
    loss for the ppo update, and the value function net loss
"""

class PPODiffusion: 
    def __init__(self, max_FTdiff_idx: float, 
                       clip_base_ploss: float, 
                       clip_rate_ploss: float, 
                       clip_vloss: float, 
                       clip_advantage_upper_quantile, 
                       clip_advantage_lower_quantile, 
                       advantage_decay_coeff):
        """
        ## Args:
          -  max_FTdiff_idx: the largest idx the fine-tuning will sample for training
          The valid range of diffusion idx consider this will be from (max_FTdiff_idx, 0)
          - clip_base_ploss
          - clip_rate_ploss 
          - clip_vloss
        """
        self.max_FTdiff_idx = max_FTdiff_idx
        self.clip_base_ploss = clip_base_ploss
        self.clip_rate_ploss = clip_rate_ploss
        self.clip_vloss = clip_vloss
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.advantage_decay_coeff = advantage_decay_coeff
        
        self.weight_advantage_ploss = None 
    
    def load_policy(self, policy:BaseImagePolicy): 
        # use the instantiated policy(should be FT style) for calculating the logprob as well as updating it through
        # optimizer.step() method 
        self.policy = policy 

    def _clip_sigratio(self, raw_ratio, denoising_idx) -> None: 
        if self.clip_base_ploss is None or self.clip_rate_ploss is None: 
            print("[PPODiffusion] clip_base_ploss or clip_rate_ploss is not set, used default clamping method for all idx")
            return torch.clamp(raw_ratio, min=0.9, max=1.1)

        if self.weight_advantage_ploss is None: 
            idx_normed = torch.arange(self.max_FTdiff_idx + 1) / self.max_FTdiff_idx 
            assert self.clip_rate_ploss > 0 # the rate should be positive
            self.weight_advantage_ploss = self.clip_base_ploss * torch.exp(-self.clip_rate_ploss * idx_normed)
        
        return torch.clamp(raw_ratio, min=1-self.weight_advantage_ploss[denoising_idx], 
                           max=1+self.weight_advantage_ploss[denoising_idx])

    def loss(self, chain_before,
             chain_next, 
             advantage, # the GAE calculated per env step
             obs, 
             denoising_idx, 
             old_value, # calculated once for the final action really being executed 
             # and the advantage is decayed according to this baseline from the GAE inside the diffusion cycle
             old_logprob, # calculated in the outer loop in the final workspace
              # which only needed to calculate once for efficiency 
            ):
        """
        ## Args: 
            - chain_before (B, Ta, Da)
            - chain_next (*chain_before.shape) *Processed through the env done for sampling using the 
            - advantage (B, )
            - obs  (B, To, Do)
                - img (B, To, c, w, h)
                - agent_pos (B, To, Dp)
            - denoising_idx (B, ) the exact transition process used for ppo update
            - old_value (B, )
            - old_logprob (B, k+1, Ta)
            - real_action_horizon (action_start_idx, action_end_idx)
        
        ## Return: 
            ploss, vloss
        """
      
        # clamping the denoising idx to not exceed the maximum set for fine-tuning 
        denoising_idx = torch.clamp(denoising_idx, min=0, max=self.max_FTdiff_idx)

        batch_size = denoising_idx.shape[0]  
        assert advantage.shape[0] == batch_size and obs['image'].shape[0] == batch_size and \
        old_value.shape[0] == batch_size and old_logprob.shape[0] == batch_size
        assert chain_before.shape[0] == chain_next.shape[0]

        new_logprob = self.policy.get_logprob(obs_dict = obs, 
                                              chain_before = chain_before, 
                                              chain_after = chain_next, 
                                              denoising_idx = denoising_idx, 
                                              isNorm_action = True, # assert the collector collecting the 
                                              # normalized action sequence from the pusht_image_runner.py
                                              isNorm_obs = False, # the obs collected is the raw obs from the pusht_env
                                              )
        # get clipped siginficant ratio
        clipped_ratio = self._clip_sigratio(raw_ratio=new_logprob/old_logprob)
        advantage_clipped = torch.clamp(advantage, min=torch.quantile(advantage, self.clip_advantage_lower_quantile, dim=0), 
                                        max=torch.quantile(advantage, self.clip_advantage_upper_quantile))
        advantage_decayed = advantage_clipped * torch.Tensor([self.advantage_decay_coeff ** idx for idx in denoising_idx])

        # Policy Gradient loss
        ploss = -advantage_decayed * clipped_ratio # negative of the total sampled advantage

        if self.clip_vloss is not None: 
            advantage_clipped_v = torch.clamp(advantage, min=1-self.clip_vloss, max=1+self.clip_vloss)
        else: advantage_clipped_v = advantage
        # Critic Qnet loss
        vloss = torch.nn.functional.mse_loss(advantage_clipped_v, old_value)

        return ploss, vloss