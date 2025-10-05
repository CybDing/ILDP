import torch
import torch.nn as nn

# The ppo for reinforcement learning for diffusion model is not implemented yet!
# this is the class where we will calculate the ppo loss (policy gradient)
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
"""

Param: 
    clip_base_ploss
    clip_rate_ploss 
    clip_vloss

    chain_before (B, Ta, Da)
    chain_next (*chain_before.shape) *Processed through the env done for sampling using the 
    GAEreturn (B, )
    obs  (B, To, Do)
        img (B, To, c, w, h)
        agent_pos (B, To, Dp)
    denoising_idx (B, ) the exact transition process used for ppo update
    old_values (B, )
    old_logprob (B, k+1, Ta)
    real_action_horizon (action_start_idx, action_end_idx)

    new_logprob (B, Ta) calculated using the same obs, chain_before and chain next
                        with the new policy
    
    advantage: the average advantage calcuated via all batches and weighted by the 
                significant ratio and the decay factor(has included in GAE) 
    loss for the ppo update, and the value function update
"""

class PPODiffusion: 
    def __init__(self, clip_base_ploss: float, 
                       clip_rate_ploss: float, 
                       clip_vloss: float, 
                       clip_advantage_upper_quantile, 
                       clip_advantage_lower_quantile):
        
        pass
    
    def load_policy(self, policy:BaseImagePolicy): 
        # use the instantiated policy for calculating the logprob as well as updating it through
        # optimizer.step() method 
        self.policy = policy

    def loss(chain_before,
             chain_after, 
             GAE, 
             obs, 
             denoising_idx, 
             old_values, # calculated once for the final action really being executed 
             # and the advantage is decayed according to this baseline from the GAE inside the diffusion cycle
             old_logprob, # calculated in the outer loop in the final workspace
              # which only needed to calculate once for efficiency 
              ):
        pass 
