import torch
import torch.nn as nn

from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.model.rl.critic import BaseCritic
from genesis_ILDP.agent.rl_agent.base_agent import BaseAgent

class PPODiffusionAgent(BaseAgent): 
    """
    PPODiffusionAgent: an PPO based RL agent for training the actor and critic for certain tasks
    
    This agent contains a actor network and a critic network which is able to be visited from trainer.py via self.agent.actor/critic,
    this is really important when I have to assign the trajcollector.py's collector actor policy with a correct model instance which
    is able to collect trajs inside the pusht_env.py and then calculating the logprob in the middleplace.
    """
    def __init__(self, max_FTdiff_idx: float, 
                       clip_base_ploss: float, 
                       clip_rate_ploss: float, 
                       clip_vloss_value: float, 
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
        self.clip_vloss_value = clip_vloss_value
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.advantage_decay_coeff = advantage_decay_coeff
        
        self.weight_advantage_ploss = None 

    def _clip_sigratio(self, raw_ratio, denoising_idx) -> None: 
        if self.clip_base_ploss is None or self.clip_rate_ploss is None: 
            print("[PPODiffusion] clip_base_ploss or clip_rate_ploss is not set, used default clamping method for all idx")
            return torch.clamp(raw_ratio, min=0.9, max=1.1)

        # the decaying weight from diff_step idx 0 - 1 (diff_steps + 1 points)
        if self.weight_advantage_ploss is None: 
            idx_normed = torch.arange(self.max_FTdiff_idx + 1, device=raw_ratio.device) / self.max_FTdiff_idx 
            self.clip_rate_ploss = self.clip_rate_ploss if self.clip_rate_ploss > 0 else 0
            self.weight_advantage_ploss = self.clip_base_ploss * torch.exp(-self.clip_rate_ploss * idx_normed)
        
        weights = self.weight_advantage_ploss.to(raw_ratio.device)
        return torch.clamp(raw_ratio, min=1-weights[denoising_idx], 
                           max=1+weights[denoising_idx])


    def train_step(self, chain_before, chain_next, advantage, obs, denoising_idx, old_value, old_logprob):
        ploss, vloss = self._loss(chain_before, chain_next)
        self.p_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

        ploss.backward()
        vloss.backward()

        self.p_optimizer.step()
        self.v_optimizer.step()


    def _loss(self, 
             chain_before,
             chain_next, 
             advantage, # the GAE calculated per env step
             obs, 
             denoising_idx, 
             old_value, # calculated once for the final action really being executed 
             # and the advantage is decayed because of the GAE inside the diffusion cycle
             
             # new_value, # the output of current critic networks respect to current observations
             old_logprob, # calculated in the outer loop in the final workspace(only needs to be calculated once)
             
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
            - new value (B, )
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

        new_logprob = self.actor.get_logprob(obs_dict = obs, 
                                              chain_before = chain_before, 
                                              chain_after = chain_next, 
                                              denoising_idx = denoising_idx, 
                                              isNorm_action = True,# normalized action sequence are collected inside the 'action_diffusion_image_policy'
                                              isNorm_obs = False, # unnormalized obs collected 
                                              )
        # --- Policy Gradient loss ---
        clipped_ratio = self._clip_sigratio(raw_ratio=new_logprob/old_logprob, denoising_idx=denoising_idx)
        advantage_clipped = torch.clamp(advantage, min=torch.quantile(advantage, self.clip_advantage_lower_quantile, dim=0),
                                        max=torch.quantile(advantage, self.clip_advantage_upper_quantile, dim=0))
        advantage_decayed = advantage_clipped * torch.Tensor([self.advantage_decay_coeff ** idx for idx in denoising_idx])

        
        ploss = -advantage_decayed * clipped_ratio # negative of the total sampled advantage
        # --- Critic Qnet loss ---
        new_value = self.critic(obs) # TODO make sure the obs input match the critic style
        vloss_unclipped = (new_value - old_value) ** 2
        value_clipped = old_value + torch.clamp(
            new_value - old_value, 
            min=-self.clip_vloss_value,
            max= self.clip_vloss_value
        )
        vloss_clipped = (value_clipped - old_value) ** 2
        vloss = torch.max(vloss_unclipped, vloss_clipped)
        vloss = 0.5 * vloss.mean()

        

        return ploss, vloss