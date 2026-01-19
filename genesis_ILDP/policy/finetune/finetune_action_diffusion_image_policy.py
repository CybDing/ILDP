from genesis_ILDP.policy.action_diffusion_image_policy import ActionDiffusionImagePolicy
import torch
import numpy as np
from genesis_ILDP.utils.cuda import *

class FTActionDiffusionImagePolicy(ActionDiffusionImagePolicy):
    """

    FTActionDiffusionImagePolicy serves additional functionalities 
    which is vital for fine-tuning using the ppo and other reinforcement learning
    algorithms such as: 
    1. the calculation of the logprob given the input action and denoised action with the diffusion step indexes
    2. ... 

    """
    def __init__(self, shape_meta, normalizer = None,
                vision_backbone = 'custom',
                diff_steps = 100, 
                scheduler_mode = 'Linear', 
                obs_steps = 2,
                horizon = 16,
                n_action_steps = 8,
                n_obs_steps = 2,
                encode_agent_pos = False,
                ddim_steps = None,
                noise_intensity = 0, 
                crop_shape=..., 
                enable_crop=True,
                img_encoding_dim=1024,
                time_encoding_dim=128,
                pos_encoding_dim=None, 
                **kwargs):
        
        super().__init__(shape_meta, normalizer, vision_backbone,
                          diff_steps, scheduler_mode, obs_steps, horizon, 
                          n_action_steps, n_obs_steps,encode_agent_pos, ddim_steps,
                          noise_intensity, crop_shape, enable_crop, img_encoding_dim,
                          time_encoding_dim, pos_encoding_dim, **kwargs)
    
    def _process_obs(self, obs_dict, should_norm = True):
        assert hasattr(FTActionDiffusionImagePolicy, 'normalizer') # assert the normalizer exist
        if isinstance(obs_dict, dict): 
            try: 
                image = obs_dict['image']
                agent_pos = obs_dict['agent_pos']
            except:
                print("[FT_action_diffusion_image_policy] Parsing obs_dict error")
        
        if isinstance(image, np.numpy):
            image = to_torch(image)
        if isinstance(agent_pos, np.numpy):
            agent_pos = to_torch(agent_pos)
        
        if image.shape[-1] == 3 or image.shape[-1] == 1: # if image channels are not swapper before the width and height 
            image = image.transpose(dim0=1, dim1=-1)
        
        if should_norm: 
            image = self.normalizer['img'].normalize(image)
            agent_pos = self.normalizer['agent_pos'].normalize(agent_pos)

        return image, agent_pos

    def _process_action(self, action, should_norm = False): 
        if should_norm:
            action = self.normalizer['action'].normalize(action)
        
        return action    

    # def _encoding_obs(self, img, agent_pos, training = True) -> tuple:
    #     batch_size = img.shape[0]
    #     # 1. concatenate the img and agent_pos over the first dim(batch dim)
    #     img = img.reshape(-1, *img.shape[-3:])
    #     agent_pos = agent_pos.reshape(-1, *agent_pos.shape[-1]) # fixed dim should be the last dim which is Dpos
    #     # 2. apply the img cropping 
    #     img = self._apply_crop(img, training=training) # training visual encoder use random crop 
    #     # 3. apply the encoding network to the obs
    #     img_encoded = self.imgs_encoding_net(img)
    #     if self.encode_agent_pos: 
    #         agent_pos_encoded = self.agent_pos_encoding(agent_pos)
    #     else: agent_pos_encoded = agent_pos
    #     # 4. stack n_obs_steps encodings together(over the first dim to form feature vectors)
    #     img_feature = img_encoded.reshape(batch_size, -1)
    #     agent_pos_feature = agent_pos_encoded.reshape(batch_size, -1)

    #     return img_feature, agent_pos_feature

    def get_logprob(self, obs_dict, chain_before:torch.Tensor, chain_next:torch.Tensor, 
                    denoising_idx: torch.Tensor, isNorm_action = True, isNorm_obs = False # default action are normalized in replay buffer, while the obs is not 
                    ):
        # chain_before/next has already being extracted with the denoising idx
        image, agent_pos = self._process_obs(obs_dict=obs_dict, should_norm=not isNorm_obs)
        chain_before = self._process_action(chain_before, should_norm=not isNorm_action)
        chain_after = self._process_action(chain_next, should_norm=not isNorm_action)

        batch_size = chain_before.shape[0]
        assert batch_size == chain_next.shape[0] and batch_size == image.shape[0] \
               and batch_size == agent_pos.shape[0] # assert batch_size correct 

        assert denoising_idx.shape[0] == batch_size # check denoising index size correct

        time_feature = self.time_encoding(denoising_idx) # use the default time encoding dim 128
        img_feature, agent_pos_feature = self._encoding_obs(image, agent_pos)
        # merge into flat tensors for conditioning the unet
        global_cond = self.merge_multimodal_encoding(img_feature, agent_pos_feature, time_feature)

        noise_pred = self.conditioned_unet(chain_before, global_cond)
        action_pred_mean = chain_before - noise_pred 

        # the denoising index start from 0 to self.diff_steps - 1 (bit different from the original setting inside the 
        # ddpm prediction) for action sequence indexed from 1 to self.diff_steps 
        variance = self.betas[denoising_idx] * (1 - self.cum_alphas[denoising_idx-1]) / (1 - self.cum_alphas[denoising_idx])

        # clip the variance to larger than the preset threshold
        # TODO check and implement inside the parent class 

        variance = torch.clamp(variance, min=self.variance_threshold, max=None) 
        
        # return the log prob: same shape(B, Ta, Da)
        # chain after is sampled from the normal distribution with mean equals 0, 
        # and variance equals 1; so for the action_pred_mean closer to the chain after
        # the log prob tends to become larger 
        log_prob = - ((chain_after - action_pred_mean) / variance)

        return log_prob