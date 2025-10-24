# this file is not created fully and we use the legacy version inside the action diffusion policy for ease 

import torch
import numpy as np
from genesis_ILDP.model.common.noise_scheduler import NoiseScheduler
from genesis_ILDP.model.diffusion.conditioned_unet import Unet

class diffuser:
    def __init__(self, 
                 scheduler:NoiseScheduler,
                 backbone:Unet, 

                 diff_steps=100, 
                 num_inference_steps=100, 
                 diffuser_mode='ddpm', 
                 noise_intensity=1.0, 

                 predict_type='epsilon'):
        
        self.scheduler = scheduler
        self.betas, self.cum_alphas = self.scheduler.get_scheduler_values(diff_steps, )
        self.model = backbone

        self.diff_steps = diff_steps
        self.num_inference_steps = num_inference_steps
        self.diffuser_mode = diffuser_mode

        self.randn_clip_value = randn_clip_value
        self.denoised_clip_value = denoised_clip_value
        self.action_clip_value = action_clip_value

        self.predict_type = predict_type

    def step(self, x, cond, ):
         # cond is the img, agent_pos, and timesteps encoding which have been cat together
         # return the pred_noise for x using this cond

        result = self.model(x=x, cond=cond)
        if self.predict_type == 'epsilon':
            return result
        else: return x - result

    def denoise(self, x, cond):
        # default cond is worked as the global cond for denoising
        for t in reversed(range(self.diff_steps + 1)):
            if t > 0:
                # beta_tilde = self.betas[t] * 
                pass
        pass 







