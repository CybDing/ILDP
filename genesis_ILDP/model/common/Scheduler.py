import numpy as np
import torch

class NoiseScheduler:

    def __init__(self, mode='cosine', beta_start=0.0001, beta_end=0.02, s=0.008, device = None):
        
        if device is None:
             device = 'cuda:0'
             print('[!] Warning: NoiseScheduler\'s device not configured, fall back to default cuda:0 device')

        self.mode = mode
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s
        self.device = device        

    def get_scheduler_values(self, diff_steps):

        if self.mode == 'linear':
            betas = torch.Tensor([self.beta_start + (self.beta_end - self.beta_start) / diff_steps * timestep
                                 for timestep in range(1, diff_steps + 1)], device=self.device)
            alphas = 1.0 - betas
            cum_alphas = torch.cumprod(alphas, dim=0)
            

        elif self.mode == 'root_linear':
            betas = torch.Tensor([np.sqrt(self.beta_start + (self.beta_end - self.beta_start) / diff_steps * timestep)
                                 for timestep in range(1, diff_steps + 1)], device=self.device)
            alphas = 1.0 - betas
            cum_alphas = torch.cumprod(alphas, dim=0)
            

        elif self.mode == 'cosine':
            # Create cosine schedule starting from timestep 1 to match other schedulers
            f_t = [np.cos((timestep / diff_steps + self.s) / (1 + self.s) * np.pi / 2) ** 2
                   for timestep in range(1, diff_steps + 1)]
            f_0 = np.cos(self.s / (1 + self.s) * np.pi / 2) ** 2

            cum_alphas_raw = [f / f_0 for f in f_t]
            alphas = cum_alphas_raw[1:] / cum_alphas_raw[0:-1]

            betas = torch.Tensor([1 - alpha for alpha in alphas], device=self.device)
            betas = torch.clamp(betas, max=0.999)  # Clip beta to prevent alpha from becoming 0
            cum_alphas = torch.cumprod(betas, device=self.cur_device) # fix the cum_alphas using the new betas being clamped here 
            

        else:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")
        return betas, cum_alphas
    