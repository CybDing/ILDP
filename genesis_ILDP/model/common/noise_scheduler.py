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
            # Create cosine schedule for timesteps 1 to diff_steps
            # We need to generate diff_steps values, one for each timestep
            f_t = [np.cos((timestep / diff_steps + self.s) / (1 + self.s) * np.pi / 2) ** 2
                   for timestep in range(0, diff_steps+1)]  # Changed: start from 0 to get diff_steps values

            dtype = torch.float32 if (isinstance(self.device, torch.device) and self.device.type == 'mps') or (isinstance(self.device, str) and str(self.device).startswith('mps')) else torch.float32
            cum_alphas_raw = torch.tensor([float(f / f_t[0]) for f in f_t], device=self.device, dtype=dtype)

            # For timestep 1, we need to compute alpha_1 = cum_alpha_1 / cum_alpha_0
            # Since cum_alpha_0 = 1.0 (no noise at t=0), we can directly use cum_alphas_raw
            # But we need to compute incremental alphas for betas
            alphas = torch.cat([cum_alphas_raw[1:] / cum_alphas_raw[:-1]])
            betas = torch.clamp(1.0 - alphas, max=0.999)
            cum_alphas = torch.cumprod(1.0 - betas, dim=0)
            # print(cum_alphas)            

        else:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")
        return betas, cum_alphas

if __name__ == '__main__':
    scheduler = NoiseScheduler(mode='cosine', device='mps')
    # scheduler.get_scheduler_values(diff_steps=100)
    print(scheduler.get_scheduler_values(diff_steps=100)[1].shape)
