from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import tqdm

# Import from genesis_ILDP custom components
from genesis_ILDP.model.conditioned_unet import Unet
from genesis_ILDP.model.encoding import global_img_encoding, time_encoding, pos_encoding, merge_multimodal_encoding
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.utils.cuda import to_torch

# Import from diffusion_policy for standard components
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Encoding dimensions
dim_time_ebd = 256
dim_imgs_ebd = 1024
dim_agentPos_ebd = 512
dim_global_features = dim_time_ebd + dim_imgs_ebd + dim_agentPos_ebd  # 1792


class ActionDiffusionImagePolicy(BaseImagePolicy):
    """
    ActionDiffusion model moved from model/diffusion/ to policy/ folder
    and adapted to follow BaseImagePolicy interface with normalizer support.
    """
    
    def __init__(self, 
                 shape_meta: dict,
                 diff_steps: int = 100,
                 scheduler_mode: str = 'Linear',
                 obs_steps: int = 2,
                 horizon: int = 16,
                 n_action_steps: int = 8,
                 n_obs_steps: int = 2,
                 **kwargs):
        super().__init__()
        
        # Parse shape_meta like other diffusion policies
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        
        obs_shape_meta = shape_meta['obs']
        # Validate expected observation keys
        assert 'image' in obs_shape_meta, "ActionDiffusionImagePolicy requires 'image' in observations"
        assert 'agent_pos' in obs_shape_meta, "ActionDiffusionImagePolicy requires 'agent_pos' in observations"
        
        # Core diffusion parameters
        self.diff_steps = diff_steps
        self.scheduler_mode = scheduler_mode
        self.obs_steps = obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        
        # Network components
        self.conditioned_unet = Unet(dim_global_features, input_dim=self.action_dim)
        self.imgs_encoding_net = global_img_encoding(in_channels=3, encoded_dim=1024)
        self.time_encoding = time_encoding  # encoding function for diffusion steps
        self.agent_pos_encoding = pos_encoding(encoded_dim=512)
        
        # Noise scheduling
        self.betas, self.cum_alphas = self.NoiseScheduler(mode=self.scheduler_mode)
        self.traj_shape = (horizon, self.action_dim)
        
        # *** PLACEHOLDER FOR NORMALIZER IMPLEMENTATION ***
        # TODO: Add normalizer support here
        self.normalizer = LinearNormalizer()
        # You need to implement:
        # 1. self.set_normalizer(normalizer) method
        # 2. Normalization in predict_action() and compute_loss()
        # 3. Handle normalization for both obs and action data
        
        print(f"ActionDiffusionImagePolicy initialized:")
        print(f"- Action dim: {self.action_dim}")
        print(f"- Horizon: {horizon}")
        print(f"- Diffusion steps: {diff_steps}")
        print(f"- Observation steps: {obs_steps}")

    def NoiseScheduler(self, mode=None, beta_start=0.0001, beta_end=0.02, s=0.008):
        """Generate noise scheduling parameters."""
        cur_mode = self.scheduler_mode
        if mode is not None:
            cur_mode = mode
        
        if cur_mode == 'Linear':
            betas = torch.Tensor([beta_start + (beta_end - beta_start) / self.diff_steps * timestep 
                                 for timestep in range(1, self.diff_steps + 1)])
            alphas = 1.0 - betas
            cum_alphas = torch.cumprod(alphas, dim=0)
            return betas, cum_alphas
        
        elif cur_mode == 'RootLinear':
            betas = torch.Tensor([np.sqrt(beta_start + (beta_end - beta_start) / self.diff_steps * timestep) 
                                 for timestep in range(1, self.diff_steps + 1)])
            alphas = 1.0 - betas
            cum_alphas = torch.cumprod(alphas, dim=0)
            return betas, cum_alphas

        elif cur_mode == 'Cosine':
            f_t = [np.cos((timestep / self.diff_steps + s) / (1 + s) * np.pi / 2) ** 2 
                   for timestep in range(self.diff_steps)]
            f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2

            tilde_alpha_t = [f / f_0 for f in f_t]
            alphas = [tilde_alpha_t[0]]

            for i in range(1, len(tilde_alpha_t)):
                alphas.append(tilde_alpha_t[i] / tilde_alpha_t[i-1])
            
            betas = torch.Tensor([1 - alpha for alpha in alphas])
            cum_alphas = torch.Tensor(tilde_alpha_t)
            return betas, cum_alphas
        
        else:
            raise ValueError(f"Unknown scheduler mode: {cur_mode}")

    # ========== BaseImagePolicy Interface ==========
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict actions given observations.
        
        Args:
            obs_dict: Dictionary containing:
                - 'image': torch.Tensor of shape (B, To, C, H, W)  
                - 'agent_pos': torch.Tensor of shape (B, To, 2)
                
        Returns:
            Dictionary containing:
                - 'action': torch.Tensor of shape (B, Ta, Da) - actions for execution
                - 'action_pred': torch.Tensor of shape (B, T, Da) - full predicted trajectory
        """
        # *** TODO: IMPLEMENT NORMALIZER HERE ***
        # You need to add:
        # 1. nobs = self.normalizer.normalize(obs_dict)
        # 2. Use nobs instead of obs_dict in the rest of the function
        # 3. Unnormalize the predicted actions before returning
        
        imgs = obs_dict['image']
        agent_pos = obs_dict['agent_pos']
        batch_size = imgs.shape[0]
        
        # Use DDPM for action generation
        action_pred = self.action_generation_ddpm(imgs, agent_pos, batch_size)
        
        # Extract action steps for execution (typically the first n_action_steps)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the data normalizer."""
        # *** TODO: IMPLEMENT NORMALIZER SETTING HERE ***
        # You need to add:
        # 1. self.normalizer.load_state_dict(normalizer.state_dict())
        # 2. Ensure normalizer has keys for 'image', 'agent_pos', 'action'
        
        self.normalizer.load_state_dict(normalizer.state_dict())
        print("Normalizer set for ActionDiffusionImagePolicy")

    def reset(self):
        """Reset policy state (no state to reset for this policy)."""
        pass

    # ========== Training Interface ==========
    
    def compute_loss(self, batch):
        """
        Compute training loss for the diffusion model.
        
        Args:
            batch: Dictionary containing:
                - 'obs': Dict with 'image' and 'agent_pos'
                - 'action': Ground truth actions
                
        Returns:
            torch.Tensor: MSE loss between predicted and actual noise
        """
        # *** TODO: IMPLEMENT NORMALIZER HERE ***
        # You need to add:
        # 1. nobs = self.normalizer.normalize(batch['obs'])
        # 2. nactions = self.normalizer['action'].normalize(batch['action'])
        # 3. Use normalized data for loss computation
        
        imgs = batch['obs']['image']
        agent_pos = batch['obs']['agent_pos']
        action = batch['action']

        # Trim observations to obs_steps if needed
        if imgs.shape[1] > self.obs_steps:
            imgs = imgs[:, :self.obs_steps]
        if agent_pos.shape[1] > self.obs_steps:
            agent_pos = agent_pos[:, :self.obs_steps]
        
        # Prepare batch data
        batch_size = imgs.shape[0]
        
        # Stack observations for encoding
        imgs_stack = imgs.reshape(-1, *imgs.shape[2:])  # (B*To, C, H, W)
        agent_pos_stack = agent_pos.reshape(-1, *agent_pos.shape[2:])  # (B*To, 2)

        # Sample random timestep and noise
        random_t = torch.randint(0, self.diff_steps, (batch_size,), device=action.device)
        random_noise = torch.randn_like(action)

        # Add noise to actions (forward diffusion process)
        sqrt_alpha_cumprod = torch.sqrt(self.cum_alphas[random_t]).reshape(batch_size, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.cum_alphas[random_t]).reshape(batch_size, 1, 1)
        
        noisy_action = sqrt_alpha_cumprod * action + sqrt_one_minus_alpha_cumprod * random_noise

        # Encode observations and time
        t_encoding = torch.stack([self.time_encoding(t.item()) for t in random_t], dim=0)
        imgs_encoding = self.imgs_encoding_net(imgs_stack).reshape(batch_size, -1)
        pos_encoding = self.agent_pos_encoding(agent_pos_stack).reshape(batch_size, -1)
        global_cond = merge_multimodal_encoding(imgs_encoding, pos_encoding, t_encoding)

        # Predict noise
        noisy_action_input = noisy_action.transpose(1, 2)  # (B, Da, T)
        predicted_noise = self.conditioned_unet(noisy_action_input, global_cond)
        predicted_noise = predicted_noise.transpose(1, 2)  # (B, T, Da)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predicted_noise, random_noise)
        return loss
    
    def action_generation_ddpm(self, imgs, agent_pos, batch_size):
        """DDPM sampling for action generation."""
        # Validate input shapes
        imgs_steps = imgs.shape[1]
        pos_steps = agent_pos.shape[1]
        assert imgs_steps == pos_steps == self.obs_steps, f"Expected {self.obs_steps} obs steps, got imgs: {imgs_steps}, pos: {pos_steps}"

        # Prepare observation features
        imgs_batched = imgs.reshape(-1, *imgs.shape[2:])
        pos_batched = agent_pos.reshape(-1, *agent_pos.shape[2:])
        
        imgs_features_batched = self.imgs_encoding_net(imgs_batched)
        pos_features_batched = self.agent_pos_encoding(pos_batched)
        imgs_features = imgs_features_batched.reshape(batch_size, -1)
        pos_features = pos_features_batched.reshape(batch_size, -1)

        # Initialize with random noise
        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.diff_steps)):
            # Time encoding
            t_features = torch.stack([torch.tensor(self.time_encoding(t), device=imgs.device) 
                                    for _ in range(batch_size)], dim=0)
            global_features_t = merge_multimodal_encoding(imgs_features, pos_features, t_features)
            
            # Predict noise
            traj_input = predicted_trajs.transpose(1, 2)  # (B, Da, T)
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2)  # (B, T, Da)

            if t > 0:
                # Standard DDPM update
                alpha_t = 1 - self.betas[t]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = self.betas[t] / torch.sqrt(1 - self.cum_alphas[t])
                mean = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)
                
                variance = self.betas[t] * (1 - self.cum_alphas[t-1]) / (1 - self.cum_alphas[t])
                noise = torch.randn_like(predicted_trajs)
                predicted_trajs = mean + torch.sqrt(variance) * noise
            else:
                # Final step (deterministic)
                alpha_t = 1 - self.betas[t]
                predicted_trajs = (predicted_trajs - self.betas[t] / torch.sqrt(1 - self.cum_alphas[t]) * predicted_noise_t) / torch.sqrt(alpha_t)
        
        return predicted_trajs
    
    def action_generation_ddim(self, imgs, agent_pos, batch_size, sample_steps=None, noise_intensity=0.0):
        """DDIM sampling for faster action generation."""
        if sample_steps is None:
            sample_steps = list(range(0, self.diff_steps, self.diff_steps // 20))  # 20 steps default
            
        # Similar setup as DDPM
        imgs_batched = imgs.reshape(-1, *imgs.shape[2:])
        pos_batched = agent_pos.reshape(-1, *agent_pos.shape[2:])
        
        imgs_features_batched = self.imgs_encoding_net(imgs_batched)
        pos_features_batched = self.agent_pos_encoding(pos_batched)
        imgs_features = imgs_features_batched.reshape(batch_size, -1)
        pos_features = pos_features_batched.reshape(batch_size, -1)
        
        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device)
        
        # DDIM reverse process
        for i in reversed(sample_steps):
            t_features = torch.stack([torch.tensor(self.time_encoding(i), device=imgs.device) 
                                    for _ in range(batch_size)], dim=0)
            global_features_t = merge_multimodal_encoding(imgs_features, pos_features, t_features)
            
            traj_input = predicted_trajs.transpose(1, 2)
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2)

            # DDIM update equations
            alpha_cumprod_t = self.cum_alphas[i]
            alpha_cumprod_prev = self.cum_alphas[i-1] if i > 0 else torch.tensor(1.0)
            
            # Predicted x0
            pred_x0 = (predicted_trajs - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise_t) / torch.sqrt(alpha_cumprod_t)
            
            # Direction towards x_t
            if i > 0:
                noise_factor = noise_intensity * torch.sqrt(1 - alpha_cumprod_prev - (1 - alpha_cumprod_t) * (alpha_cumprod_prev / alpha_cumprod_t))
                direction_xt = torch.sqrt(1 - alpha_cumprod_prev - noise_factor**2) * predicted_noise_t
                
                random_noise = torch.randn_like(predicted_trajs) if noise_intensity > 0 else 0
                predicted_trajs = torch.sqrt(alpha_cumprod_prev) * pred_x0 + direction_xt + noise_factor * random_noise
            else:
                predicted_trajs = pred_x0

        return predicted_trajs