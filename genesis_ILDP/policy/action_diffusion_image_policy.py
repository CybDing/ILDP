from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import tqdm
import genesis as gs

# Import from genesis_ILDP custom components
from genesis_ILDP.model.conditioned_unet import Unet
from genesis_ILDP.model.encoding import global_img_encoding, time_encoding, pos_encoding, merge_multimodal_encoding
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.utils.cuda import to_torch
from genesis_ILDP.dataset.pusht_image_dataset import PushTImageDataset
# Import from diffusion_policy for standard components
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Encoding dimensions
dim_time_ebd = 128
dim_imgs_ebd = 1024  # per observation step
dim_agentPos_ebd = 512  # per observation step
# For n_obs_steps=2: time(256) + imgs(2*1024) + agent_pos(2*512) = 256 + 2048 + 1024 = 3328
n_obs_steps_default = 2
dim_global_features = dim_time_ebd + (dim_imgs_ebd + dim_agentPos_ebd) * n_obs_steps_default  # 3328


class ActionDiffusionImagePolicy(BaseImagePolicy):
       
    def __init__(self,
                 shape_meta: dict,
                 normalizer: LinearNormalizer = None,
                 vision_backbone: str = 'custom',
                 diff_steps: int = 100,
                 scheduler_mode: str = 'Linear',
                 obs_steps: int = 2,
                 horizon: int = 16,
                 n_action_steps: int = 8,
                 n_obs_steps: int = 2,
                 encode_agent_pos: bool = False,  # Whether to encode agent_pos with MLP
                 ddim_steps: int = None,  # Number of DDIM steps (if None, auto-calculate)
                 noise_intensity: float = 0.0,  # DDIM noise intensity
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
        
        self.diff_steps = diff_steps
        self.scheduler_mode = scheduler_mode
        self.obs_steps = obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.encode_agent_pos = encode_agent_pos
        self.ddim_steps = ddim_steps
        self.noise_intensity = noise_intensity

        # Calculate global feature dimension based on encoding choice
        img_encoding_dim = 1024 # 1024 
        time_encoding_dim = 128 #
        if encode_agent_pos:
            agent_pos_encoding_dim = 512
            dim_global_features = img_encoding_dim * 2 + time_encoding_dim + agent_pos_encoding_dim
        else:
            # Raw agent_pos has 2 dimensions per observation step
            agent_pos_dim = 2 * n_obs_steps
            dim_global_features = img_encoding_dim * 2 + time_encoding_dim + agent_pos_dim

        self.conditioned_unet = Unet(dim_global_features, input_dim=self.action_dim)
        self.imgs_encoding_net = global_img_encoding(in_channels=3, encoded_dim=img_encoding_dim, backbone=vision_backbone)
        self.time_encoding = time_encoding  # encoding function for diffusion steps

        if encode_agent_pos:
            self.agent_pos_encoding = pos_encoding(encoded_dim=512)
        else:
            self.agent_pos_encoding = None  # Use raw agent_pos

        self.betas, self.cum_alphas = self.NoiseScheduler(mode=self.scheduler_mode)
        self.traj_shape = (horizon, self.action_dim)
        
        # Initialize normalizer - will be set properly in workspace
        self.normalizer = LinearNormalizer()
        if normalizer is not None:
            self.set_normalizer(normalizer)
        # Note: normalizer will be set via set_normalizer() call from workspace
        # after dataset provides a properly fitted normalizer
        
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

        # Get device from model parameters
        device = next(self.parameters()).device

        if cur_mode == 'Linear':
            betas = torch.Tensor([beta_start + (beta_end - beta_start) / self.diff_steps * timestep
                                 for timestep in range(1, self.diff_steps + 1)], device=device)
            alphas = 1.0 - betas
            cum_alphas = torch.cumprod(alphas, dim=0)
            return betas, cum_alphas

        elif cur_mode == 'RootLinear':
            betas = torch.Tensor([np.sqrt(beta_start + (beta_end - beta_start) / self.diff_steps * timestep)
                                 for timestep in range(1, self.diff_steps + 1)], device=device)
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

            betas = torch.Tensor([1 - alpha for alpha in alphas], device=device)
            cum_alphas = torch.Tensor(tilde_alpha_t, device=device)
            return betas, cum_alphas

        else:
            raise ValueError(f"Unknown scheduler mode: {cur_mode}")

    # ========== BaseImagePolicy Interface ==========
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict actions given observations.
        """
        # Normalize observations like in training.
        # 1) Ensure channels-first before normalization if needed.
        imgs = obs_dict['image']
        agent_pos = obs_dict['agent_pos']

        # Debug print to understand input shapes during rollout
        print(f"predict_action input shapes - imgs: {imgs.shape}, agent_pos: {agent_pos.shape}")

        if len(imgs.shape) == 5 and imgs.shape[-1] == 3:
            print(f"Converting image dimensions from {imgs.shape} to channels-first format")
            imgs = imgs.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
            print(f"After conversion: {imgs.shape}")

        # 2) Trim to expected observation steps
        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]

        # 3) Normalize on the current device
        device = next(self.parameters()).device
        nobs = self.normalizer.normalize({
            'image': imgs.to(device),
            'agent_pos': agent_pos.to(device),
        })

        nimgs = nobs['image']
        nagent_pos = nobs['agent_pos']
        batch_size = nimgs.shape[0]

        # Use DDIM for faster action generation (sampler returns UNNORMALIZED actions already)
        # Use configured DDIM parameters if available
        # ddim_steps = self.ddim_steps if self.ddim_steps is not None else None
        # noise_intensity = self.noise_intensity if hasattr(self, 'noise_intensity') else 0.0

        # action_pred = self.action_generation_ddim(nimgs, nagent_pos, batch_size,
        #                                          sample_steps=ddim_steps, noise_intensity=noise_intensity)

        action_pred = self.action_generation_ddpm(nimgs, nagent_pos, batch_size)

        # Extract action steps for execution (typically the first n_action_steps)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        print(f"predict_action output shapes - action: {action.shape}, action_pred: {action_pred.shape}")

        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the data normalizer."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        # Ensure buffers are on the same device as the policy
        self.normalizer.to(next(self.parameters()).device)

        # Check that required keys exist in the normalizer
        if hasattr(self.normalizer, 'params_dict') and len(self.normalizer.params_dict) > 0:
            assert 'image' in self.normalizer.params_dict, "Normalizer missing 'image' key"
            assert 'agent_pos' in self.normalizer.params_dict, "Normalizer missing 'agent_pos' key"
            assert 'action' in self.normalizer.params_dict, "Normalizer missing 'action' key"
            print("Normalizer set for ActionDiffusionImagePolicy")
        else:
            print("Warning: Empty normalizer set - will be configured later by workspace")

    def reset(self):
        """Reset policy state (no state to reset for this policy)."""
        pass
    
    def compute_loss(self, batch):
        """
        Compute training loss for the diffusion model.
        
        Args:
            batch: Dictionary containing:
                - 'obs': Dict with 'image' and 'agent_pos'
                - 'action': Ground truth actions
                
        """
        # Input shape here is(B * n_action_steps * (*dim))
        # Normalize observations and actions separately like diffusion_policy
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        imgs = nobs['image']
        agent_pos = nobs['agent_pos']
        action = nactions

        # Trim observations to obs_steps if needed
        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]
        
        batch_size = imgs.shape[0]
        
        # Stack observations for encoding
        imgs_stack = imgs.reshape(-1, *imgs.shape[2:])  # (B*To, C, H, W)
        agent_pos_stack = agent_pos.reshape(-1, *agent_pos.shape[2:])  # (B*To, 2)

        # Sample random timestep and noise
        random_t = torch.randint(0, self.diff_steps, (batch_size,), device=action.device)
        random_noise = torch.randn_like(action)

        # Add noise to actions (forward diffusion process)
        # Ensure cum_alphas is on the same device as the action tensor
        cum_alphas_device = self.cum_alphas.to(action.device)
        sqrt_alpha_cumprod = torch.sqrt(cum_alphas_device[random_t]).reshape(batch_size, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - cum_alphas_device[random_t]).reshape(batch_size, 1, 1)
        
        noisy_action = sqrt_alpha_cumprod * action + sqrt_one_minus_alpha_cumprod * random_noise

        # Encode observations and time
        t_encoding = torch.stack([self.time_encoding(t.item()) for t in random_t], dim=0)
        imgs_encoding = self.imgs_encoding_net(imgs_stack).reshape(batch_size, -1)

        if self.encode_agent_pos:
            pos_encoding = self.agent_pos_encoding(agent_pos_stack).reshape(batch_size, -1)
        else:
            # Use raw agent position - flatten across observation steps
            pos_encoding = agent_pos_stack.reshape(batch_size, -1)  # (B, n_obs_steps * 2)

        global_cond = merge_multimodal_encoding(imgs_encoding, pos_encoding, t_encoding)

        # Predict noise
        noisy_action_input = noisy_action.transpose(1, 2).contiguous()  # (B, Da, T) - ensure contiguous

        predicted_noise = self.conditioned_unet(noisy_action_input, global_cond)
        predicted_noise = predicted_noise.transpose(1, 2).contiguous()  # (B, T, Da) - ensure contiguous

        loss = torch.nn.functional.mse_loss(predicted_noise, random_noise)
        return loss
    
    def action_generation_ddpm(self, imgs, agent_pos, batch_size):
        """DDPM sampling for action generation."""
        # Validate input shapes
        imgs_steps = imgs.shape[1]
        pos_steps = agent_pos.shape[1]
        assert imgs_steps == pos_steps == self.n_obs_steps, f"Expected {self.n_obs_steps} obs steps, got imgs: {imgs_steps}, pos: {pos_steps}"

        # Prepare observation features
        imgs_batched = imgs.reshape(-1, *imgs.shape[2:])
        pos_batched = agent_pos.reshape(-1, *agent_pos.shape[2:])

        imgs_features_batched = self.imgs_encoding_net(imgs_batched)

        if self.encode_agent_pos:
            pos_features_batched = self.agent_pos_encoding(pos_batched)
        else:
            # Use raw agent position
            pos_features_batched = pos_batched  # No encoding, use raw values

        imgs_features = imgs_features_batched.reshape(batch_size, self.n_obs_steps * 1024)

        if self.encode_agent_pos:
            pos_features = pos_features_batched.reshape(batch_size, self.n_obs_steps * 512)
        else:
            # Raw agent_pos is already (batch_size * n_obs_steps, 2)
            pos_features = pos_features_batched.reshape(batch_size, self.n_obs_steps * 2)

        # Initialize with random noise
        print(f"action_generation_ddpm: batch_size={batch_size}, traj_shape={self.traj_shape}")
        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device)
        print(f"action_generation_ddpm: predicted_trajs initial shape={predicted_trajs.shape}")

        # Move scheduler tensors to the correct device
        betas_device = self.betas.to(imgs.device)
        cum_alphas_device = self.cum_alphas.to(imgs.device)

        # Reverse diffusion process
        for t in reversed(range(self.diff_steps)):
            # Time encoding
            t_features = torch.stack([self.time_encoding(t).to(device=imgs.device)
                                    for _ in range(batch_size)], dim=0)
            global_features_t = merge_multimodal_encoding(imgs_features, pos_features, t_features)
            
            # Predict noise
            traj_input = predicted_trajs.transpose(1, 2).contiguous()  # (B, Da, T)
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()  # (B, T, Da)

            if t > 0:
                # Standard DDPM update
                alpha_t = 1 - betas_device[t]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = betas_device[t] / torch.sqrt(1 - cum_alphas_device[t])
                mean = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)

                variance = betas_device[t] * (1 - cum_alphas_device[t-1]) / (1 - cum_alphas_device[t])
                noise = torch.randn_like(predicted_trajs)
                predicted_trajs = mean + torch.sqrt(variance) * noise
            else:
                # Final step (deterministic)
                alpha_t = 1 - betas_device[t]
                predicted_trajs = (predicted_trajs - betas_device[t] / torch.sqrt(1 - cum_alphas_device[t]) * predicted_noise_t) / torch.sqrt(alpha_t)
        
        action_dict = {"action": predicted_trajs}
        action_unnormalized = self.normalizer.unnormalize(action_dict)
        predicted_trajs_unnormalized = action_unnormalized["action"]
        
        # Add constant dimension (0.27) to the end
        # predicted_trajs_unnormalized shape: (batch_size, horizon, action_dim)
        batch_size, horizon, action_dim = predicted_trajs_unnormalized.shape
        const_dim = torch.full((batch_size, horizon, 1), 0.27, 
                              device=predicted_trajs_unnormalized.device, 
                              dtype=predicted_trajs_unnormalized.dtype)
        predicted_trajs_with_const = torch.cat([predicted_trajs_unnormalized, const_dim], dim=-1)
        
        return predicted_trajs_with_const
    
    def action_generation_ddim(self, imgs, agent_pos, batch_size, sample_steps=None, noise_intensity=0.0):
        """DDIM sampling for faster action generation."""
        if sample_steps is None:
            # Optimal step selection: uniform spacing for better coverage
            num_steps = min(20, max(10, self.diff_steps // 15))  # 10-20 steps
            sample_steps = np.linspace(0, self.diff_steps - 1, num_steps, dtype=int).tolist()
            sample_steps = sorted(set(sample_steps))  # Remove duplicates and ensure sorted
        elif isinstance(sample_steps, int):
            # If integer provided, create that many evenly spaced steps
            num_steps = min(sample_steps, self.diff_steps)
            sample_steps = np.linspace(0, self.diff_steps - 1, num_steps, dtype=int).tolist()
            sample_steps = sorted(set(sample_steps))  # Remove duplicates and ensure sorted
            
        imgs_batched = imgs.reshape(-1, *imgs.shape[2:])
        pos_batched = agent_pos.reshape(-1, *agent_pos.shape[2:])
        
        imgs_features_batched = self.imgs_encoding_net(imgs_batched)

        if self.encode_agent_pos:
            pos_features_batched = self.agent_pos_encoding(pos_batched)
            pos_features = pos_features_batched.reshape(batch_size, self.n_obs_steps * 512)
        else:
            # Use raw agent position
            pos_features_batched = pos_batched  # No encoding, use raw values
            pos_features = pos_features_batched.reshape(batch_size, self.n_obs_steps * 2)

        imgs_features = imgs_features_batched.reshape(batch_size, self.n_obs_steps * 1024)
        
        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device)

        # Move scheduler tensors to the correct device
        cum_alphas_device = self.cum_alphas.to(imgs.device)

        # DDIM reverse process
        for i in reversed(sample_steps):
            t_features = torch.stack([self.time_encoding(i).to(device=imgs.device)
                                    for _ in range(batch_size)], dim=0)
            global_features_t = merge_multimodal_encoding(imgs_features, pos_features, t_features)
            
            traj_input = predicted_trajs.transpose(1, 2).contiguous()
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()

            # DDIM update equations
            alpha_cumprod_t = cum_alphas_device[i]
            alpha_cumprod_prev = cum_alphas_device[i-1] if i > 0 else torch.tensor(1.0, device=imgs.device)
            
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

        action_dict = {"action": predicted_trajs}
        action_unnormalized = self.normalizer.unnormalize(action_dict)
        predicted_trajs_unnormalized = action_unnormalized["action"]
        
        # Add constant dimension (0.27) to the end
        # predicted_trajs_unnormalized shape: (batch_size, horizon, action_dim)
        batch_size, horizon, action_dim = predicted_trajs_unnormalized.shape
        const_dim = torch.full((batch_size, horizon, 1), 0.27, 
                              device=predicted_trajs_unnormalized.device, 
                              dtype=predicted_trajs_unnormalized.dtype)
        predicted_trajs_with_const = torch.cat([predicted_trajs_unnormalized, const_dim], dim=-1)
        
        return predicted_trajs_with_const