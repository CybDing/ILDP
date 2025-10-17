from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import tqdm
import genesis as gs
from typing import Union

from genesis_ILDP.model.conditioned_unet import Unet
from genesis_ILDP.model.encoding import global_img_encoding, time_encoding, pos_encoding, merge_multimodal_encoding
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.utils.cuda import to_torch
from genesis_ILDP.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

class ActionDiffusionImagePolicy(BaseImagePolicy):
       
    def __init__(self,
                 shape_meta: dict,
                 normalizer: LinearNormalizer = None,
                 vision_backbone: str = 'custom',
                 vision_pretrained: bool = False,  # Whether to use pretrained vision backbone
                 diff_steps: int = 100,
                 scheduler_mode: str = 'Linear',
                 obs_steps: int = 2,
                 horizon: int = 16,
                 n_action_steps: int = 8,
                 n_obs_steps: int = 2,
                 encode_agent_pos: bool = False,  # Whether to encode agent_pos with MLP
                 ddim_steps: int = None,  # Number of DDIM steps (if None, auto-calculate)
                 noise_intensity: float = 0.0,  # DDIM noise intensity
                 crop_shape = (3, 76, 76),  # Crop dimensions (height, width) for random cropping
                 enable_crop = True,  # Whether to enable random cropping for data augmentation
                 img_encoding_dim = 512,
                 time_encoding_dim = 128,
                 pos_encoding_dim = None,
                 variance_threshold = None,
                 use_spatial_softmax: bool = False,  # Use spatial softmax pooling
                 spatial_softmax_temp: float = 1.0,  # Temperature for spatial softmax
                 vision_encoder_dropout: float = 0.0,  # Dropout rate for vision encoder MLP
                 pos_encoder_dropout: float = 0.0,  # Dropout rate for position encoder MLP
                 unet_dropout: float = 0.0,  # Dropout rate for UNet FiLM layers
                 time_encoder_dropout: float = 0.0,  # Dropout rate for time encoding MLP
                 **kwargs):
        super().__init__()
        
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]

        obs_shape_meta = shape_meta['obs']
        raw_img_shape = shape_meta['obs']['image']['shape'] # get the original image shape discarding the channel at the last position
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
        self.enable_crop = enable_crop
        self.crop_shape = crop_shape

        # Note: img_encoding_dim will be updated after encoder initialization
        self.time_encoding_dim = time_encoding_dim
        self.pos_encoding_dim = pos_encoding_dim

        if variance_threshold is not None: 
            self.variance_threshold = variance_threshold
        else: 
            self.variance_threshold = None

        if self.enable_crop:
            # Extract spatial dimensions for validation (assuming shape is [C, H, W])
            input_height, input_width = raw_img_shape[1], raw_img_shape[2]  # Skip channel dimension
            crop_height, crop_width = crop_shape[1], crop_shape[2]  # Skip channel dimension

            # Validate crop dimensions against spatial dimensions only
            if crop_height > input_height or crop_width > input_width:
                raise ValueError(f"Crop spatial dims ({crop_height}, {crop_width}) cannot be larger than input spatial dims ({input_height}, {input_width})")

            self.cropper = CropRandomizer(input_shape=raw_img_shape, crop_height=crop_height,
                                          crop_width=crop_width, num_crops=1, pos_enc=False)
            print(f"Random cropping enabled: {raw_img_shape} -> crop ({crop_height}, {crop_width})")
        else:
            self.cropper = None
            print(f"Random cropping disabled, using full image: {raw_img_shape}")

        # img_encoding_dim = img_encoding_dim # 1024 
        # time_encoding_dim = time_encoding_dim # 128
        if encode_agent_pos:
            if pos_encoding_dim is None:
                self.pos_encoding_dim =128 # default encoding dim for pos_encoding when enabling it
            dim_global_features = img_encoding_dim * 2 + time_encoding_dim + pos_encoding_dim * 2
        else:
            agent_pos_dim = 2 * n_obs_steps
            self.pos_encoding_dim = 2
            dim_global_features = img_encoding_dim * 2 + time_encoding_dim + agent_pos_dim

        # Initialize vision encoder first to get actual output dimensions
        # Extract spatial dimensions from crop_shape: (C, H, W)
        input_img_shape = (crop_shape[1], crop_shape[2]) if enable_crop else (raw_img_shape[1], raw_img_shape[2])

        self.imgs_encoding_net = global_img_encoding(
            in_channels=3,
            encoded_dim=img_encoding_dim,
            backbone=vision_backbone,
            pretrained=vision_pretrained,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_temp=spatial_softmax_temp,
            input_image_shape=input_img_shape,
            dropout=vision_encoder_dropout
        )

        # Get the actual output dimension from the encoder
        # (will be 1024 if spatial_softmax is used with img_encoding_dim=512)
        actual_img_encoding_dim = self.imgs_encoding_net.get_feature_dim()
        self.img_encoding_dim = actual_img_encoding_dim  # Update to actual dimension
        print(f"Vision encoder output dimension: {actual_img_encoding_dim} (spatial_softmax={use_spatial_softmax})")

        # Recalculate global features dimension with actual img encoding dim
        if encode_agent_pos:
            dim_global_features = actual_img_encoding_dim * 2 + time_encoding_dim + (pos_encoding_dim if pos_encoding_dim else 128) * 2
        else:
            dim_global_features = actual_img_encoding_dim * 2 + time_encoding_dim + agent_pos_dim

        self.conditioned_unet = Unet(dim_global_features, input_dim=self.action_dim, dropout=unet_dropout)  # forward(x, global_cond)

        # Initialize time encoding module (now a nn.Module with MLP processing)
        self.time_encoding_net = time_encoding(t_emb_dim=time_encoding_dim, dropout=time_encoder_dropout)
        self.merge_multimodal_encoding = merge_multimodal_encoding

        if encode_agent_pos:
            self.agent_pos_encoding = pos_encoding(encoded_dim=self.pos_encoding_dim, dropout=pos_encoder_dropout)
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
            # Create cosine schedule starting from timestep 1 to match other schedulers
            f_t = [np.cos((timestep / self.diff_steps + s) / (1 + s) * np.pi / 2) ** 2
                   for timestep in range(1, self.diff_steps + 1)]
            f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2

            # Calculate cum_alphas (these are the alpha_bar values)
            cum_alphas_list = [f / f_0 for f in f_t]

            # Calculate individual alphas from cum_alphas
            alphas = [cum_alphas_list[0]]  # alpha_1 = alpha_bar_1
            for i in range(1, len(cum_alphas_list)):
                alphas.append(cum_alphas_list[i] / cum_alphas_list[i-1])

            betas = torch.Tensor([1 - alpha for alpha in alphas], device=device)
            betas = torch.clamp(betas, max=0.999)  # Clip beta to prevent alpha from becoming 0
            cum_alphas = torch.Tensor(cum_alphas_list, device=device)
            return betas, cum_alphas

        else:
            raise ValueError(f"Unknown scheduler mode: {cur_mode}")

    def _apply_crop(self, images: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply cropping to images if enabled.

        Args:
            images: Input images tensor of shape (B, C, H, W)
            training: If True, apply random cropping. If False, apply center cropping.

        Returns:
            Cropped images tensor of shape (B, C, crop_height, crop_width)
        """
        if not self.enable_crop or self.cropper is None:
            return images

        if hasattr(self.cropper, 'to'):
            self.cropper = self.cropper.to(images.device)

        if training:
            self.cropper.train()  
        else:
            self.cropper.eval()  

        cropped_images = self.cropper(images)

        return cropped_images

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], recording_diffusion=False) -> Dict[str, torch.Tensor]:
        """
        Predict actions given observations.
        """
        # Normalize observations like in training.
        # 1) Ensure channels-first before normalization if needed.
        imgs = obs_dict['image']
        agent_pos = obs_dict['agent_pos']

        if len(imgs.shape) == 5 and imgs.shape[-1] == 3:
            print(f"Converting image dimensions from {imgs.shape} to channels-first format")
            imgs = imgs.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
            print(f"After conversion: {imgs.shape}")
        elif len(imgs.shape) == 4:
            print(f"WARNING: Unexpected 4D image tensor: {imgs.shape}")
            print(f"Expected 5D: [batch, time, height, width, channels] but got 4D")
            print("This suggests missing width dimension in environment observation!")
        else:
            print(f"WARNING: Unexpected image tensor shape: {imgs.shape} (expected 5D)")

        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
            print('[action_diffusion_policy] Warning: Input to policy for predicting action shape does not match with n_obs_steps')
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]
        elif imgs.shape[1] < self.n_obs_steps or imgs.shape[1] < self.n_obs_steps:
            raise ValueError("[action diffusion policy] Error! Input shape obs steps smaller than set n_obs_steps, could not predict actions")

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

        result = self.action_generation_ddpm(nimgs, nagent_pos, batch_size, recording_diffusion)

        # Handle return value based on recording_diffusion flag
        if recording_diffusion:
            action_pred, action_diffusion_buffer = result
        else:
            action_pred = result

        # Extract action steps for execution (typically the first n_action_steps)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        print(f"predict_action output shapes - action: {action.shape}, action_pred: {action_pred.shape}")

        result_dict = {
            'action': action,
            'action_pred': action_pred
        }

        if recording_diffusion:
            result_dict['action_diffusion_buffer'] = action_diffusion_buffer

        return result_dict

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        # Ensure buffers are on the same device as the policy
        self.normalizer.to(next(self.parameters()).device)

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
        # Sample timesteps from 1 to diff_steps, then convert to array indices
        random_t = torch.randint(1, self.diff_steps + 1, (batch_size,), device=action.device)
        random_noise = torch.randn_like(action)

        # Add noise to actions (forward diffusion process)
        # Ensure cum_alphas is on the same device as the action tensor
        cum_alphas_device = self.cum_alphas.to(action.device)
        # Convert timesteps to array indices: timestep t uses index t-1
        sqrt_alpha_cumprod = torch.sqrt(cum_alphas_device[random_t - 1]).reshape(batch_size, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - cum_alphas_device[random_t - 1]).reshape(batch_size, 1, 1)
        
        noisy_action = sqrt_alpha_cumprod * action + sqrt_one_minus_alpha_cumprod * random_noise

        # Encode observations and time
        t_encoding = torch.stack([self.time_encoding_net(t.item()) for t in random_t], dim=0)
        # Apply cropping before vision encoding (training mode)
        imgs_cropped = self._apply_crop(imgs_stack, training=True)
        imgs_encoding = self.imgs_encoding_net(imgs_cropped).reshape(batch_size, -1)

        if self.encode_agent_pos:
            pos_encoding = self.agent_pos_encoding(agent_pos_stack).reshape(batch_size, -1)
        else:
            # Use raw agent position - flatten across observation steps
            pos_encoding = agent_pos_stack.reshape(batch_size, -1)  # (B, n_obs_steps * 2)

        global_cond = self.merge_multimodal_encoding(imgs_encoding, pos_encoding, t_encoding)

        # Predict noise
        noisy_action_input = noisy_action.transpose(1, 2).contiguous()  # (B, Da, T) - ensure contiguous

        predicted_noise = self.conditioned_unet(noisy_action_input, global_cond)
        predicted_noise = predicted_noise.transpose(1, 2).contiguous()  # (B, T, Da) - ensure contiguous

        loss = torch.nn.functional.mse_loss(predicted_noise, random_noise)
        return loss
    
    def action_generation_ddpm(self, imgs, agent_pos, batch_size, recording_diffusion = False)\
        -> Union[torch.Tensor, tuple]:
        """DDPM sampling for action generation."""
        # Validate input shapes
        imgs_steps = imgs.shape[1]
        pos_steps = agent_pos.shape[1]
        assert imgs_steps == pos_steps == self.n_obs_steps, f"Expected {self.n_obs_steps} obs steps, got imgs: {imgs_steps}, pos: {pos_steps}"

        # Prepare observation features
        imgs_batched = imgs.reshape(-1, *imgs.shape[2:])
        pos_batched = agent_pos.reshape(-1, *agent_pos.shape[2:])

        # Apply cropping before vision encoding (inference mode)
        imgs_cropped = self._apply_crop(imgs_batched, training=False)
        imgs_features_batched = self.imgs_encoding_net(imgs_cropped)

        if self.encode_agent_pos:
            pos_features_batched = self.agent_pos_encoding(pos_batched)
        else:
            # Use raw agent position
            pos_features_batched = pos_batched  # No encoding, use raw values

        imgs_features = imgs_features_batched.reshape(batch_size, self.n_obs_steps * self.img_encoding_dim)

        pos_features = pos_features_batched.reshape(batch_size, self.n_obs_steps * self.pos_encoding_dim)

        # Initialize with random noise
        # print(f"action_generation_ddpm: batch_size={batch_size}, traj_shape={self.traj_shape}")
        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device)
        # print(f"action_generation_ddpm: predicted_trajs initial shape={predicted_trajs.shape}")

        # Move scheduler tensors to the correct device
        betas_device = self.betas.to(imgs.device)
        cum_alphas_device = self.cum_alphas.to(imgs.device)

        if recording_diffusion:
            action_buffer = [predicted_trajs] # which will hold for (B, horizons, Da)           

        for step_idx, t in enumerate(reversed(range(1, self.diff_steps + 1))):
            # Time encoding
            t_features = torch.stack([self.time_encoding_net(t).to(device=imgs.device)
                                    for _ in range(batch_size)], dim=0)

            global_features_t = self.merge_multimodal_encoding(imgs_features, pos_features, t_features)
            # print(f"[DEBUG DDPM] global_features_t.shape: {global_features_t.shape}")

            # Predict noise
            traj_input = predicted_trajs.transpose(1, 2).contiguous()  # (B, Da, T)
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)

            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()  # (B, T, Da)

            if t > 1:
                # Standard DDPM update
                # Timestep t uses array index t-1
                alpha_t = 1 - betas_device[t-1]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = betas_device[t-1] / torch.sqrt(1 - cum_alphas_device[t-1])
                mean = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)

                # Variance calculation: cum_alphas at timestep t-1 and t-2
                variance = betas_device[t-1] * (1 - cum_alphas_device[t-2]) / (1 - cum_alphas_device[t-1])

                noise = torch.randn_like(predicted_trajs)
                predicted_trajs = mean + torch.sqrt(variance) * noise
            else:
                alpha_t = 1 - betas_device[t-1]  # t=1 uses index 0
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = betas_device[t-1] / torch.sqrt(1 - cum_alphas_device[t-1])
                predicted_trajs = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)
            if recording_diffusion:
                action_buffer.append(predicted_trajs)  # append the predicted_trajs

            if torch.isnan(predicted_trajs).any():
                print(f"  NaN detected in predicted_trajs after DDPM step {step_idx}!")
                print(f"  Breaking early to prevent propagation...")
                break        
        
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

        if not recording_diffusion:
            return predicted_trajs_with_const
        else:
            action_diffusion_buffer = torch.stack(action_buffer, dim=0).transpose(dim0=0, dim1=1)
            # (diff_steps + 1, cur_envs, horizons, Da) -> (cur_envs, diff_steps + 1, horizons, Da)
            return (predicted_trajs_with_const, action_diffusion_buffer)
    
    def action_generation_ddim(self, imgs, agent_pos, batch_size, sample_steps=None, noise_intensity=0.0):
        """DDIM sampling for faster action generation."""
        if sample_steps is None:
            # Optimal step selection: uniform spacing for better coverage
            # DDIM uses actual timesteps, which go from 1 to diff_steps
            num_steps = min(20, max(10, self.diff_steps // 15))  # 10-20 steps
            sample_steps = np.linspace(1, self.diff_steps, num_steps, dtype=int).tolist()
            sample_steps = sorted(set(sample_steps))  # Remove duplicates and ensure sorted
        elif isinstance(sample_steps, int):
            # If integer provided, create that many evenly spaced steps
            num_steps = min(sample_steps, self.diff_steps)
            sample_steps = np.linspace(1, self.diff_steps, num_steps, dtype=int).tolist()
            sample_steps = sorted(set(sample_steps))  # Remove duplicates and ensure sorted
            
        imgs_batched = imgs.reshape(-1, *imgs.shape[2:])
        pos_batched = agent_pos.reshape(-1, *agent_pos.shape[2:])

        # Apply cropping before vision encoding (inference mode)
        imgs_cropped = self._apply_crop(imgs_batched, training=False)
        imgs_features_batched = self.imgs_encoding_net(imgs_cropped)

        if self.encode_agent_pos:
            pos_features_batched = self.agent_pos_encoding(pos_batched)
            pos_features = pos_features_batched.reshape(batch_size, self.n_obs_steps * self.pos_encoding_dim)
        else:
            # Use raw agent position
            pos_features_batched = pos_batched  # No encoding, use raw values
            pos_features = pos_features_batched.reshape(batch_size, self.n_obs_steps * self.pos_encoding_dim)

        imgs_features = imgs_features_batched.reshape(batch_size, self.n_obs_steps * self.img_encoding_dim)
        
        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device)

        # Move scheduler tensors to the correct device
        cum_alphas_device = self.cum_alphas.to(imgs.device)

        # DEBUG: Print scheduler info
        print(f"DDIM Sampling Debug Info:")
        print(f"  Scheduler mode: {self.scheduler_mode}")
        print(f"  Sample steps: {sample_steps}")
        print(f"  Noise intensity: {noise_intensity}")
        print(f"  cum_alphas min/max: {cum_alphas_device.min().item():.8e} / {cum_alphas_device.max().item():.8e}")
        print(f"  Smallest 5 cum_alphas: {cum_alphas_device.sort()[0][:5]}")
        print(f"  Initial predicted_trajs range: [{predicted_trajs.min().item():.3f}, {predicted_trajs.max().item():.3f}]")

        # DDIM reverse process
        for step_idx, i in enumerate(reversed(sample_steps)):
            t_features = torch.stack([self.time_encoding_net(i).to(device=imgs.device)
                                    for _ in range(batch_size)], dim=0)
            global_features_t = self.merge_multimodal_encoding(imgs_features, pos_features, t_features)

            traj_input = predicted_trajs.transpose(1, 2).contiguous()
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()

            # DDIM update equations
            # Arrays are 0-indexed for timesteps 1,2,...,diff_steps
            # So timestep i uses index i-1
            alpha_cumprod_t = cum_alphas_device[i-1]

            # For previous timestep: if i > 1, use i-2; if i == 1, use alpha_0 = 1.0
            if i > 1:
                alpha_cumprod_prev = cum_alphas_device[i-2]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=imgs.device)

            # DEBUG: Print values that could cause NaN
            if step_idx < 5 or i < 10 or torch.isnan(predicted_trajs).any():
                print(f"DDIM Step {step_idx}, timestep {i}:")
                print(f"  alpha_cumprod_t: {alpha_cumprod_t.item():.8e}")
                print(f"  alpha_cumprod_prev: {alpha_cumprod_prev.item():.8e}")
                print(f"  sqrt(alpha_cumprod_t): {torch.sqrt(alpha_cumprod_t).item():.8e}")
                print(f"  1/sqrt(alpha_cumprod_t): {(1.0/torch.sqrt(alpha_cumprod_t)).item():.8e}")
                print(f"  predicted_trajs has NaN: {torch.isnan(predicted_trajs).any()}")
                print(f"  predicted_noise_t has NaN: {torch.isnan(predicted_noise_t).any()}")
                print(f"  predicted_trajs range: [{predicted_trajs.min().item():.3f}, {predicted_trajs.max().item():.3f}]")
                print(f"  predicted_noise_t range: [{predicted_noise_t.min().item():.3f}, {predicted_noise_t.max().item():.3f}]")

            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            pred_x0 = (predicted_trajs - sqrt_one_minus_alpha_cumprod_t * predicted_noise_t) / sqrt_alpha_cumprod_t

            if torch.isnan(pred_x0).any() or step_idx < 5 or i < 10:
                print(f"  pred_x0 has NaN: {torch.isnan(pred_x0).any()}")
                print(f"  pred_x0 range: [{pred_x0.min().item():.3f}, {pred_x0.max().item():.3f}]")

            if i > 1:  # Not the final step (timestep 1)
                ratio_term = (1 - alpha_cumprod_t) * (alpha_cumprod_prev / alpha_cumprod_t)
                sqrt_term = 1 - alpha_cumprod_prev - ratio_term

                if sqrt_term < 0:
                    print(f"  WARNING: sqrt_term is negative: {sqrt_term.item():.8e}")

                noise_factor = noise_intensity * torch.sqrt(torch.clamp(sqrt_term, min=0.0))
                direction_xt = torch.sqrt(1 - alpha_cumprod_prev - noise_factor**2) * predicted_noise_t

                random_noise = torch.randn_like(predicted_trajs) if noise_intensity > 0 else 0
                predicted_trajs = torch.sqrt(alpha_cumprod_prev) * pred_x0 + direction_xt + noise_factor * random_noise
            else:
                # Final step: timestep 1 -> 0, deterministic
                predicted_trajs = pred_x0

            if torch.isnan(predicted_trajs).any():
                print(f"  ❌ NaN detected in predicted_trajs after step {step_idx}!")
                print(f"  Breaking early to prevent propagation...")
                break

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