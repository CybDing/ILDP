from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import tqdm
import genesis as gs
from typing import Union

from genesis_ILDP.model.diffusion.conditioned_unet import Unet
from genesis_ILDP.model.encoding import time_encoding, pos_encoding, merge_multimodal_encoding
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.utils.cuda import to_torch
from genesis_ILDP.dataset.pusht_image_dataset import PushTImageDataset
from genesis_ILDP.model.common.noise_scheduler import NoiseScheduler
from genesis_ILDP.model.vision.cnn_encoding import img_encoding_cnn
from genesis_ILDP.model.common.modules import RandomShiftsAug, RandomPosShifter

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer


class ActionDiffusionImagePolicy(BaseImagePolicy):
       
    def __init__(self,
                 shape_meta: dict,
                 cropper: CropRandomizer,
                 pos_encoding: pos_encoding,
                 time_encoding: time_encoding,
                 imgs_encoding_net: img_encoding_cnn,
                 conditioned_unet: Unet,
                 noise_scheduler: NoiseScheduler,
                 normalizer: LinearNormalizer=None,

                 diff_steps: int = 100,
                 obs_steps: int = 2,
                 horizon: int = 16,
                 n_action_steps: int = 8,
                 n_obs_steps: int = 2,

                 noise_intensity: float = 1.0,

                 use_ddpm = True,
                 ddim_steps: int = None,

                 randn_clip_value: float | tuple | None = 5.0,
                 denoised_clip_value: float | tuple | None = 10.0,
                 action_clip_value: float | tuple | None = None,

                 shift_augmenter: RandomShiftsAug = None,
                 pos_shifter: RandomPosShifter = None,

                 **kwargs):
        super().__init__()
        
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]

        obs_shape_meta = shape_meta['obs']
        
        assert 'image' in obs_shape_meta, "ActionDiffusionImagePolicy requires 'image' in observations"
        assert 'agent_pos' in obs_shape_meta, "ActionDiffusionImagePolicy requires 'agent_pos' in observations"
        
        self.diff_steps = diff_steps
        self.obs_steps = obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.ddim_steps = ddim_steps
        self.noise_intensity = noise_intensity
        self.use_ddpm = use_ddpm

        self.randn_clip_value = randn_clip_value
        self.denoised_clip_value = denoised_clip_value
        self.action_clip_value = action_clip_value

        self.cropper = cropper
        if cropper is None: self.enable_crop = False
        else: self.enable_crop = True

        self.shift_augmenter = shift_augmenter
        self.pos_shifter = pos_shifter

        self.imgs_encoding_net = imgs_encoding_net

        self.conditioned_unet = conditioned_unet

        self.time_encoding = time_encoding
        self.merge_multimodal_encoding = merge_multimodal_encoding

        self.pos_encoding = pos_encoding

        if self.pos_encoding is None: self.encode_agent_pos = False
        else: self.encode_agent_pos = True

        betas, cum_alphas = noise_scheduler.get_scheduler_values(self.diff_steps)
        self.register_buffer('betas', betas)
        self.register_buffer('cum_alphas', cum_alphas)

        self.traj_shape = (horizon, self.action_dim)
        
        self.normalizer = LinearNormalizer()
        if normalizer is not None:
            self.set_normalizer(normalizer)

        print(f"------- ActionDiffusionImagePolicy initialized: -------")
        print(f"- Action dim: {self.action_dim}")
        print(f"- Horizon: {horizon}")
        print(f"- Diffusion steps: {diff_steps}")
        print(f"- Observation steps: {obs_steps}")
        print(f"- Clip settings: randn={self.randn_clip_value}, denoised={self.denoised_clip_value}, action={self.action_clip_value}")

    # --------------------
    # Helper clamp methods
    # --------------------
    def _clip_tensor(self, x: torch.Tensor, clip_value):
        """Clamp tensor x by clip_value if provided.
        - If clip_value is a scalar (float/int): clamp to [-clip_value, +clip_value]
        - If clip_value is a tuple/list of (min, max): clamp to [min, max]
        - If clip_value is None: return x unchanged
        """
        if clip_value is None:
            return x
        if isinstance(clip_value, (tuple, list)) and len(clip_value) == 2:
            min_v, max_v = clip_value
            return torch.clamp(x, min=min_v, max=max_v)
        try:
            v = float(clip_value)
        except Exception:
            return x
        return torch.clamp(x, min=-v, max=v)

    def _get_action_clip_bounds(self, device, dtype):
        """Expect action_clip_value as (min_list, max_list), each length == action_dim.
        Returns tensors shaped (1, 1, Da) for broadcasting.
        """
        v = self.action_clip_value
        if v is None:
            return None, None
        if not (isinstance(v, (tuple, list)) and len(v) == 2):
            raise ValueError("action_clip_value must be (min_list, max_list)")
        min_list, max_list = v
        Da = self.action_dim
        min_arr = torch.as_tensor(min_list, device=device, dtype=dtype)
        max_arr = torch.as_tensor(max_list, device=device, dtype=dtype)
        if min_arr.numel() != Da or max_arr.numel() != Da:
            raise ValueError(f"Length of min/max must equal action_dim={Da}, got {min_arr.numel()}/{max_arr.numel()}")
        return min_arr.view(1, 1, Da), max_arr.view(1, 1, Da)

    def _apply_crop(self, images: torch.Tensor, training: bool = True, generator=None) -> torch.Tensor:
        if not self.enable_crop or self.cropper is None:
            return images

        if training:
            self.cropper.train()
        else:
            self.cropper.eval()

        cropped_images = self.cropper(images, generator=generator)
        return cropped_images
    
    def _preprocess_obs(self, imgs, agent_pos):
        # Reorder to channels-first if needed: [B, T, H, W, C] -> [B, T, C, H, W]
        if len(imgs.shape) == 5 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 1, 4, 2, 3)
        elif len(imgs.shape) != 5:
            print(f"[ActionDiffusionImagePolicy] WARNING: unexpected image tensor shape: {imgs.shape} (expected 5D)")

        # Enforce n_obs_steps length consistently on both modalities
        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
            print('[ActionDiffusionImagePolicy] Warning: trimming images to n_obs_steps')
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]
            print('[ActionDiffusionImagePolicy] Warning: trimming agent_pos to n_obs_steps')
        if imgs.shape[1] < self.n_obs_steps or agent_pos.shape[1] < self.n_obs_steps:
            raise ValueError("[ActionDiffusionImagePolicy] Error! obs steps smaller than n_obs_steps; cannot predict actions")
        
        return imgs, agent_pos
    
    def _encoding_obs(self, img, agent_pos, training=True, generator=None) -> tuple:
        batch_size = img.shape[0]
        img = img.reshape(-1, *img.shape[-3:])
        agent_pos = agent_pos.reshape(-1, agent_pos.shape[-1])

        img = self._apply_crop(img, training=training, generator=generator)

        img_encoded = self.imgs_encoding_net(img)
        if self.encode_agent_pos:
            agent_pos_encoded = self.pos_encoding(agent_pos)
        else: agent_pos_encoded = agent_pos

        img_feature = img_encoded.reshape(batch_size, -1)
        agent_pos_feature = agent_pos_encoded.reshape(batch_size, -1)

        return img_feature, agent_pos_feature

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], recording_diffusion=False, generator=None) -> Dict[str, torch.Tensor]:
        """
        Predict actions given observations.
        """
        imgs = obs_dict['image']
        agent_pos = obs_dict['agent_pos']

        imgs, agent_pos = self._preprocess_obs(imgs, agent_pos)

        device = next(self.parameters()).device

        # Generator device must match tensor device (MPS/CUDA/CPU)
        if generator is not None and generator.device != device:
            new_generator = torch.Generator(device=device)
            new_generator.set_state(generator.get_state())
            generator = new_generator

        nobs = self.normalizer.normalize({
            'image': imgs.to(device),
            'agent_pos': agent_pos.to(device),
        })

        nimgs = nobs['image']
        nagent_pos = nobs['agent_pos']
        batch_size = nimgs.shape[0]

        if self.use_ddpm:
            result = self.action_generation_ddpm(nimgs, nagent_pos, batch_size, recording_diffusion, generator=generator)

        else:
            if self.ddim_steps is None:
                raise ValueError('ddim steps are not configured for ddim sampling')
            if self.noise_intensity is None:
                raise ValueError('noise intensity is not configured for ddim sampling')
            result = self.action_generation_ddim(nimgs, nagent_pos, batch_size, sample_steps=self.ddim_steps, noise_intensity=self.noise_intensity, generator=generator)

        # TODO check whether ddim could record actions
                    
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
        imgs = batch['obs']['image']
        agent_pos = batch['obs']['agent_pos']

        imgs, agent_pos = self._preprocess_obs(imgs, agent_pos)
        device = next(self.parameters()).device

        # Apply augmentations (training only)
        if self.shift_augmenter is not None:
            B, T = imgs.shape[:2]
            imgs_flat = imgs.reshape(-1, *imgs.shape[-3:])
            imgs_flat, shift_pixels = self.shift_augmenter(imgs_flat)
            imgs = imgs_flat.reshape(B, T, *imgs_flat.shape[-3:])

        if self.pos_shifter is not None:
            agent_pos, shift_value = self.pos_shifter(agent_pos)
            action = batch['action'].to(device).clone()
            action = action + shift_value.unsqueeze(1)
        else:
            action = batch['action'].to(device)

        nobs = self.normalizer.normalize({
            'image': imgs.to(device),
            'agent_pos': agent_pos.to(device)
        })
        nactions = self.normalizer['action'].normalize(action)

        nimgs = nobs['image']
        nagent_pos = nobs['agent_pos']
        action = nactions
        batch_size = nimgs.shape[0]

        # Sample random timestep and noise
        random_t = torch.randint(1, self.diff_steps + 1, (batch_size,), device=action.device)
        random_noise = torch.randn(action.shape, device=action.device, dtype=action.dtype)
        random_noise = self._clip_tensor(random_noise, self.randn_clip_value)

        sqrt_alpha_cumprod = torch.sqrt(self.cum_alphas[random_t - 1]).reshape(batch_size, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.cum_alphas[random_t - 1]).reshape(batch_size, 1, 1)
        
        noisy_action = sqrt_alpha_cumprod * action + sqrt_one_minus_alpha_cumprod * random_noise

        # Encode observations and time
        t_encoding = torch.stack([self.time_encoding(t.item()) for t in random_t], dim=0).to(device)
        imgs_encoding, pos_encoding = self._encoding_obs(nimgs, nagent_pos, training=True)

        global_cond = self.merge_multimodal_encoding(imgs_encoding, pos_encoding, t_encoding)

        noisy_action_input = noisy_action.transpose(1, 2).contiguous() 

        predicted_noise = self.conditioned_unet(noisy_action_input, global_cond)
        predicted_noise = predicted_noise.transpose(1, 2).contiguous()  # (B, T, Da)

        loss = torch.nn.functional.mse_loss(predicted_noise, random_noise)
        return loss
    
    def action_generation_ddpm(self, imgs, agent_pos, batch_size, recording_diffusion=False, generator=None)\
        -> Union[torch.Tensor, tuple]:
        """DDPM sampling for action generation."""
        imgs_steps = imgs.shape[1]
        pos_steps = agent_pos.shape[1]
        assert imgs_steps == pos_steps == self.n_obs_steps, f"Expected {self.n_obs_steps} obs steps, got imgs: {imgs_steps}, pos: {pos_steps}"

        imgs_features, pos_features = self._encoding_obs(imgs, agent_pos, training=False, generator=generator)

        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device, generator=generator)
        predicted_trajs = self._clip_tensor(predicted_trajs, self.randn_clip_value)

        if recording_diffusion:
            action_buffer = [predicted_trajs.clone()] # snapshot (B, horizons, Da)           

        for step_idx, t in enumerate(reversed(range(1, self.diff_steps + 1))):
            # Time encoding
            t_features = torch.stack([self.time_encoding(t).to(device=imgs.device)
                                    for _ in range(batch_size)], dim=0)
            global_features_t = self.merge_multimodal_encoding(imgs_features, pos_features, t_features)

            # Predict noise
            traj_input = predicted_trajs.transpose(1, 2).contiguous()  # (B, Da, T)
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)

            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()  # (B, T, Da)

            if t > 1:
                alpha_t = 1 - self.betas[t-1]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = self.betas[t-1] / torch.sqrt(1 - self.cum_alphas[t-1])
                mean = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)

                variance = self.betas[t-1] * (1 - self.cum_alphas[t-2]) / (1 - self.cum_alphas[t-1])

                noise = torch.randn_like(predicted_trajs, generator=generator)
                noise = self._clip_tensor(noise, self.randn_clip_value)
                predicted_trajs = mean + torch.sqrt(variance) * noise
                predicted_trajs = self._clip_tensor(predicted_trajs, self.denoised_clip_value)
            else:
                alpha_t = 1 - self.betas[t-1]  # t=1 uses index 0
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = self.betas[t-1] / torch.sqrt(1 - self.cum_alphas[t-1])
                predicted_trajs = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)
                predicted_trajs = self._clip_tensor(predicted_trajs, self.denoised_clip_value)
            if recording_diffusion:
                action_buffer.append(predicted_trajs.clone())  # append snapshot of predicted_trajs

            if torch.isnan(predicted_trajs).any():
                print(f"  NaN detected in predicted_trajs after DDPM step {step_idx}!")
                print(f"  Breaking early to prevent propagation...")
                break        
        
        action_dict = {"action": predicted_trajs}
        action_unnormalized = self.normalizer.unnormalize(action_dict)
        predicted_trajs_unnormalized = action_unnormalized["action"]
        # Optional: clamp final actions in real action space (support per-dim bounds)
        if self.action_clip_value is not None:
            amin, amax = self._get_action_clip_bounds(predicted_trajs_unnormalized.device, predicted_trajs_unnormalized.dtype)
            predicted_trajs_unnormalized = torch.clamp(predicted_trajs_unnormalized, min=amin, max=amax)

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
    
    def action_generation_ddim(self, imgs, agent_pos, batch_size, sample_steps=None, noise_intensity=0.0, generator=None):
        """DDIM sampling for faster action generation."""
        if sample_steps is None:
            num_steps = min(20, max(10, self.diff_steps // 10))
            sample_steps = np.linspace(1, self.diff_steps, num_steps, dtype=int).tolist()
            sample_steps = sorted(set(sample_steps))
        elif isinstance(sample_steps, int):
            num_steps = min(sample_steps, self.diff_steps)
            sample_steps = np.linspace(1, self.diff_steps, num_steps, dtype=int).tolist()
            sample_steps = sorted(set(sample_steps))

        imgs_features, pos_features = self._encoding_obs(imgs, agent_pos, training=False, generator=generator)

        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=imgs.device, generator=generator)
        predicted_trajs = self._clip_tensor(predicted_trajs, self.randn_clip_value)

        # DDIM reverse process
        for step_idx, i in enumerate(reversed(sample_steps)):
            t_features = torch.stack([self.time_encoding(i).to(device=imgs.device)
                                    for _ in range(batch_size)], dim=0)
            global_features_t = self.merge_multimodal_encoding(imgs_features, pos_features, t_features)

            traj_input = predicted_trajs.transpose(1, 2).contiguous()
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()

            # DDIM update equations
            # Arrays are 0-indexed for timesteps 1,2,...,diff_steps
            # So timestep i uses index i-1
            alpha_cumprod_t = self.cum_alphas[i-1]

            # For previous timestep: if i > 1, use i-2; if i == 1, use alpha_0 = 1.0
            if i > 1:
                alpha_cumprod_prev = self.cum_alphas[i-2]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=imgs.device)

            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            pred_x0 = (predicted_trajs - sqrt_one_minus_alpha_cumprod_t * predicted_noise_t) / sqrt_alpha_cumprod_t
            # Optional: clamp the denoised estimate
            pred_x0 = self._clip_tensor(pred_x0, self.denoised_clip_value)

            if torch.isnan(pred_x0).any() or step_idx < 5 or i < 10:
                print(f"  pred_x0 has NaN: {torch.isnan(pred_x0).any()}")
                print(f"  pred_x0 range: [{pred_x0.min().item():.3f}, {pred_x0.max().item():.3f}]")

            if i > 1:
                ratio_term = (1 - alpha_cumprod_t) * (alpha_cumprod_prev / alpha_cumprod_t)
                sqrt_term = 1 - alpha_cumprod_prev - ratio_term

                if sqrt_term < 0:
                    print(f"  WARNING: sqrt_term is negative: {sqrt_term.item():.8e}")

                noise_factor = noise_intensity * torch.sqrt(torch.clamp(sqrt_term, min=0.0))
                direction_xt = torch.sqrt(1 - alpha_cumprod_prev - noise_factor**2) * predicted_noise_t

                random_noise = torch.randn_like(predicted_trajs, generator=generator) if noise_intensity > 0 else 0
                if isinstance(random_noise, torch.Tensor):
                    random_noise = self._clip_tensor(random_noise, self.randn_clip_value)
                predicted_trajs = torch.sqrt(alpha_cumprod_prev) * pred_x0 + direction_xt + noise_factor * random_noise
                predicted_trajs = self._clip_tensor(predicted_trajs, self.denoised_clip_value)
            else:
                predicted_trajs = pred_x0


        action_dict = {"action": predicted_trajs}
        action_unnormalized = self.normalizer.unnormalize(action_dict)
        predicted_trajs_unnormalized = action_unnormalized["action"]
        if self.action_clip_value is not None:
            amin, amax = self._get_action_clip_bounds(predicted_trajs_unnormalized.device, predicted_trajs_unnormalized.dtype)
            predicted_trajs_unnormalized = torch.clamp(predicted_trajs_unnormalized, min=amin, max=amax)
        
        # Add constant dimension (0.27) to the end
        # predicted_trajs_unnormalized shape: (batch_size, horizon, action_dim)
        batch_size, horizon, action_dim = predicted_trajs_unnormalized.shape
        const_dim = torch.full((batch_size, horizon, 1), 0.27, 
                              device=predicted_trajs_unnormalized.device, 
                              dtype=predicted_trajs_unnormalized.dtype)
        predicted_trajs_with_const = torch.cat([predicted_trajs_unnormalized, const_dim], dim=-1)
        
        return predicted_trajs_with_const   