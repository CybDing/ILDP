from typing import Dict, Union
import torch
import torch.nn as nn
import numpy as np

from genesis_ILDP.model.diffusion.conditioned_unet import Unet
from genesis_ILDP.model.encoding import time_encoding, pos_encoding, merge_multimodal_encoding
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy
from genesis_ILDP.model.common.noise_scheduler import NoiseScheduler
from genesis_ILDP.model.vision.cnn_encoding import img_encoding_cnn
from genesis_ILDP.model.common.modules import RandomShiftsAug, RandomPosShifter
from genesis_ILDP.model.memory.memoryNET import MemoryNet

from diffusion_policy.model.common.normalizer import LinearNormalizer
from genesis_ILDP.model.vision.crop_randomizer import CropRandomizer
from genesis_ILDP.model.common.workspace_limiter import WorkspaceLimiter


class ActionDiffusionMemoryPolicy(BaseImagePolicy):

    def __init__(self,
                 shape_meta: dict,
                 cropper: CropRandomizer,
                 pos_encoding: pos_encoding,
                 time_encoding: time_encoding,
                 imgs_encoding_net: img_encoding_cnn,
                 conditioned_unet: Unet,
                 noise_scheduler: NoiseScheduler,
                 normalizer: LinearNormalizer,
                 memory_net: MemoryNet,

                 diff_steps: int = 100,
                 obs_steps: int = 2,
                 horizon: int = 16,
                 n_action_steps: int = 8,
                 n_obs_steps: int = 2,

                 noise_intensity: float = 1.0,
                 use_state_prob: float = 0.3,

                 use_ddpm = True,
                 ddim_steps: int = 100,

                 randn_clip_value: float | tuple | None = None,
                 denoised_clip_value: float | tuple | None = None,
                 noise_clip_percentile: float | None = None,

                 shift_augmenter: RandomShiftsAug | None = None,
                 pos_shifter: RandomPosShifter | None = None,
                 workspace_limiter: WorkspaceLimiter | None = None,

                 **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]

        obs_shape_meta = shape_meta['obs']
        assert 'image' in obs_shape_meta
        assert 'agent_pos' in obs_shape_meta

        self.diff_steps = diff_steps
        self.obs_steps = obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.ddim_steps = ddim_steps
        self.noise_intensity = noise_intensity
        self.use_ddpm = use_ddpm
        self.use_state_prob = use_state_prob

        self.randn_clip_value = randn_clip_value
        self.denoised_clip_value = denoised_clip_value
        self.noise_clip_percentile = noise_clip_percentile

        self.cropper = cropper
        self.enable_crop = cropper is not None

        self.shift_augmenter = shift_augmenter
        self.pos_shifter = pos_shifter
        self.workspace_limiter = workspace_limiter

        self.imgs_encoding_net = imgs_encoding_net
        self.conditioned_unet = conditioned_unet
        self.memory_net = memory_net

        self.time_encoding = time_encoding
        self.merge_multimodal_encoding = merge_multimodal_encoding
        self.pos_encoding = pos_encoding
        self.encode_agent_pos = pos_encoding is not None

        betas, cum_alphas = noise_scheduler.get_scheduler_values(self.diff_steps)
        self.register_buffer('betas', betas)
        self.register_buffer('cum_alphas', cum_alphas)

        self.traj_shape = (horizon, self.action_dim)

        self.normalizer = LinearNormalizer()
        if normalizer is not None:
            self.set_normalizer(normalizer)

        print(f"------- ActionDiffusionMemoryPolicy initialized: -------")
        print(f"- Action dim: {self.action_dim}")
        print(f"- Horizon: {horizon}")
        print(f"- Diffusion steps: {diff_steps}")
        print(f"- use_state_prob: {use_state_prob}")

    def _clip_tensor(self, x: torch.Tensor, clip_value):
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

    def _clip_noise_by_percentile(self, noise: torch.Tensor, percentile: float):
        if percentile is None or percentile <= 0 or percentile >= 100:
            return noise
        threshold = torch.quantile(torch.abs(noise), percentile / 100.0)
        return torch.clamp(noise, min=-threshold, max=threshold)

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
        if len(imgs.shape) == 5 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 1, 4, 2, 3)
        elif len(imgs.shape) != 5:
            print(f"[ActionDiffusionMemoryPolicy] WARNING: unexpected image tensor shape: {imgs.shape}")

        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]
        if imgs.shape[1] < self.n_obs_steps or agent_pos.shape[1] < self.n_obs_steps:
            raise ValueError("[ActionDiffusionMemoryPolicy] Error! obs steps smaller than n_obs_steps")

        return imgs, agent_pos

    def _encoding_obs(self, img, agent_pos, training=True, generator=None) -> tuple:
        batch_size = img.shape[0]
        img = img.reshape(-1, *img.shape[-3:])
        agent_pos = agent_pos.reshape(-1, agent_pos.shape[-1])

        img = self._apply_crop(img, training=training, generator=generator)

        img_encoded = self.imgs_encoding_net(img)
        if self.encode_agent_pos:
            agent_pos_encoded = self.pos_encoding(agent_pos)
        else:
            agent_pos_encoded = agent_pos

        img_feature = img_encoded.reshape(batch_size, -1)
        agent_pos_feature = agent_pos_encoded.reshape(batch_size, -1)

        return img_feature, agent_pos_feature

    def compute_initial_state(self, img_tensor: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        """Compute initial state from first observation (stateless)"""
        device = next(self.parameters()).device

        if len(img_tensor.shape) == 4:
            img_tensor = img_tensor.unsqueeze(1)
        if len(agent_pos.shape) == 2:
            agent_pos = agent_pos.unsqueeze(1)

        imgs, agent_pos = self._preprocess_obs(img_tensor, agent_pos)

        nobs = self.normalizer.normalize({
            'image': imgs.to(device),
            'agent_pos': agent_pos.to(device)
        })

        nimgs = nobs['image']
        nagent_pos = nobs['agent_pos']

        img_features, _ = self._encoding_obs(nimgs, nagent_pos, training=False)

        return img_features

    def compute_next_state(self, prev_state: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """Compute next state given previous state and action (stateless)"""
        device = next(self.parameters()).device

        naction = self.normalizer['action'].normalize(prev_action.to(device))

        next_state = self.memory_net.world_model(prev_state, naction)

        return next_state

    def predict_action_from_state(self, state: torch.Tensor, agent_pos: torch.Tensor, generator=None) -> torch.Tensor:
        """Predict action from state (used during rollout)"""
        device = next(self.parameters()).device

        if len(agent_pos.shape) == 2:
            agent_pos = agent_pos.unsqueeze(1)

        nagent_pos = self.normalizer['agent_pos'].normalize(agent_pos.to(device))

        if self.encode_agent_pos:
            agent_pos_encoded = self.pos_encoding(nagent_pos.reshape(-1, nagent_pos.shape[-1]))
            pos_features = agent_pos_encoded.reshape(state.shape[0], -1)
        else:
            pos_features = nagent_pos.reshape(state.shape[0], -1)

        batch_size = state.shape[0]

        if self.use_ddpm:
            action_pred = self._action_generation_ddpm_from_state(state, pos_features, batch_size, generator)
        else:
            raise NotImplementedError("DDIM not implemented for memory policy")

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return action

    def _action_generation_ddpm_from_state(self, state_features, pos_features, batch_size, generator=None) -> torch.Tensor:
        """DDPM sampling using state features instead of image features"""
        device = state_features.device

        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=device, generator=generator)
        predicted_trajs = self._clip_tensor(predicted_trajs, self.randn_clip_value)

        for step_idx, t in enumerate(reversed(range(1, self.diff_steps+1))):
            t_features = torch.stack([self.time_encoding(t).to(device=device) for _ in range(batch_size)], dim=0)
            global_features_t = self.merge_multimodal_encoding(state_features, pos_features, t_features)

            traj_input = predicted_trajs.transpose(1, 2).contiguous()
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()

            if t > 1:
                alpha_t = 1 - self.betas[t-1]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = self.betas[t-1] / torch.sqrt(1 - self.cum_alphas[t-1])
                mean = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)

                variance = self.betas[t-1] * (1 - self.cum_alphas[t-2]) / (1 - self.cum_alphas[t-1])

                noise = torch.randn(predicted_trajs.shape, device=device, dtype=predicted_trajs.dtype, generator=generator)
                noise = self._clip_tensor(noise, self.randn_clip_value)
                noise = self._clip_noise_by_percentile(noise, self.noise_clip_percentile)
                predicted_trajs = mean + torch.sqrt(variance) * noise
                predicted_trajs = self._clip_tensor(predicted_trajs, self.denoised_clip_value)
            else:
                alpha_t = 1 - self.betas[t-1]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = self.betas[t-1] / torch.sqrt(1 - self.cum_alphas[t-1])
                predicted_trajs = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)
                predicted_trajs = self._clip_tensor(predicted_trajs, self.denoised_clip_value)

            if torch.isnan(predicted_trajs).any():
                print(f"  NaN detected in predicted_trajs after DDPM step {step_idx}!")
                break

        action_dict = {"action": predicted_trajs}
        action_unnormalized = self.normalizer.unnormalize(action_dict)
        predicted_trajs_unnormalized = action_unnormalized["action"]

        if self.workspace_limiter is not None:
            predicted_trajs_unnormalized = self.workspace_limiter.clip(predicted_trajs_unnormalized)

        return predicted_trajs_unnormalized

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], generator=None) -> Dict[str, torch.Tensor]:
        """Standard prediction using image (not state) - used for validation"""
        imgs = obs_dict['image']
        agent_pos = obs_dict['agent_pos']

        imgs, agent_pos = self._preprocess_obs(imgs, agent_pos)
        device = next(self.parameters()).device

        nobs = self.normalizer.normalize({
            'image': imgs.to(device),
            'agent_pos': agent_pos.to(device)
        })

        nimgs = nobs['image']
        nagent_pos = nobs['agent_pos']
        batch_size = nimgs.shape[0]

        imgs_features, pos_features = self._encoding_obs(nimgs, nagent_pos, training=False, generator=generator)

        predicted_trajs = torch.randn(size=(batch_size, *self.traj_shape), device=device, generator=generator)
        predicted_trajs = self._clip_tensor(predicted_trajs, self.randn_clip_value)

        for step_idx, t in enumerate(reversed(range(1, self.diff_steps+1))):
            t_features = torch.stack([self.time_encoding(t).to(device=device) for _ in range(batch_size)], dim=0)
            global_features_t = self.merge_multimodal_encoding(imgs_features, pos_features, t_features)

            traj_input = predicted_trajs.transpose(1, 2).contiguous()
            predicted_noise = self.conditioned_unet(traj_input, global_features_t)
            predicted_noise_t = predicted_noise.transpose(1, 2).contiguous()

            if t > 1:
                alpha_t = 1 - self.betas[t-1]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = self.betas[t-1] / torch.sqrt(1 - self.cum_alphas[t-1])
                mean = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)
                variance = self.betas[t-1] * (1 - self.cum_alphas[t-2]) / (1 - self.cum_alphas[t-1])
                noise = torch.randn(predicted_trajs.shape, device=device, dtype=predicted_trajs.dtype, generator=generator)
                noise = self._clip_tensor(noise, self.randn_clip_value)
                noise = self._clip_noise_by_percentile(noise, self.noise_clip_percentile)
                predicted_trajs = mean + torch.sqrt(variance) * noise
                predicted_trajs = self._clip_tensor(predicted_trajs, self.denoised_clip_value)
            else:
                alpha_t = 1 - self.betas[t-1]
                coeff1 = 1.0 / torch.sqrt(alpha_t)
                coeff2 = self.betas[t-1] / torch.sqrt(1 - self.cum_alphas[t-1])
                predicted_trajs = coeff1 * (predicted_trajs - coeff2 * predicted_noise_t)
                predicted_trajs = self._clip_tensor(predicted_trajs, self.denoised_clip_value)

            if torch.isnan(predicted_trajs).any():
                break

        action_dict = {"action": predicted_trajs}
        action_unnormalized = self.normalizer.unnormalize(action_dict)
        predicted_trajs_unnormalized = action_unnormalized["action"]

        if self.workspace_limiter is not None:
            predicted_trajs_unnormalized = self.workspace_limiter.clip(predicted_trajs_unnormalized)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = predicted_trajs_unnormalized[:, start:end]

        return {
            'action': action,
            'action_pred': predicted_trajs_unnormalized
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(next(self.parameters()).device)

    def reset(self):
        """Reset policy state (stateless, does nothing)"""
        pass

    def compute_loss(self, batch):
        """
        Compute training loss with memory-based state evolution.

        Training strategy:
        1. Initialize state from first image
        2. For each timestep: evolve state via world model, fuse with image via gate
        3. With probability use_state_prob, use state as condition; otherwise use image
        4. Compute standard diffusion loss
        """
        imgs = batch['obs']['image']  # (T, C, H, W) or (B, T, C, H, W)
        agent_pos = batch['obs']['agent_pos']  # (T, 2) or (B, T, 2)
        actions = batch['action']  # (T, action_dim) or (B, T, action_dim)
        
        # Handle single sample (no batch dimension) - add batch dim if needed
        if len(imgs.shape) == 4:  # (T, C, H, W) - no batch dimension
            imgs = imgs.unsqueeze(0)  # (1, T, C, H, W)
            agent_pos = agent_pos.unsqueeze(0)  # (1, T, 2)
            actions = actions.unsqueeze(0)  # (1, T, action_dim)
        
        chunk_length = batch.get('chunk_length', imgs.shape[1])

        device = next(self.parameters()).device
        B, T = imgs.shape[:2]

        # Apply augmentations
        if self.shift_augmenter is not None:
            imgs_flat = imgs.reshape(-1, *imgs.shape[-3:])
            imgs_flat, shift_pixels = self.shift_augmenter(imgs_flat)
            imgs = imgs_flat.reshape(B, T, *imgs_flat.shape[-3:])

        if self.pos_shifter is not None:
            agent_pos, shift_value = self.pos_shifter(agent_pos)
            actions = actions.to(device).clone()
            shift_value_for_action = shift_value[:, 0:1, :].expand(-1, actions.shape[1], -1)
            actions = actions + shift_value_for_action
        else:
            actions = actions.to(device)

        # Normalize observations and actions
        nobs = self.normalizer.normalize({
            'image': imgs.to(device),
            'agent_pos': agent_pos.to(device)
        })
        nactions = self.normalizer['action'].normalize(actions)

        nimgs = nobs['image']
        nagent_pos = nobs['agent_pos']

        # Encode all images upfront
        all_img_features = []
        all_pos_features = []
        for t in range(T):
            img_feat, pos_feat = self._encoding_obs(
                nimgs[:, t:t+1], nagent_pos[:, t:t+1], training=True
            )
            all_img_features.append(img_feat)
            all_pos_features.append(pos_feat)

        # Initialize state from first image
        state = all_img_features[0].clone()

        total_loss = 0.0
        valid_steps = 0

        for t in range(T):
            # World model prediction (if t > 0)
            if t > 0:
                # Prepare previous action in correct format
                prev_action = nactions[:, t-1:t]  # (B, 1, action_dim)

                # Expand to action_horizon if needed for world model
                if prev_action.shape[1] < self.horizon:
                    prev_action_expanded = prev_action.repeat(1, self.horizon, 1)
                else:
                    prev_action_expanded = prev_action[:, :self.horizon, :]

                state_hat = self.memory_net.world_model(state, prev_action_expanded)
            else:
                state_hat = state

            # Gate fusion
            img_vec = all_img_features[t]
            state = self.memory_net(img_vec, prev_state=state_hat, prev_action=None)

            # Randomly choose to use state or image features
            use_state_mask = torch.rand(B, device=device) < self.use_state_prob
            cond_feature = torch.where(
                use_state_mask.unsqueeze(-1),
                state,
                img_vec
            )

            # Compute diffusion loss for this timestep
            if t + self.horizon <= T:
                action_chunk = nactions[:, t:t+self.horizon, :]

                random_t = torch.randint(1, self.diff_steps+1, (B,), device=device)
                random_noise = torch.randn(action_chunk.shape, device=device, dtype=action_chunk.dtype)
                random_noise = self._clip_tensor(random_noise, self.randn_clip_value)

                sqrt_alpha_cumprod = torch.sqrt(self.cum_alphas[random_t - 1]).reshape(B, 1, 1)
                sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.cum_alphas[random_t - 1]).reshape(B, 1, 1)

                noisy_action = sqrt_alpha_cumprod * action_chunk + sqrt_one_minus_alpha_cumprod * random_noise

                t_encoding = torch.stack([self.time_encoding(t_val.item()) for t_val in random_t], dim=0).to(device)
                
                # Gather n_obs_steps frames of features for conditioning (to match UNet expectation)
                # Use frames [max(0, t-n_obs_steps+1) : t+1] for observation context
                obs_start = max(0, t - self.n_obs_steps + 1)
                obs_end = t + 1
                
                # Concatenate observation features from multiple frames
                cond_features_list = []
                pos_features_list = []
                for obs_t in range(obs_start, obs_end):
                    cond_features_list.append(all_img_features[obs_t] if not use_state_mask.all() else state)
                    pos_features_list.append(all_pos_features[obs_t])
                
                # Pad if we don't have enough frames (at the beginning)
                while len(cond_features_list) < self.n_obs_steps:
                    cond_features_list.insert(0, cond_features_list[0])
                    pos_features_list.insert(0, pos_features_list[0])
                
                # Concatenate to match expected dimension: (B, 512 * n_obs_steps)
                cond_feature_cat = torch.cat(cond_features_list, dim=-1)
                pos_feature_cat = torch.cat(pos_features_list, dim=-1)
                
                global_cond = self.merge_multimodal_encoding(cond_feature_cat, pos_feature_cat, t_encoding)

                noisy_action_input = noisy_action.transpose(1, 2).contiguous()
                predicted_noise = self.conditioned_unet(noisy_action_input, global_cond)
                predicted_noise = predicted_noise.transpose(1, 2).contiguous()

                loss_t = torch.nn.functional.mse_loss(predicted_noise, random_noise)
                total_loss += loss_t
                valid_steps += 1

        if valid_steps > 0:
            return total_loss / valid_steps
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
