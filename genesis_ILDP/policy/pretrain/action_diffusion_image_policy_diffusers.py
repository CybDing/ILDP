from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import tqdm
from typing import Union

from diffusers.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler

from genesis_ILDP.model.conditioned_unet import Unet
from genesis_ILDP.model.encoding import global_img_encoding, time_encoding, pos_encoding, merge_multimodal_encoding
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer


class ActionDiffusionImagePolicyDiffusers(BaseImagePolicy):
    """
    Action diffusion policy using diffusers library schedulers.
    """

    def __init__(self,
                 shape_meta: dict,
                 normalizer: LinearNormalizer = None,
                 vision_backbone: str = 'custom',
                 vision_pretrained: bool = False,
                 diff_steps: int = 100,
                 scheduler_type: str = 'DDPM',  # 'DDPM', 'DDIM', 'PNDM', 'DPMSolver'
                 beta_schedule: str = 'squaredcos_cap_v2',  # 'linear', 'scaled_linear', 'squaredcos_cap_v2'
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 prediction_type: str = 'epsilon',  # 'epsilon' or 'sample'
                 clip_sample: bool = True,
                 obs_steps: int = 2,
                 horizon: int = 16,
                 n_action_steps: int = 8,
                 n_obs_steps: int = 2,
                 encode_agent_pos: bool = False,
                 num_inference_steps: int = None,  # For DDIM/fast sampling (default: use diff_steps)
                 crop_shape=(3, 76, 76),
                 enable_crop=True,
                 img_encoding_dim=512,
                 time_encoding_dim=128,
                 pos_encoding_dim=None,
                 use_spatial_softmax: bool = False,
                 spatial_softmax_temp: float = 1.0,
                 vision_encoder_dropout: float = 0.0,
                 pos_encoder_dropout: float = 0.0,
                 unet_dropout: float = 0.0,
                 time_encoder_dropout: float = 0.0,
                 **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]

        obs_shape_meta = shape_meta['obs']
        raw_img_shape = shape_meta['obs']['image']['shape']
        assert 'image' in obs_shape_meta
        assert 'agent_pos' in obs_shape_meta

        self.diff_steps = diff_steps
        self.scheduler_type = scheduler_type
        self.obs_steps = obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.encode_agent_pos = encode_agent_pos
        self.num_inference_steps = num_inference_steps if num_inference_steps is not None else diff_steps
        self.enable_crop = enable_crop
        self.crop_shape = crop_shape
        self.time_encoding_dim = time_encoding_dim
        self.pos_encoding_dim = pos_encoding_dim

        # Setup cropping
        if self.enable_crop:
            input_height, input_width = raw_img_shape[1], raw_img_shape[2]
            crop_height, crop_width = crop_shape[1], crop_shape[2]

            if crop_height > input_height or crop_width > input_width:
                raise ValueError(f"Crop dims ({crop_height}, {crop_width}) > input dims ({input_height}, {input_width})")

            self.cropper = CropRandomizer(input_shape=raw_img_shape, crop_height=crop_height,
                                          crop_width=crop_width, num_crops=1, pos_enc=False)
            print(f"Random cropping enabled: {raw_img_shape} -> crop ({crop_height}, {crop_width})")
        else:
            self.cropper = None
            print(f"Random cropping disabled, using full image: {raw_img_shape}")

        # Calculate global features dimension
        if encode_agent_pos:
            if pos_encoding_dim is None:
                self.pos_encoding_dim = 128
            dim_global_features = img_encoding_dim * 2 + time_encoding_dim + pos_encoding_dim * 2
        else:
            agent_pos_dim = 2 * n_obs_steps
            self.pos_encoding_dim = 2
            dim_global_features = img_encoding_dim * 2 + time_encoding_dim + agent_pos_dim

        # Initialize vision encoder
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

        # Get actual output dimension
        actual_img_encoding_dim = self.imgs_encoding_net.get_feature_dim()
        self.img_encoding_dim = actual_img_encoding_dim
        print(f"Vision encoder output dimension: {actual_img_encoding_dim} (spatial_softmax={use_spatial_softmax})")

        # Recalculate global features dimension
        if encode_agent_pos:
            dim_global_features = actual_img_encoding_dim * 2 + time_encoding_dim + (pos_encoding_dim if pos_encoding_dim else 128) * 2
        else:
            dim_global_features = actual_img_encoding_dim * 2 + time_encoding_dim + agent_pos_dim

        # Initialize UNet (same as original)
        self.conditioned_unet = Unet(dim_global_features, input_dim=self.action_dim, dropout=unet_dropout)

        # Initialize time encoding module
        self.time_encoding_net = time_encoding(t_emb_dim=time_encoding_dim, dropout=time_encoder_dropout)
        self.merge_multimodal_encoding = merge_multimodal_encoding

        if encode_agent_pos:
            self.agent_pos_encoding = pos_encoding(encoded_dim=self.pos_encoding_dim, dropout=pos_encoder_dropout)
        else:
            self.agent_pos_encoding = None

        # Initialize diffusers scheduler
        self._init_scheduler(scheduler_type, beta_schedule, beta_start, beta_end, prediction_type, clip_sample)

        self.traj_shape = (horizon, self.action_dim)

        # Initialize normalizer
        self.normalizer = LinearNormalizer()
        if normalizer is not None:
            self.set_normalizer(normalizer)

        print(f"ActionDiffusionImagePolicyDiffusers initialized:")
        print(f"- Action dim: {self.action_dim}")
        print(f"- Horizon: {horizon}")
        print(f"- Diffusion steps: {diff_steps}")
        print(f"- Scheduler type: {scheduler_type}")
        print(f"- Beta schedule: {beta_schedule}")
        print(f"- Prediction type: {prediction_type}")
        print(f"- Num inference steps: {self.num_inference_steps}")

    def _init_scheduler(self, scheduler_type, beta_schedule, beta_start, beta_end, prediction_type, clip_sample):
        """Initialize the diffusers scheduler."""
        scheduler_kwargs = {
            'num_train_timesteps': self.diff_steps,
            'beta_start': beta_start,
            'beta_end': beta_end,
            'beta_schedule': beta_schedule,
            'prediction_type': prediction_type,
        }

        if scheduler_type == 'DDPM':
            scheduler_kwargs['clip_sample'] = clip_sample
            scheduler_kwargs['variance_type'] = 'fixed_small'
            self.noise_scheduler = DDPMScheduler(**scheduler_kwargs)
        elif scheduler_type == 'DDIM':
            scheduler_kwargs['clip_sample'] = clip_sample
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        elif scheduler_type == 'PNDM':
            self.noise_scheduler = PNDMScheduler(**scheduler_kwargs)
        elif scheduler_type == 'DPMSolver':
            self.noise_scheduler = DPMSolverMultistepScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported: DDPM, DDIM, PNDM, DPMSolver")

        print(f"Initialized {scheduler_type} scheduler with {beta_schedule} beta schedule")

    def _apply_crop(self, images: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply cropping to images if enabled."""
        if not self.enable_crop or self.cropper is None:
            return images

        if hasattr(self.cropper, 'to'):
            self.cropper = self.cropper.to(images.device)

        if training:
            self.cropper.train()
        else:
            self.cropper.eval()

        return self.cropper(images)

    def compute_loss(self, batch):
        """
        Compute diffusion loss using diffusers scheduler.

        This replaces manual noise scheduling with scheduler.add_noise().
        """
        # Normalize inputs
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Get observations
        imgs = nobs['image']  # (B, horizon, C, H, W)
        agent_pos = nobs['agent_pos']  # (B, horizon, 2)

        # Trim observations to n_obs_steps if needed (dataset may provide full horizon)
        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]

        # Stack images from observation steps
        imgs_stack = imgs.reshape(-1, *imgs.shape[2:])  # (B*n_obs_steps, C, H, W)

        # Process agent positions
        if self.encode_agent_pos:
            agent_pos_flat = agent_pos.reshape(-1, *agent_pos.shape[2:])  # (B*n_obs_steps, 2)
            pos_encoding_res = self.agent_pos_encoding(agent_pos_flat)
            pos_encoding_res = pos_encoding_res.reshape(batch_size, self.n_obs_steps, -1)
        else:
            pos_encoding_res = agent_pos

        pos_features = pos_encoding_res.reshape(batch_size, -1)

        # Get action
        action = nactions[:, :horizon, :self.action_dim]

        # Sample random timesteps for each sample in batch
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=action.device
        ).long()

        # Sample noise
        noise = torch.randn(action.shape, device=action.device)

        # Add noise to actions using diffusers scheduler
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        # Encode time using timesteps
        t_encoding = torch.stack([self.time_encoding_net(t.item()) for t in timesteps], dim=0)

        # Apply cropping and encode images
        imgs_cropped = self._apply_crop(imgs_stack, training=True)
        imgs_encoding = self.imgs_encoding_net(imgs_cropped).reshape(batch_size, -1)

        global_features = self.merge_multimodal_encoding(imgs_encoding, pos_features, t_encoding)

        noisy_action_input = noisy_action.transpose(1, 2).contiguous()
        predicted_noise = self.conditioned_unet(noisy_action_input, global_features)
        predicted_noise = predicted_noise.transpose(1, 2).contiguous()

        loss = nn.functional.mse_loss(predicted_noise, noise)

        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], recording_diffusion=False) -> Dict[str, torch.Tensor]:
        """
        Predict actions using diffusers scheduler for denoising.

        This replaces manual DDPM/DDIM sampling with scheduler.step().
        """
        # Get raw observations (before normalization)
        imgs = obs_dict['image']
        agent_pos = obs_dict['agent_pos']

        # Convert image dimensions if needed: [B, T, H, W, C] -> [B, T, C, H, W]
        if len(imgs.shape) == 5 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]

        # Trim to expected observation steps BEFORE normalization
        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]

        # Move to device and normalize
        device = next(self.parameters()).device
        nobs = self.normalizer.normalize({
            'image': imgs.to(device),
            'agent_pos': agent_pos.to(device),
        })

        # Get normalized observations
        imgs = nobs['image']  # (B, n_obs_steps, C, H, W)
        agent_pos = nobs['agent_pos']  # (B, n_obs_steps, 2)

        batch_size = imgs.shape[0]
        imgs_stack = imgs.reshape(-1, *imgs.shape[2:])  # (B*n_obs_steps, C, H, W)

        # Encode images
        imgs_cropped = self._apply_crop(imgs_stack, training=False)
        imgs_features = self.imgs_encoding_net(imgs_cropped).reshape(batch_size, -1)

        # Process agent positions
        if self.encode_agent_pos:
            agent_pos_flat = agent_pos.reshape(-1, *agent_pos.shape[2:])  # (B*n_obs_steps, 2)
            pos_encoding_res = self.agent_pos_encoding(agent_pos_flat)
            pos_encoding_res = pos_encoding_res.reshape(batch_size, self.n_obs_steps, -1)
        else:
            pos_encoding_res = agent_pos

        pos_features = pos_encoding_res.reshape(batch_size, -1)

        predicted_trajs = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            device=device, dtype=imgs.dtype
        )

        # Set timesteps for inference
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Denoising loop using scheduler
        for t in tqdm.tqdm(self.noise_scheduler.timesteps, desc="Diffusion sampling", leave=False):
            # Encode timestep
            t_features = torch.stack([self.time_encoding_net(t.item()).to(device=imgs.device)
                                     for _ in range(batch_size)], dim=0)

            # Merge features
            global_features_t = self.merge_multimodal_encoding(imgs_features, pos_features, t_features)

            # Predict noise
            traj_input = predicted_trajs.transpose(1, 2).contiguous()
            model_output = self.conditioned_unet(traj_input, global_features_t)
            model_output = model_output.transpose(1, 2).contiguous()

            # Denoise using scheduler
            predicted_trajs = self.noise_scheduler.step(
                model_output, t, predicted_trajs
            ).prev_sample

        # Denormalize actions
        naction_pred = predicted_trajs
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Add constant z-dimension (0.27) for robot arm height maintenance
        # action_pred shape: (batch_size, horizon, action_dim=2) -> (batch_size, horizon, 3)
        batch_size, horizon, action_dim = action_pred.shape
        const_z_dim = torch.full((batch_size, horizon, 1), 0.27,
                                 device=action_pred.device,
                                 dtype=action_pred.dtype)
        action_pred_with_z = torch.cat([action_pred, const_z_dim], dim=-1)

        # Extract action steps to execute
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred_with_z[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred_with_z
        }

        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
