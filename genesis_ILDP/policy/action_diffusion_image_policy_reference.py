from typing import Dict
import torch
import torch.nn as nn
import torchvision

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from genesis_ILDP.policy.base_image_policy import BaseImagePolicy


class ActionDiffusionImagePolicyReference(BaseImagePolicy):

    def __init__(self,
                 shape_meta: dict,
                 horizon: int = 16,
                 n_action_steps: int = 8,
                 n_obs_steps: int = 2,
                 num_inference_steps: int = 100,
                 # Scheduler settings
                 diff_steps: int = 100,
                 beta_schedule: str = 'squaredcos_cap_v2',
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 prediction_type: str = 'epsilon',
                 clip_sample: bool = True,
                 variance_type: str = 'fixed_small',  # Options: 'fixed_small', 'fixed_small_log'
                 # Vision encoder settings
                 vision_backbone: str = 'resnet18',
                 vision_pretrained: bool = False,  # Whether to use pretrained weights
                 use_group_norm: bool = True,
                 share_rgb_model: bool = False,  # Share rgb model across observations
                 imagenet_norm: bool = False,  # Use ImageNet normalization
                 resize_shape: tuple = None,  # (height, width) for resizing
                 crop_shape: tuple = None,  # (height, width) for cropping
                 random_crop: bool = True,  # Use random crop vs center crop
                 # UNet settings
                 diffusion_step_embed_dim: int = 256,
                 down_dims: tuple = (256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 cond_predict_scale: bool = True,  # Conditional prediction scaling
                 obs_as_global_cond: bool = True,
                 # Additional parameters (passed to scheduler.step during inference)
                 **kwargs):
        super().__init__()

        self.shape_meta = shape_meta
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.obs_as_global_cond = obs_as_global_cond

        # Get action dimension
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]

        # === Create vision encoder (MultiImageObsEncoder) ===
        # This matches the reference implementation's vision encoder setup

        if vision_backbone == 'resnet18':
            # Use ResNet18 with optional pretrained weights
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if vision_pretrained else None
                rgb_model = torchvision.models.resnet18(weights=weights)
            except (ImportError, AttributeError):
                # Fallback for older torchvision versions
                rgb_model = torchvision.models.resnet18(pretrained=vision_pretrained)

            # Modify first conv layer if needed for different input channels
            obs_shape = shape_meta['obs']['image']['shape']
            in_channels = obs_shape[0]
            if in_channels != 3:
                rgb_model.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            # Remove final FC layer, keep avgpool - outputs (B, 512)
            rgb_model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported vision backbone: {vision_backbone}")

        # Initialize MultiImageObsEncoder with the rgb_model
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            resize_shape=resize_shape,  # Optional resizing
            crop_shape=crop_shape,  # Optional cropping
            random_crop=random_crop,  # Random vs center crop
            use_group_norm=use_group_norm,  # Replace BatchNorm with GroupNorm
            share_rgb_model=share_rgb_model,  # Share rgb model across observations
            imagenet_norm=imagenet_norm  # Use ImageNet normalization
        )

        # === Create noise scheduler (DDPMScheduler) ===
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diff_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            variance_type=variance_type,  
        )

        # === Create reference DiffusionUnetImagePolicy ===
        self.reference_policy = DiffusionUnetImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=self.noise_scheduler,
            obs_encoder=self.obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            **kwargs  
        )

        # Create normalizer (will be set by training workspace)
        self.normalizer = LinearNormalizer()

        # Store imagenet_norm flag for later use in set_normalizer
        self.imagenet_norm = imagenet_norm

    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Set normalizer from training workspace.

        IMPORTANT: When imagenet_norm=True, skip image normalization to avoid
        double normalization (ImageNet expects [0,1], not [-1,1]).
        """
        if self.imagenet_norm:
            # Remove image normalizer - keep images in [0,1] for ImageNet norm
            modified_normalizer = LinearNormalizer()
            modified_normalizer.load_state_dict(normalizer.state_dict())

            # Remove 'image' key if it exists
            if 'image' in modified_normalizer.params_dict:
                del modified_normalizer.params_dict['image']

            self.normalizer.load_state_dict(modified_normalizer.state_dict())
            self.reference_policy.set_normalizer(modified_normalizer)
        else:
            # Use normalizer as-is (with image range normalization)
            self.normalizer.load_state_dict(normalizer.state_dict())
            self.reference_policy.set_normalizer(normalizer)

    def compute_loss(self, batch):
        """
        Compute loss using reference policy.

        Args:
            batch: Dictionary with 'obs' and 'action' keys
                   obs['image']: (B, T, C, H, W) in [0, 1] range (raw from dataset)
                   obs['agent_pos']: (B, T, 2)
                   action: (B, T, 2) unnormalized

        Returns:
            loss: Scalar tensor
        """
        return self.reference_policy.compute_loss(batch)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], generator=None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Predict action using reference policy.

        Args:
            obs_dict: Dictionary with 'image' and 'agent_pos' keys
                      image: (B, T, C, H, W), the img from the real env is not normalized, so we normalize the img inside the predict_action function call 
                      agent_pos: (B, T, 2)
            generator: Optional torch.Generator for reproducible sampling

        Returns:
            Dictionary with 'action' key containing predicted actions
            action: (B, n_action_steps, 3) - includes z-dimension (0.27)
        """
        device = self.device

        imgs = obs_dict['image']
        agent_pos = obs_dict['agent_pos']

        if len(imgs.shape) == 5 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 1, 4, 2, 3)

        if imgs.shape[1] > self.n_obs_steps:
            imgs = imgs[:, :self.n_obs_steps]
        if agent_pos.shape[1] > self.n_obs_steps:
            agent_pos = agent_pos[:, :self.n_obs_steps]

        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        
        obs_dict = {
            'image': imgs,
            'agent_pos': agent_pos
        }
        # nobs_dict = self.normalizer.normalize(obs_dict) # normalize the obs before passing to the policy for predicting the action 
        result = self.reference_policy.predict_action(obs_dict, generator=generator) # the obs will be normalized inside the policy itself

        # Extract predicted action (B, n_action_steps, 2)
        action_pred = result['action']

        # Add constant z-dimension (0.27) for robot arm height maintenance
        batch_size = action_pred.shape[0]
        n_steps = action_pred.shape[1]
        const_z_dim = torch.full((batch_size, n_steps, 1), 0.27,
                                 device=action_pred.device,
                                 dtype=action_pred.dtype)
        action_pred_with_z = torch.cat([action_pred, const_z_dim], dim=-1)

        return {
            'action': action_pred_with_z,
            'action_pred': result.get('action_pred', action_pred_with_z)
        }

    def reset(self):
        """Reset policy state"""
        self.reference_policy.reset()
