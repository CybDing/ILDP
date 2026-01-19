import torch
import numpy as np
import torch.nn as nn
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

class Critic(nn.Module):

    def __init__(self,
                 is_doubleQ = False,
                 shape_meta = None,
                 layers_Sdim = [2, 4, 6],  # Normalized layer dims: actual = [d_input*2, d_input*4, d_input*6]
                 activation = 'ReLU',
                 use_pretrained_diff_encoder = True,
                 n_obs = 2,
                 img_feature_dim = 1024,
                 agent_pos_feature_dim = 512,
                 enable_crop = True,
                 crop_shape = (3, 76, 76),  # (C, H, W)
                 ):
        super().__init__()

        self.is_doubleQ = is_doubleQ
        if shape_meta is not None and 'obs' in shape_meta:
            self.input_args = list(shape_meta['obs'].keys())
        else:
            self.input_args = []

        self.layers_Sdim = layers_Sdim

        # Get activation function
        if hasattr(nn, activation):
            self.activation = getattr(nn, activation)
        elif hasattr(nn.functional, activation.lower()):
            self.activation_fn = getattr(nn.functional, activation.lower())
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.use_pretrained_diff_encoder = use_pretrained_diff_encoder

        self.img_encoder = None
        self.agent_pos_encoder = None
        self.train_encoder = False  # Default: freeze encoder for fine-tuning
        self.n_obs = n_obs
        self.shape_meta = shape_meta

        self.img_feature_dim = img_feature_dim
        self.agent_pos_feature_dim = agent_pos_feature_dim

        self.Qnet = None
        self.normalizer = LinearNormalizer()

        # Setup center cropping
        self.enable_crop = enable_crop
        self.crop_shape = crop_shape
        self.cropper = None

        if self.enable_crop and shape_meta is not None and 'obs' in shape_meta and 'image' in shape_meta['obs']:
            raw_img_shape = shape_meta['obs']['image']['shape']
            input_height, input_width = raw_img_shape[1], raw_img_shape[2]
            crop_height, crop_width = crop_shape[1], crop_shape[2]

            if crop_height > input_height or crop_width > input_width:
                print(f"Warning: Crop dims ({crop_height}, {crop_width}) > input dims ({input_height}, {input_width}), disabling crop")
                self.enable_crop = False
            else:
                # pos_enc=False: don't add spatial position encoding channels
                self.cropper = CropRandomizer(
                    input_shape=raw_img_shape,
                    crop_height=crop_height,
                    crop_width=crop_width,
                    num_crops=1,
                    pos_enc=False
                )
                print(f"Center cropping enabled for Critic: {raw_img_shape} -> ({crop_height}, {crop_width})")
        else:
            self.enable_crop = False
            print(f"Center cropping disabled for Critic")

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Directly reference policy's normalizer (shared parameters)."""
        self.normalizer = normalizer
        print("Normalizer set for Critic network")

    def adapt_obs_encoder(self, img_encoder, agent_pos_encoder=None, train_encoder=False):
        """Adapt encoders from policy. Default: freeze encoders."""
        self.img_encoder = img_encoder
        self.agent_pos_encoder = agent_pos_encoder
        self.train_encoder = train_encoder

        if not train_encoder:
            if self.img_encoder is not None:
                for param in self.img_encoder.parameters():
                    param.requires_grad = False
            if self.agent_pos_encoder is not None:
                for param in self.agent_pos_encoder.parameters():
                    param.requires_grad = False

        self._create_QNet()

    def _create_QNet(self):
        """Create Q-network with normalized layer dims."""
        if self.n_obs is None:
            raise ValueError("[Qnet] n_obs is not given")

        # Calculate input dimension
        d_input = 0
        for key in self.input_args:
            if key == 'image':
                d_input += self.img_feature_dim * self.n_obs
            if key == 'agent_pos':
                if self.agent_pos_encoder is not None:
                    d_input += self.agent_pos_feature_dim * self.n_obs
                else:
                    if self.shape_meta and 'obs' in self.shape_meta and 'agent_pos' in self.shape_meta['obs']:
                        d_input += self.shape_meta['obs']['agent_pos']['shape'][-1] * self.n_obs
                    else:
                        d_input += 2 * self.n_obs

        # Build MLP: layers_Sdim=[2,4,6] -> actual dims=[d_input, 2*d_input, 4*d_input, 6*d_input, 1]
        layers = []
        layer_dims = [d_input] + [d_input * s for s in self.layers_Sdim] + [1]

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                if self.activation is not None:
                    layers.append(self.activation())

        self.Qnet = nn.Sequential(*layers)
        print(f"Q-network: input_dim={d_input}, structure={self.layers_Sdim}, actual_dims={layer_dims}")

    def _encode_obs(self, obs_dict) -> torch.Tensor:
        """Encode observations to features."""
        obs_keys = list(obs_dict.keys())
        for arg in self.input_args:
            if arg not in obs_keys:
                raise ValueError(f"[Critic] Missing '{arg}' in obs_dict")

        def __encode_obs_dict():
            features = []
            first_key = obs_keys[0]
            batch_size = obs_dict[first_key].shape[0]

            for key in self.input_args:
                if key == 'image':
                    raw_imgs = obs_dict[key]
                    assert len(raw_imgs.shape) == 5, f"Expected 5D image, got {raw_imgs.shape}"
                    assert raw_imgs.shape[0] == batch_size

                    # Ensure channels-first: (B, To, C, H, W)
                    if raw_imgs.shape[-1] == 3 or raw_imgs.shape[-1] == 1:
                        raw_imgs = raw_imgs.permute(0, 1, 4, 2, 3)

                    raw_imgs_batched = raw_imgs.reshape(-1, *raw_imgs.shape[2:])

                    # Apply center crop
                    if self.enable_crop and self.cropper is not None:
                        if hasattr(self.cropper, 'to'):
                            self.cropper = self.cropper.to(raw_imgs_batched.device)
                        self.cropper.eval()  # Use center crop
                        raw_imgs_batched = self.cropper(raw_imgs_batched)

                    img_features = self.img_encoder(raw_imgs_batched)
                    img_features = img_features.reshape(batch_size, -1)
                    features.append(img_features)

                elif key == 'agent_pos':
                    agent_pos = obs_dict[key]
                    assert len(agent_pos.shape) == 3, f"Expected 3D agent_pos, got {agent_pos.shape}"
                    assert agent_pos.shape[0] == batch_size

                    agent_pos_batched = agent_pos.reshape(-1, agent_pos.shape[2])

                    if self.agent_pos_encoder is not None:
                        agent_pos_features = self.agent_pos_encoder(agent_pos_batched)
                    else:
                        agent_pos_features = agent_pos_batched

                    agent_pos_features = agent_pos_features.reshape(batch_size, -1)
                    features.append(agent_pos_features)

            features = torch.cat(features, dim=-1)
            return features

        if not self.train_encoder:
            with torch.no_grad():
                return __encode_obs_dict()
        else:
            return __encode_obs_dict()

    def forward(self, obs_dict):
        """Forward: normalize -> encode -> Q-value."""
        if self.Qnet is None:
            raise ValueError("[Critic] Q-network not created. Call adapt_obs_encoder() first")

        # Normalize observations
        device = next(self.parameters()).device
        nobs = self.normalizer.normalize({
            'image': obs_dict['image'].to(device),
            'agent_pos': obs_dict['agent_pos'].to(device),
        })

        # Encode and compute Q-value
        obs_features = self._encode_obs(nobs)
        value = self.Qnet(obs_features)

        return value