# This file serve as the basic encoding for the global images being stacked; and provide the API functions that is used for time encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from genesis_ILDP.utils.cuda import to_torch
from torchvision import models

def torch_stack_images(imgs:list):
    # images_stack (B * 2 * W * H * C)

    imgs_counts = len(imgs)
    imgs = [np.concatenate(imgs[seq][0], imgs[seq][1], axis = 1) for seq in range(imgs_counts)]
    imgs = np.moveaxis(imgs, -1, 1)  # swap the channel with the Width
    return np.stack(
        imgs, axis = 0
    )


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax pooling as described in 'Deep Spatial Autoencoders for Visuomotor Learning'.
    Converts feature maps to coordinate expectation (x, y) for each channel.

    Args:
        input_shape: Tuple of (C, H, W) where C is channels, H is height, W is width
        temperature: Temperature parameter for softmax (default: 1.0, lower = sharper, higher = smoother)

    Output:
        Tensor of shape (B, 2*C) containing (x, y) coordinates for each of C channels
    """
    def __init__(self, input_shape, temperature=1.0):
        super().__init__()
        assert len(input_shape) == 3, f"Expected input_shape to be (C, H, W), got {input_shape}"
        self.channel, self.height, self.width = input_shape
        self.temperature = temperature

        # Create normalized coordinate grids in range [-1, 1]
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1, 1, self.width),
            np.linspace(-1, 1, self.height)
        )
        # Flatten spatial dimensions: (H, W) -> (H*W,)
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()

        # Register as buffers (moved to GPU automatically with model)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        """
        Args:
            feature: Input tensor of shape (B, C, H, W)

        Returns:
            coords: Tensor of shape (B, 2*C) with (x, y) coordinates for each channel
        """
        assert feature.shape[1] == self.channel, \
            f"Expected {self.channel} channels, got {feature.shape[1]}"
        assert feature.shape[2] == self.height and feature.shape[3] == self.width, \
            f"Expected spatial dims ({self.height}, {self.width}), got ({feature.shape[2]}, {feature.shape[3]})"

        batch_size = feature.shape[0]

        # Reshape: (B, C, H, W) -> (B, C, H*W)
        feature = feature.reshape(batch_size, self.channel, self.height * self.width)

        # Apply temperature and softmax over spatial dimension
        # (B, C, H*W) -> softmax along dim=2
        softmax_attention = F.softmax(feature / self.temperature, dim=2)

        # Compute expected x, y coordinates weighted by softmax attention
        # (B, C, H*W) * (H*W,) -> (B, C)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=2)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=2)

        # Concatenate: (B, C) + (B, C) -> (B, 2*C)
        coords = torch.cat([expected_x, expected_y], dim=1)

        return coords

    def __repr__(self):
        return f"SpatialSoftmax(channels={self.channel}, spatial_dims=({self.height}, {self.width}), temperature={self.temperature})"

class img_encoding_cnn(nn.Module):
    def __init__(self, in_channels, encoded_dim=512, backbone='resnet18', pretrained=False,
                 use_spatial_softmax=False, spatial_softmax_temp=1.0, input_image_shape=None,
                 dropout=0.0):
        """
        Image encoder with configurable pooling strategy.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            encoded_dim: Output feature dimension (default: 512)
            backbone: Backbone architecture ('resnet18' or 'custom')
            pretrained: Whether to use pretrained weights
            use_spatial_softmax: If True, use spatial softmax pooling instead of global avg pooling
            spatial_softmax_temp: Temperature for spatial softmax (default: 1.0)
            input_image_shape: Tuple (H, W) of input image size for spatial softmax initialization
            dropout: Dropout rate for regularization (default: 0.0, disabled)
        """
        super().__init__()
        self.encoded_dim = encoded_dim
        self.backbone_type = backbone
        self.use_spatial_softmax = use_spatial_softmax
        self.spatial_softmax_temp = spatial_softmax_temp
        self.dropout_rate = dropout

        if backbone == 'resnet18':
            # Use weights parameter for torchvision >= 0.13 (compatible with newer PyTorch)
            try:
                from torchvision.models import ResNet18_Weights
                # Use DEFAULT weights which automatically picks the best available version
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                resnet = models.resnet18(weights=weights)
            except (ImportError, AttributeError):
                # Fallback for older torchvision versions
                resnet = models.resnet18(pretrained=pretrained)

            # Adjust first conv layer if input channels != 3
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Replace BatchNorm with GroupNorm for EMA compatibility
            from genesis_ILDP.utils.pytorch_util import replace_submodules
            resnet = replace_submodules(
                root_module=resnet,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features)
            )

            if use_spatial_softmax:
                # Remove avgpool and fc layer for spatial softmax
                # Keep conv layers that output (B, 512, H, W)
                self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

                # Calculate spatial dimensions by doing a forward pass
                # ResNet18 with input (B, C, H, W) outputs (B, 512, H', W') after layer4
                # We need to determine exact H', W' for spatial softmax initialization
                with torch.no_grad():
                    # Use provided input shape or default to 86x86
                    if input_image_shape is not None:
                        img_h, img_w = input_image_shape
                    else:
                        img_h, img_w = 86, 86  # Default crop size
                        print(f"Warning: input_image_shape not provided, using default {img_h}x{img_w}")

                    dummy_input = torch.zeros(1, in_channels, img_h, img_w)
                    dummy_output = self.feature_extractor(dummy_input)
                    _, C, H, W = dummy_output.shape
                    print(f"Initializing SpatialSoftmax: input ({img_h}x{img_w}) -> feature map (C={C}, H={H}, W={W})")

                # Initialize spatial softmax with determined dimensions
                self.spatial_softmax = SpatialSoftmax(
                    input_shape=(C, H, W),
                    temperature=spatial_softmax_temp
                )
                self.pooling_layer = None

                # Spatial softmax outputs C*2 features (1024 for ResNet18)
                # If encoded_dim=512, we keep 1024 to preserve spatial information
                if encoded_dim == 512:
                    self.output_projection = None
                    self.encoded_dim = C * 2  # Update to actual output dim
                else:
                    self.output_projection = nn.Linear(C * 2, encoded_dim)
            else:
                # Original: keep avgpool, remove fc
                # Output: (B, 512, 1, 1)
                self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
                self.pooling_layer = nn.Flatten()
                self.spatial_softmax = None
                # Remove the redundant FC layer when encoded_dim==512
                if encoded_dim == 512:
                    self.output_projection = None  # No projection needed, already 512
                else:
                    self.output_projection = nn.Linear(512, encoded_dim)

            # Add dropout layer for regularization (applied after pooling/softmax, before projection)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        elif backbone == 'custom':
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, padding=1),
                nn.GroupNorm(num_groups=32//16, num_channels=32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
                nn.GroupNorm(num_groups=64//16, num_channels=64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
                nn.GroupNorm(num_groups=128//16, num_channels=128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1),
                nn.GroupNorm(num_groups=256//16, num_channels=256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, encoded_dim)
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        else:
            raise ValueError(f"Unknown backbone type: {backbone}")

    def forward(self, imgs:torch.Tensor):
        """
        Args:
            imgs: Input images of shape (B, C, H, W)

        Returns:
            features: Encoded features of shape (B, encoded_dim)
        """
        features = self.feature_extractor(imgs)

        if self.backbone_type == 'resnet18':
            if self.use_spatial_softmax:
                # Apply spatial softmax: (B, 512, H, W) -> (B, 1024)
                features = self.spatial_softmax(features)

                # Apply dropout for regularization
                if self.dropout is not None:
                    features = self.dropout(features)

                # Optionally project to desired dimension
                if self.output_projection is not None:
                    features = self.output_projection(features)
            else:
                # Flatten: (B, 512, 1, 1) -> (B, 512)
                features = self.pooling_layer(features)

                # Apply dropout for regularization
                if self.dropout is not None:
                    features = self.dropout(features)

                # Optionally project to desired dimension
                if self.output_projection is not None:
                    features = self.output_projection(features)
        elif self.backbone_type == 'custom':
            # For custom backbone, dropout is already applied if needed
            if self.dropout is not None:
                features = self.dropout(features)

        return features
    
    def get_feature_dim(self, ):
        return self.encoded_dim 

class time_encoding(nn.Module):
    """
    Sinusoidal time embedding with MLP processing.
    Keeps original sinusoidal encoding, then processes through MLP for better representation.
    """
    def __init__(self, t_emb_dim=128, dropout=0.0, use_mlp_layer=True):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.use_mlp_layer = use_mlp_layer
        
        layers = [
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.Mish(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(t_emb_dim * 4, t_emb_dim))

        if use_mlp_layer:
            self.mlp = nn.Sequential(*layers)

    def forward(self, t):
        """
        Args:
            t: Timestep value (scalar, tensor, or numpy array)
        Returns:
            Time embedding of shape (t_emb_dim,) or (batch_size, t_emb_dim)
        """
        half_embedding_dim = self.t_emb_dim // 2
        base_freq = torch.tensor([1 / (10000 ** ((2 * i) / self.t_emb_dim)) for i in range(half_embedding_dim)])

        if isinstance(t, np.ndarray):
            t = to_torch(t)

        modified_freq = t * base_freq
        sinusoidal_encoding = torch.cat([torch.sin(modified_freq), torch.cos(modified_freq)], dim=0).contiguous()

        if self.use_mlp_layer:
            final_encoding = self.mlp(sinusoidal_encoding)
        else:
            final_encoding = sinusoidal_encoding

        return final_encoding


class pos_encoding(nn.Module):
    # Input: (B * 2) agent positions
    def __init__(self, encoded_dim = 512, agent_pos_dim = 2, dropout=0.0):
        super().__init__()
        self.pos_emb_dim = encoded_dim
        self.agent_pos_dim = agent_pos_dim

        # Build encoder with optional dropout
        layers = [
            nn.Linear(agent_pos_dim, 2 * self.pos_emb_dim),
            nn.ReLU()
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(2 * self.pos_emb_dim, self.pos_emb_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


def merge_multimodal_encoding(imgs_encoding, agent_pos_encoding, time_encoding):
    # Expected input shape: B * emd_imgs/ emb_time / emd_agent_pos
    # Note: parameter order changed to match usage in policy
    if imgs_encoding.shape[0] != time_encoding.shape[0] or imgs_encoding.shape[0] != agent_pos_encoding.shape[0] or time_encoding.shape[0] != agent_pos_encoding.shape[0]:
        raise ValueError("The batch size is not compatible for three types of encoding!")

    encodings = [imgs_encoding, agent_pos_encoding, time_encoding]
    # Use torch.cat instead of np.concatenate for tensor operations
    return torch.cat(encodings, dim=1).contiguous()


if __name__ == "__main__":
    print(time_encoding(10, 100))
