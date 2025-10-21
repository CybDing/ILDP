import torch 
import numpy as np
import torch.nn as nn

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