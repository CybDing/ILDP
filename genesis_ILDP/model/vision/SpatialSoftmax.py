import torch 
import numpy as np
import torch.nn as nn

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