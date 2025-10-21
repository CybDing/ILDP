"""
Vision Transformer (ViT) encoding module for image encoding.
This is a wrapper around the ViT implementation that provides a clean interface
matching the encoding API used by the policy.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from genesis_ILDP.model.vision_transformer import (
    vit_tiny, vit_small, vit_base, vit_large, vit_custom
)


class vit_img_encoding(nn.Module):
    """
    Vision Transformer image encoder.
    Provides a clean interface for using ViT as a vision backbone in policies.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        encoded_dim: Output feature dimension
        vit_variant: ViT variant ('vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_custom')
        pretrained: Whether to use pretrained weights (not implemented yet)
        input_image_shape: Tuple (H, W) of input image size
        dropout: Dropout rate for regularization

        ViT-specific parameters:
        vit_patch_size: Size of image patches (default: 16)
        vit_depth: Number of transformer layers (default: 12)
        vit_num_heads: Number of attention heads (default: 12)
        vit_mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        vit_use_cls_token: Use [CLS] token (True) or global avg pooling (False) (default: True)
        vit_embed_dim: ViT embedding dimension (required for vit_custom)
    """
    def __init__(
        self,
        in_channels: int,
        encoded_dim: int = 512,
        vit_variant: str = 'vit_base',
        pretrained: bool = False,
        input_image_shape: Optional[Tuple[int, int]] = None,
        dropout: float = 0.0,
        vit_patch_size: int = 16,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        vit_mlp_ratio: float = 4.0,
        vit_use_cls_token: bool = True,
        vit_embed_dim: Optional[int] = None
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.vit_variant = vit_variant
        self.in_channels = in_channels

        # Determine image size (default to 224 if not provided)
        if input_image_shape is not None:
            img_h, img_w = input_image_shape
            # ViT expects square images, use the smaller dimension
            img_size = min(img_h, img_w)
            if img_h != img_w:
                print(f"Warning: ViT expects square images. Using size {img_size}x{img_size}")
        else:
            img_size = 224  # Default ViT input size
            print(f"Info: input_image_shape not provided, using default {img_size}x{img_size}")

        # Common ViT parameters
        vit_kwargs = {
            'img_size': img_size,
            'in_channels': in_channels,
            'output_dim': encoded_dim,
            'patch_size': vit_patch_size,
            'dropout': dropout,
            'use_cls_token': vit_use_cls_token,
            'pretrained': pretrained
        }

        # Select ViT variant
        if vit_variant == 'vit_tiny':
            self.vit = vit_tiny(**vit_kwargs)
        elif vit_variant == 'vit_small':
            self.vit = vit_small(**vit_kwargs)
        elif vit_variant == 'vit_base':
            self.vit = vit_base(**vit_kwargs)
        elif vit_variant == 'vit_large':
            self.vit = vit_large(**vit_kwargs)
        elif vit_variant == 'vit_custom':
            # For custom ViT, use all provided parameters
            if vit_embed_dim is None:
                raise ValueError("vit_embed_dim must be specified for vit_custom variant")
            vit_kwargs.update({
                'embed_dim': vit_embed_dim,
                'depth': vit_depth,
                'num_heads': vit_num_heads,
                'mlp_ratio': vit_mlp_ratio
            })
            self.vit = vit_custom(**vit_kwargs)
        else:
            raise ValueError(
                f"Unknown ViT variant: {vit_variant}. "
                f"Choose from: vit_tiny, vit_small, vit_base, vit_large, vit_custom"
            )

    def forward(self, imgs: torch.Tensor):
        """
        Forward pass of ViT encoder.

        Args:
            imgs: Input images of shape (B, C, H, W)

        Returns:
            features: Encoded features of shape (B, encoded_dim)
        """
        return self.vit(imgs)

    def get_feature_dim(self):
        """Return output feature dimension."""
        return self.encoded_dim

    def __repr__(self):
        return (f"vit_img_encoding(variant={self.vit_variant}, "
                f"in_channels={self.in_channels}, "
                f"encoded_dim={self.encoded_dim})")


# Convenience functions for creating ViT encoders
def create_vit_tiny_encoder(in_channels=3, encoded_dim=512, input_image_shape=None, **kwargs):
    """Create a ViT-Tiny encoder."""
    return vit_img_encoding(
        in_channels=in_channels,
        encoded_dim=encoded_dim,
        vit_variant='vit_tiny',
        input_image_shape=input_image_shape,
        **kwargs
    )


def create_vit_small_encoder(in_channels=3, encoded_dim=512, input_image_shape=None, **kwargs):
    """Create a ViT-Small encoder."""
    return vit_img_encoding(
        in_channels=in_channels,
        encoded_dim=encoded_dim,
        vit_variant='vit_small',
        input_image_shape=input_image_shape,
        **kwargs
    )


def create_vit_base_encoder(in_channels=3, encoded_dim=512, input_image_shape=None, **kwargs):
    """Create a ViT-Base encoder."""
    return vit_img_encoding(
        in_channels=in_channels,
        encoded_dim=encoded_dim,
        vit_variant='vit_base',
        input_image_shape=input_image_shape,
        **kwargs
    )


def create_vit_large_encoder(in_channels=3, encoded_dim=512, input_image_shape=None, **kwargs):
    """Create a ViT-Large encoder."""
    return vit_img_encoding(
        in_channels=in_channels,
        encoded_dim=encoded_dim,
        vit_variant='vit_large',
        input_image_shape=input_image_shape,
        **kwargs
    )


def create_vit_custom_encoder(
    in_channels=3,
    encoded_dim=512,
    input_image_shape=None,
    vit_embed_dim=384,
    vit_depth=6,
    vit_num_heads=6,
    **kwargs
):
    """
    Create a custom ViT encoder with specified architecture.

    Example usage:
        encoder = create_vit_custom_encoder(
            in_channels=3,
            encoded_dim=512,
            input_image_shape=(96, 96),
            vit_embed_dim=384,
            vit_depth=6,
            vit_num_heads=6,
            vit_patch_size=8,
            dropout=0.1
        )
    """
    return vit_img_encoding(
        in_channels=in_channels,
        encoded_dim=encoded_dim,
        vit_variant='vit_custom',
        input_image_shape=input_image_shape,
        vit_embed_dim=vit_embed_dim,
        vit_depth=vit_depth,
        vit_num_heads=vit_num_heads,
        **kwargs
    )


if __name__ == "__main__":
    # Test ViT encoder
    print("Testing ViT encoders...")

    # Test ViT-Small
    encoder = create_vit_small_encoder(
        in_channels=3,
        encoded_dim=512,
        input_image_shape=(96, 96),
        dropout=0.1
    )
    print(f"\n{encoder}")

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 96, 96)

    # Forward pass
    output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature dim: {encoder.get_feature_dim()}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
