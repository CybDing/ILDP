"""
Vision Transformer (ViT) implementation for image encoding.
Fully customizable through config parameters.

Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.

    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Projection layer: Conv2d can be more efficient than reshaping + linear
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        qkv_bias: Whether to use bias in qkv projection
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Combined QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim) where N is sequence length
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Feed-forward network (MLP) with GELU activation.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (typically 4x in_features)
        out_features: Output dimension
        dropout: Dropout rate
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with pre-normalization.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout: Dropout rate
        attn_dropout: Attention dropout rate
        qkv_bias: Whether to use bias in qkv projection
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()

        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            qkv_bias=qkv_bias
        )

        # Layer normalization
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        # Pre-norm + attention + residual
        x = x + self.attn(self.norm1(x))

        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image encoding.
    Fully customizable through configuration parameters.

    Args:
        img_size: Input image size (height/width, assumes square)
        patch_size: Size of image patches
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout: Dropout rate for MLP and projection
        attn_dropout: Dropout rate for attention
        qkv_bias: Whether to use bias in qkv projection
        use_cls_token: Whether to use [CLS] token (True) or global average pooling (False)
        output_dim: Output feature dimension (if None, uses embed_dim)
        pretrained: Whether to use pretrained weights (not implemented yet, placeholder)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        qkv_bias: bool = True,
        use_cls_token: bool = True,
        output_dim: Optional[int] = None,
        pretrained: bool = False
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_cls_token = use_cls_token
        self.output_dim = output_dim or embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Class token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = num_patches + 1
        else:
            self.cls_token = None
            num_tokens = num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                qkv_bias=qkv_bias
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection (if output_dim != embed_dim)
        if self.output_dim != embed_dim:
            self.head = nn.Linear(embed_dim, self.output_dim)
        else:
            self.head = nn.Identity()

        # Initialize weights
        self._init_weights()

        if pretrained:
            print("Warning: Pretrained weights loading not implemented yet.")

    def _init_weights(self):
        """Initialize weights following ViT paper."""
        # Initialize positional embedding with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize cls token with truncated normal
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize other layers
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        """Initialize weights for each layer type."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Following timm's initialization
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of Vision Transformer.

        Args:
            x: (B, C, H, W) - Batch of images

        Returns:
            (B, output_dim) - Feature vectors
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token if used
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Extract features
        if self.use_cls_token:
            # Use [CLS] token
            x = x[:, 0]  # (B, embed_dim)
        else:
            # Global average pooling
            x = x.mean(dim=1)  # (B, embed_dim)

        # Output projection
        x = self.head(x)  # (B, output_dim)

        return x

    def get_feature_dim(self):
        """Return output feature dimension."""
        return self.output_dim


def vit_tiny(img_size=224, in_channels=3, output_dim=None, **kwargs):
    """ViT-Tiny configuration."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        embed_dim=192,
        depth=12,
        num_heads=3,
        output_dim=output_dim,
        **kwargs
    )


def vit_small(img_size=224, in_channels=3, output_dim=None, **kwargs):
    """ViT-Small configuration."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        embed_dim=384,
        depth=12,
        num_heads=6,
        output_dim=output_dim,
        **kwargs
    )


def vit_base(img_size=224, in_channels=3, output_dim=None, **kwargs):
    """ViT-Base configuration."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        embed_dim=768,
        depth=12,
        num_heads=12,
        output_dim=output_dim,
        **kwargs
    )


def vit_large(img_size=224, in_channels=3, output_dim=None, **kwargs):
    """ViT-Large configuration."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        output_dim=output_dim,
        **kwargs
    )


def vit_custom(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=512,
    depth=6,
    num_heads=8,
    output_dim=None,
    **kwargs
):
    """
    Custom ViT configuration - fully configurable.
    This is the most flexible option for experimentation.
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        output_dim=output_dim,
        **kwargs
    )
