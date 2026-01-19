# Vision Transformer (ViT) Implementation

This directory contains a fully customizable Vision Transformer implementation for image encoding in the ILDP policy framework.

## Overview

The Vision Transformer (ViT) is an alternative to CNN-based encoders (like ResNet) that uses self-attention mechanisms to process images. ViT splits images into patches and processes them through transformer layers.

## File Structure

```
genesis_ILDP/model/
├── vision_transformer.py    # Core ViT implementation (patches, attention, blocks)
├── vit_encoding.py          # Clean wrapper for policy integration
└── encoding.py              # Main encoding module (supports ViT, ResNet, custom CNN)

genesis_ILDP/config/train/
├── train_action_diffusion_pusht_vit.yaml         # Ready-to-use ViT-Small config
└── train_action_diffusion_pusht_vit_custom.yaml  # Fully customizable ViT config
```

## Quick Start

### 1. Using Pre-configured ViT (Recommended)

Use the ready-to-run ViT-Small configuration:

```bash
python scripts/train.py --config-name=train_action_diffusion_pusht_vit
```

This uses ViT-Small with optimized settings for 88x88 images.

### 2. Using Custom ViT (For Experimentation)

Use the fully customizable configuration:

```bash
python scripts/train.py --config-name=train_action_diffusion_pusht_vit_custom
```

Edit `config/train/train_action_diffusion_pusht_vit_custom.yaml` to customize architecture.

### 3. Integrating ViT into Your Own Config

Add these parameters to your policy config:

```yaml
policy:
  # Vision backbone selection
  vision_backbone: 'vit_small'  # or 'vit_tiny', 'vit_base', 'vit_large', 'vit_custom'
  img_encoding_dim: 512         # Output feature dimension

  # ViT-specific parameters
  vit_patch_size: 8             # Patch size (8, 16, or 32)
  vit_use_cls_token: true       # Use [CLS] token for pooling
  vision_encoder_dropout: 0.1   # Dropout rate

  # For vit_custom only:
  # vit_embed_dim: 512          # Embedding dimension (REQUIRED)
  # vit_depth: 8                # Number of transformer layers
  # vit_num_heads: 8            # Number of attention heads
  # vit_mlp_ratio: 4.0          # MLP expansion ratio
```

## ViT Variants

### Pre-defined Variants

| Variant | Embed Dim | Depth | Heads | Params | Use Case |
|---------|-----------|-------|-------|--------|----------|
| `vit_tiny` | 192 | 12 | 3 | ~5.5M | Fast prototyping, limited compute |
| `vit_small` | 384 | 12 | 6 | ~22M | **Recommended** for 88x88 images |
| `vit_base` | 768 | 12 | 12 | ~86M | Large images, lots of data |
| `vit_large` | 1024 | 24 | 16 | ~307M | Very large images, extensive data |
| `vit_custom` | Custom | Custom | Custom | Varies | Full control over architecture |

### When to Use Each Variant

- **vit_tiny**: Quick experiments, testing pipeline
- **vit_small**: Production use for small-medium images (88-224px)
- **vit_base**: Larger images (224px+) or complex scenes
- **vit_large**: Research, large-scale datasets
- **vit_custom**: Architecture search, specific constraints

## Architecture Parameters

### Core Parameters

#### `vit_patch_size` (default: 16)
Size of image patches in pixels. Smaller patches capture more detail but increase computation.

- **8**: High detail, 4x more patches than 16, best for small images (88-128px)
- **16**: Balanced, standard choice for most tasks
- **32**: Fast, fewer patches, suitable for large images (224px+)

**Example**: For 96x96 image with patch_size=16 → 6×6=36 patches

#### `vit_embed_dim` (required for `vit_custom`)
Embedding dimension of the transformer. Higher = more capacity but more memory.

- **256**: Lightweight, fast, may underfit
- **384**: Good balance for small images
- **512**: Strong performance, moderate compute
- **768**: ViT-Base standard, high capacity

**Note**: Must be divisible by `vit_num_heads`

#### `vit_depth` (default: 12)
Number of transformer layers. More layers = deeper network.

- **4-6**: Shallow, fast, good for simple patterns
- **8-12**: Standard, good for most tasks
- **12-24**: Deep, captures complex patterns, slower

#### `vit_num_heads` (default: 12)
Number of attention heads in multi-head attention.

- **4**: Fewer heads, faster
- **6-8**: Balanced
- **12-16**: More attention diversity

**Constraint**: `vit_embed_dim % vit_num_heads == 0`

#### `vit_mlp_ratio` (default: 4.0)
Ratio of MLP hidden dimension to embedding dimension.

- **2.0**: Compact, less non-linearity
- **4.0**: Standard, balanced
- **8.0**: High capacity, more non-linearity

#### `vit_use_cls_token` (default: true)
How to pool patch features into a single vector.

- **true**: Use learnable [CLS] token (better for classification-like tasks)
- **false**: Global average pooling (better for dense prediction)

### Regularization Parameters

#### `vision_encoder_dropout` (default: 0.1)
Dropout rate applied in attention and MLP layers.

- **0.0**: No dropout, may overfit
- **0.1**: Standard, good regularization
- **0.2-0.3**: Strong regularization for small datasets

## Training Tips

### 1. Learning Rate

ViT typically needs higher learning rates than CNNs:

```yaml
optimizer:
  lr: 0.0003  # ViT (vs 0.0001 for ResNet)
```

### 2. Weight Decay

ViT benefits from stronger weight decay:

```yaml
optimizer:
  weight_decay: 0.05  # ViT (vs 1e-5 for ResNet)
```

### 3. Batch Size

ViT uses more memory than ResNet. Reduce batch size if OOM:

```yaml
training:
  batch_size: 64  # ViT (vs 128 for ResNet)
```

### 4. Warmup

Use warmup for stable training:

```yaml
training:
  lr_warmup_steps: 500
```

### 5. Dropout

Start with 0.1 and increase if overfitting:

```yaml
policy:
  vision_encoder_dropout: 0.1
```

## Example Configurations

### Lightweight ViT (Fast, ~10M params)

```yaml
vision_backbone: 'vit_custom'
vit_patch_size: 16
vit_embed_dim: 256
vit_depth: 6
vit_num_heads: 4
vit_mlp_ratio: 4.0
vision_encoder_dropout: 0.1
```

### Balanced ViT (Recommended, ~35M params)

```yaml
vision_backbone: 'vit_custom'
vit_patch_size: 16
vit_embed_dim: 512
vit_depth: 8
vit_num_heads: 8
vit_mlp_ratio: 4.0
vision_encoder_dropout: 0.1
```

### High-Detail ViT (Small patches, ~25M params)

```yaml
vision_backbone: 'vit_custom'
vit_patch_size: 8
vit_embed_dim: 384
vit_depth: 6
vit_num_heads: 6
vit_mlp_ratio: 4.0
vision_encoder_dropout: 0.1
```

## Programmatic Usage

### Using the Clean Wrapper

```python
from genesis_ILDP.model.vit_encoding import vit_img_encoding

# Create ViT-Small encoder
encoder = vit_img_encoding(
    in_channels=3,
    encoded_dim=512,
    vit_variant='vit_small',
    input_image_shape=(88, 88),
    dropout=0.1
)

# Forward pass
import torch
images = torch.randn(4, 3, 88, 88)  # Batch of 4 images
features = encoder(images)  # (4, 512)
```

### Using the Policy Integration

The ViT is automatically integrated into `global_img_encoding`:

```python
from genesis_ILDP.model.encoding import global_img_encoding

# Create ViT encoder
encoder = global_img_encoding(
    in_channels=3,
    encoded_dim=512,
    backbone='vit_small',
    input_image_shape=(88, 88),
    dropout=0.1,
    vit_patch_size=8
)

# Forward pass
features = encoder(images)
```

### Using Convenience Functions

```python
from genesis_ILDP.model.vit_encoding import (
    create_vit_small_encoder,
    create_vit_custom_encoder
)

# ViT-Small
encoder = create_vit_small_encoder(
    encoded_dim=512,
    input_image_shape=(88, 88),
    dropout=0.1
)

# Custom ViT
encoder = create_vit_custom_encoder(
    encoded_dim=512,
    input_image_shape=(96, 96),
    vit_embed_dim=384,
    vit_depth=6,
    vit_num_heads=6,
    vit_patch_size=8,
    dropout=0.1
)
```

## Performance Considerations

### Memory Usage

ViT uses more memory than ResNet due to self-attention:

| Model | Image Size | Batch Size | GPU Memory |
|-------|------------|------------|------------|
| ResNet18 | 96x96 | 128 | ~4 GB |
| ViT-Small | 96x96 | 64 | ~4 GB |
| ViT-Base | 96x96 | 32 | ~4 GB |

**Tip**: Reduce batch size if you encounter OOM errors.

### Computation Time

| Model | Image Size | Patches | FLOPs (relative) |
|-------|------------|---------|------------------|
| ResNet18 | 96x96 | - | 1.0x |
| ViT-Small (patch=16) | 96x96 | 36 | 1.2x |
| ViT-Small (patch=8) | 96x96 | 144 | 2.5x |

**Tip**: Use larger patches (16 or 32) for faster inference.

### Parameter Count

| Model | Parameters | Relative Size |
|-------|------------|---------------|
| ResNet18 | ~11M | 1.0x |
| ViT-Tiny | ~5.5M | 0.5x |
| ViT-Small | ~22M | 2.0x |
| ViT-Base | ~86M | 7.8x |

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce batch size: `batch_size: 32` or `batch_size: 64`
2. Use smaller variant: `vit_tiny` or `vit_small`
3. Increase patch size: `vit_patch_size: 16` or `32`
4. Reduce embedding dim (for custom): `vit_embed_dim: 256`

### Issue: "vit_embed_dim must be divisible by num_heads"

**Solution**: Ensure `vit_embed_dim % vit_num_heads == 0`

Valid combinations:
- embed_dim=384, num_heads=6 ✓
- embed_dim=512, num_heads=8 ✓
- embed_dim=768, num_heads=12 ✓
- embed_dim=512, num_heads=6 ✗ (512 % 6 ≠ 0)

### Issue: Poor Performance / Underfitting

**Solutions**:
1. Increase model capacity:
   - Use `vit_small` or `vit_base` instead of `vit_tiny`
   - Increase depth: `vit_depth: 12`
   - Increase embed_dim: `vit_embed_dim: 512` or `768`
2. Reduce regularization:
   - Lower dropout: `vision_encoder_dropout: 0.05`
   - Lower weight decay: `weight_decay: 0.01`
3. Train longer: `num_epochs: 1000`

### Issue: Overfitting

**Solutions**:
1. Increase regularization:
   - Higher dropout: `vision_encoder_dropout: 0.2`
   - Higher weight decay: `weight_decay: 0.1`
2. Use data augmentation
3. Reduce model capacity:
   - Use smaller variant: `vit_tiny`
   - Reduce depth: `vit_depth: 6`

### Issue: Slow Training

**Solutions**:
1. Reduce model size: Use `vit_tiny` or `vit_small`
2. Increase patch size: `vit_patch_size: 16` or `32`
3. Reduce depth: `vit_depth: 6` or `8`
4. Enable mixed precision training (if supported)

## Architecture Details

### Patch Embedding

Images are split into non-overlapping patches and linearly projected:

```
Input: (B, 3, H, W)
  ↓ Conv2d(3, embed_dim, kernel=patch_size, stride=patch_size)
  ↓ Flatten spatial dims
Output: (B, num_patches, embed_dim)
```

### Positional Embedding

Learnable position embeddings are added to patch embeddings:

```
pos_embed: (1, num_patches+1, embed_dim)  # +1 for [CLS] token
output = patch_embed + pos_embed
```

### Transformer Block

Each block consists of:
1. Layer Normalization
2. Multi-Head Self-Attention
3. Residual connection
4. Layer Normalization
5. MLP (FC → GELU → Dropout → FC)
6. Residual connection

### Output

Two pooling strategies:
- **[CLS] token**: Use the first token's representation
- **Global Average Pooling**: Average all patch tokens

## References

- Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  - https://arxiv.org/abs/2010.11929
- Original implementation: https://github.com/google-research/vision_transformer

## Support

For issues or questions about the ViT implementation:
1. Check this README
2. Review the example configs in `config/train/`
3. Examine `vision_transformer.py` for implementation details
4. Test with the clean wrapper in `vit_encoding.py`
