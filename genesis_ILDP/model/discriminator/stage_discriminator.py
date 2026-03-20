import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional, Callable

from genesis_ILDP.model.discriminator.base_discriminator import BaseImgDiscriminator
from genesis_ILDP.model.common.mlp import MLP
from genesis_ILDP.model.vision.vit import ViTEncoder


class StageDiscriminator(BaseImgDiscriminator):
    """
    Predict normalized task progress (0-1) from a single image.

    The encoder is pluggable: by default a ViT is used, but a custom vision
    backbone (e.g., the one used for action prediction) can be injected
    via `vision_encoder`. If no explicit output dimension is provided, it
    will be inferred automatically.
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        output_mlp_hidden: list,
        emb_dim: int = 256,
        num_heads: int = 4,
        depth: int = 4,
        use_cls_token: bool = False,
        hidden_activation: str = "ReLU",
        out_activation: str = "Sigmoid",
        pool: str = "mean",
        vision_encoder: Optional[nn.Module] = None,
        encoder_output_dim: Optional[int] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(in_channels=in_channels, img_size=img_size)

        # optionally reuse an external encoder (e.g., action policy backbone)
        if vision_encoder is None:
            self.encoder = ViTEncoder(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_channels,
                embed_dim=emb_dim,
                num_heads=num_heads,
                depth=depth,
                use_cls_token=use_cls_token,
            )
            inferred_out_dim = emb_dim
        else:
            self.encoder = vision_encoder
            inferred_out_dim = self._infer_output_dim(
                vision_encoder, in_channels=in_channels, img_size=img_size
            )

        self.pool = pool
        self.encoder_output_dim = encoder_output_dim or inferred_out_dim
        self.freeze_encoder = freeze_encoder

        dim_list = [self.encoder_output_dim] + list(output_mlp_hidden) + [1]
        self.output_mlp = MLP(
            dim_list=dim_list,
            activation_type=hidden_activation,
            out_activation_type=out_activation,
            use_drop_final=False,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _infer_output_dim(self, encoder: nn.Module, in_channels: int, img_size: int) -> int:
        if hasattr(encoder, "output_dim"):
            out_dim = encoder.output_dim()
            if isinstance(out_dim, int):
                return out_dim
        # fallback: run a tiny forward pass (works for encoders returning tokens or vectors)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            encoded = encoder(dummy)
            if encoded.dim() > 2:
                encoded = encoded.flatten(start_dim=1)
            return int(encoded.shape[-1])

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, ViTEncoder):
            encoded = self.encoder(obs, pool=self.pool)
        else:
            if hasattr(self.encoder, "encode"):
                encoded = self.encoder.encode(obs)
            else:
                encoded = self.encoder(obs)

        if encoded.dim() > 2:
            encoded = encoded.flatten(start_dim=1)
        return encoded

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, 1) with normalized progress in [0, 1]
        """
        features = self._encode(obs)
        pred_progress = self.output_mlp(features).squeeze(-1)
        return pred_progress

    def _compute_loss(self, obs: torch.Tensor, ref_progress: torch.Tensor):
        pred_progress = self(obs)
        ref_progress = ref_progress.squeeze(-1)
        progress_loss = F.mse_loss(pred_progress, ref_progress)
        return progress_loss
    
    def optimizer_setup(self, vit_lr: float, mlp_lr: float, weight_decay: float = 0.0):
        if not self.freeze_encoder:
            self.vit_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.encoder.parameters()),
                lr=vit_lr,
                weight_decay=weight_decay,
            )
        else:
            self.vit_optimizer = None

        self.mlp_optimizer = torch.optim.Adam(
            self.output_mlp.parameters(),
            lr=mlp_lr,
            weight_decay=weight_decay,
        )
    
