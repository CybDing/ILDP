import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union


class PatchTokenizer(nn.Module):
    """
    PatchTokenizer: from imgs to tokens patches.
    Output: [B, N, D]
    """
    def __init__(self, in_chans: int = 3, embed_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)                      # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)      # [B, N, D]
        return x
    
class ViTEncoder(nn.Module):
    """
    ViT Encoder:
    - PatchTokenizer (Conv stride patch)
    - (optional) CLS token
    - + learnable pos_embed (支持任意分辨率时做插值)
    - TransformerEncoder (torch 内置)
    - 输出 token hidden states
    """
    def __init__(
        self,
        img_size: int = 224,                 # 需要保证训练，evaluation 中间图片的输入尽可能是一致的，否则在使用的时候需要将pos_encoding进行插值
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,             # Pre-LN: True 更接近主流 ViT
        use_cls_token: bool = True,
        return_all_layers: bool = False,     # True: 返回每层输出（需要 enable hook 逻辑）
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.return_all_layers = return_all_layers

        # 1) patch tokenize
        self.tokenizer = PatchTokenizer(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)

        # 2) CLS + pos embed
        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        num_patches = grid_h * grid_w
        self.num_patches_init = num_patches
        self.patch_size = patch_size

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        ff_dim = int(embed_dim * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,         # 关键：让输入用 [B, N, D]
            norm_first=norm_first,    # True => Pre-LN
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.final_norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 若需要每层输出，可用 forward hook 收集
        self._layer_outputs: List[torch.Tensor] = []
        if self.return_all_layers:
            for layer in self.encoder.layers:
                layer.register_forward_hook(self._save_layer_output_hook)

    def _save_layer_output_hook(self, module, inputs, output):
        # output: [B, N, D]
        self._layer_outputs.append(output)

    def _resize_pos_embed(self, pos_embed: torch.Tensor, new_grid_hw: Tuple[int, int]) -> torch.Tensor:
        """
        输入分辨率不等于初始化 img_size 时，对 patch pos_embed 做插值。
        pos_embed: [1, N(+1), D]
        """
        if self.use_cls_token:
            cls_pos = pos_embed[:, :1, :]              # [1,1,D]
            patch_pos = pos_embed[:, 1:, :]            # [1,N,D]
        else:
            cls_pos = None
            patch_pos = pos_embed

        N, D = patch_pos.shape[1], patch_pos.shape[2]
        old_grid = int(math.sqrt(N))
        if old_grid * old_grid != N:
            # 初始化时如果不是正方形网格，你需要记录 grid_h/grid_w；这里给简化实现（常见是正方形输入）
            raise ValueError(f"Cannot infer old grid from N={N}. Provide square init or store grid_h/grid_w.")

        patch_pos = patch_pos.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)  # [1,D,H,W]
        patch_pos = torch.nn.functional.interpolate(
            patch_pos, size=new_grid_hw, mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid_hw[0] * new_grid_hw[1], D)  # [1,newN,D]

        if self.use_cls_token:
            return torch.cat([cls_pos, patch_pos], dim=1)  # [1,1+newN,D]
        return patch_pos

    def output_dim(self) -> int:
        return self.embed_dim

    def forward(
        self,
        x: torch.Tensor,
        return_tokens: bool = True,
        pool: str = "none",  # "none" | "cls" | "mean"
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        x: [B,C,H,W]
        return:
          - pool="none": token hidden states [B, N(+1), D]
          - pool="cls":  [B, D]
          - pool="mean": [B, D]  (对 patch tokens 求均值；若 use_cls_token, 则不含 cls)
        """
        self._layer_outputs = []  # reset (用于 return_all_layers)

        tokens = self.tokenizer(x)  # [B, N, D]
        B, N, D = tokens.shape

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)     # [B,1,D]
            tokens = torch.cat([cls, tokens], dim=1)   # [B,1+N,D]

        # pos embed：若输入分辨率变化，做插值
        H, W = x.shape[-2], x.shape[-1]
        new_grid = (H // self.patch_size, W // self.patch_size)
        expected_len = (new_grid[0] * new_grid[1]) + (1 if self.use_cls_token else 0)

        pos = self.pos_embed
        if pos.shape[1] != expected_len:
            pos = self._resize_pos_embed(pos, new_grid)

        tokens = self.pos_drop(tokens + pos)

        out = self.encoder(tokens)      # [B, N(+1), D]
        out = self.final_norm(out)

        if pool == "cls":
            if not self.use_cls_token:
                raise ValueError("pool='cls' requires use_cls_token=True")
            pooled = out[:, 0, :]
            result = pooled
        elif pool == "mean":
            if self.use_cls_token:
                pooled = out[:, 1:, :].mean(dim=1)
            else:
                pooled = out.mean(dim=1)
            result = pooled
        else:
            result = out  # token-level hidden states

        if self.return_all_layers:
            return result, self._layer_outputs

        return result