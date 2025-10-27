"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main
This file is modified based on dppo lib model/common/module.py file, which is used for argumentation
"""

import torch
import torch.nn as nn
import numpy as np


class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):
        super().__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z


class RandomShiftsAug(nn.Module):
    """Random spatial shift augmentation using grid_sample."""
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        """
        Args:
            x: (N, C, H, W) images
        Returns:
            shifted_x: (N, C, H, W) shifted images
            shift_pixels: (N, 2) pixel shifts [shift_x, shift_y]
        """
        n, c, h, w = x.size()
        assert h == w, f"Image must be square, got {h}x{w}"

        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        # Generate random shifts and record them
        shift_pixels = torch.randint(
            0, 2 * self.pad + 1, size=(n, 2), device=x.device
        ).float()  # (N, 2) for [shift_x, shift_y]

        shift_norm = shift_pixels.view(n, 1, 1, 2) * 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift_norm

        shifted_x = nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )
        return shifted_x, shift_pixels


class RandomPosShifter(nn.Module):
    """
    Random position shift to decouple visual and absolute position relationship.
    Forces model to learn relative spatial understanding from images.
    """
    def __init__(self,
                 workspace_xlim: tuple,
                 workspace_ylim: tuple,
                 shift_ratio: float = 0.3,
                 randomness_intensity: float = 0.5,
                 max_abs_shift: float = None):
        """
        Args:
            workspace_xlim: (min, max) e.g., (-0.2, 0.2)
            workspace_ylim: (min, max) e.g., (-0.2, 0.2)
            shift_ratio: Max shift as ratio of available space (0-1)
            randomness_intensity: Probability of applying shift (0-1)
            max_abs_shift: Absolute maximum shift value (safety limit). If None, no limit.
        """
        super().__init__()
        self.workspace_xlim = workspace_xlim
        self.workspace_ylim = workspace_ylim
        self.randomness_intensity = np.clip(randomness_intensity, 0, 1)
        self.shift_ratio = np.clip(shift_ratio, 0, 1)
        self.max_abs_shift = max_abs_shift

    def forward(self, agent_pos):
        """
        Args:
            agent_pos: (B, T, 2) agent positions

        Returns:
            shifted_pos: (B, T, 2) shifted positions
            shift_value: (B, T, 2) applied shift (for action adjustment)
        """
        batch_size, horizon, _ = agent_pos.shape
        device = agent_pos.device

        # Initialize zero shift
        shift_value = torch.zeros_like(agent_pos)

        # Apply shift with probability
        if np.random.uniform(0, 1) > self.randomness_intensity:
            return agent_pos, shift_value

        # Calculate trajectory boundaries
        agent_pos_max, _ = torch.max(agent_pos, dim=1)  # (B, 2)
        agent_pos_min, _ = torch.min(agent_pos, dim=1)  # (B, 2)

        # Available space to boundaries
        space_left_x = agent_pos_min[:, 0] - self.workspace_xlim[0]  # (B,)
        space_right_x = self.workspace_xlim[1] - agent_pos_max[:, 0]
        space_down_y = agent_pos_min[:, 1] - self.workspace_ylim[0]
        space_up_y = self.workspace_ylim[1] - agent_pos_max[:, 1]

        # Random direction: 0=negative, 1=positive
        dir_x = torch.randint(0, 2, (batch_size,), device=device)
        dir_y = torch.randint(0, 2, (batch_size,), device=device)

        # Random shift intensity
        shift_intensity = torch.rand(batch_size, 2, device=device) * self.shift_ratio

        # Calculate shift
        shift_x = torch.where(
            dir_x == 0,
            -space_left_x * shift_intensity[:, 0],
            space_right_x * shift_intensity[:, 0]
        )
        shift_y = torch.where(
            dir_y == 0,
            -space_down_y * shift_intensity[:, 1],
            space_up_y * shift_intensity[:, 1]
        )

        # Apply absolute maximum shift limit if specified
        if self.max_abs_shift is not None:
            shift_x = torch.clamp(shift_x, -self.max_abs_shift, self.max_abs_shift)
            shift_y = torch.clamp(shift_y, -self.max_abs_shift, self.max_abs_shift)

        # Expand to (B, T, 2)
        shift_value = torch.stack([shift_x, shift_y], dim=1).unsqueeze(1).expand(-1, horizon, -1)
        shifted_pos = agent_pos + shift_value

        return shifted_pos, shift_value


# test random shift
if __name__ == "__main__":
    from PIL import Image
    import requests

    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((96, 96))

    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
    aug = RandomShiftsAug(pad=10)
    image_aug, shift_pixels = aug(image)
    print(f"Shift pixels: {shift_pixels}")
    image_aug = image_aug.squeeze().permute(1, 2, 0).numpy()
    image_aug = Image.fromarray(image_aug.astype(np.uint8))
    image_aug.show()
