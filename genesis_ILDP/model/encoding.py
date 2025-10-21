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
