import torch
import numpy as np 
import torch.nn as nn


class BaseImgDiscriminator(nn.Module):
    """
    Minimal base class for image-based discriminators.
    Provides a tiny optimizer stepping helper that assumes
    subclasses define `vit_optimizer` and `mlp_optimizer`.
    """
    def __init__(self, in_channels, img_size):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size

    def forward(self, x):
        raise NotImplementedError
    
    def compute_loss(self):
        raise NotImplementedError

    def step(self, obs, ref_progress):
        if not hasattr(self, "vit_optimizer") or not hasattr(self, "mlp_optimizer"):
            raise RuntimeError("Optimizers are not initialized; call `optimizer_setup` first.")

        if self.vit_optimizer is not None:
            self.vit_optimizer.zero_grad(set_to_none=True)
        if self.mlp_optimizer is not None:
            self.mlp_optimizer.zero_grad(set_to_none=True)

        loss = self._compute_loss(obs, ref_progress)
        loss.backward()
        
        if self.vit_optimizer is not None:
            self.vit_optimizer.step()
        if self.mlp_optimizer is not None:
            self.mlp_optimizer.step()

        return loss


# backward compatibility with previous import style
BaseDiscriminator = BaseImgDiscriminator
    
