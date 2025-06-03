import torch
import genesis as gs
import numpy as np

def to_torch(array: np.array, float=True):
    if float:
        return torch.from_numpy(array).float().to(gs.device)
    else: return torch.from_numpy(array).int().to(gs.device)

def to_numpy(tensor:torch.tensor, float=True):
    if float:
        return tensor.to('cpu').float().numpy()
    else:
        return tensor.to('cpu').int().numpy()