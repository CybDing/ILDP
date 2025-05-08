import torch

def to_torch(array, float=True):
    if float == True:
        return torch.from_numpy(array).float()
    else: return torch.from_numpy(array).int()
    