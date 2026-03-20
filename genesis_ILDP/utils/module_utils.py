import torch.nn as nn
import torch
import torch.optim as optim
from typing import Type

ModuleDict = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'sigmoid': nn.Sigmoid,
    'batchnorm': nn.BatchNorm1d,
    None: None,
}
ParamType = Type[nn.Parameter]

def get_optimizer(type:str, 
                 params: ParamType,
                 lr: float, 
                 weight_decay: float):
    if type.lower() == 'adam':
        optimizer = optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    
    return optimizer