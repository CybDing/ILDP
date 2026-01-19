import torch 
import torch.nn as nn


class BaseEncoding(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forwards(self, ):
        raise NotImplementedError
    
    def __len__(self, ):
        return self.output_dim