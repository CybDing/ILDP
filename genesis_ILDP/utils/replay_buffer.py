import torch 
from collections import OrderedDict

class ReplayBuffer:
    def __init__(self, buffer_size, ):
        self.buffer_size = buffer_size
        self.data = OrderedDict(
            s1=torch.empty(size=(self.buffer_size, )), 
            s2=torch.empty(size=(self.buffer_size, )), 
            a1=torch.empty(size=(self.buffer_size, )), 
            r1=torch.empty(size=(self.buffer_size, )), 
            d=torch.empty(size=(self.buffer_size, ))
        )
        self.ptr = 0 
        self.full=False
    
    def add(self, s1, s2, a1, r1, d):
        # add: chain_before, chain_after, chain_action, reward, done
        idx = self.ptr % self.buffer_size
        self.data['s1'][idx] = s1
        self.data['s2'][idx] = s2
        self.data['a1'][idx] = a1
        self.data['r1'][idx] = r1
        self.data['d'][idx] = d
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
