import torch 
import torch.nn as nn

def DummyMask(width, height):
    """
    DummyMask is called when we need to generate mask that is filled with zeros everywhere in the mask matrix

    Param: 
        width: int, 
        height: int, 
    Return:
        dummy mask which is width * height shape
    
    """
    
    mask = torch.zeros(size=(width, height))
    return mask

