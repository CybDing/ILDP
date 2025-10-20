"""
This file will contain the primitive of memoryNET which is used to grasp the overall understanding of the robot current situation in the latent space
and trained real time with the robot with the visual encoder. 

The net is mainly a sequential model, which takes in the current observation obs for example the img, the agent pos, and update its own representation 
of the feature map just like the temporal features created by the agent_pos and the current_img, and the new encoding will work together as the cond
? to predict the noise(or something else for merging) the training stage includes the random sampling the ratio of when to use the memory and when 
to use combined features for training both the encoder and the visual encoder or train the memory unit only(which require it has a good start from the 
visual encoder information (or just the same time))

problems to solve: what are some good benchmarks showing this is effective in image hinderance (long task completion rate and so on)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MemoryNet:
    """
    MemoryNet is a neural networks that try to combine the markov chains evolution(which mostly should be under the physic law and change
    in certain way in the latent space that should be quite similar with the adjacent scenes across the time dimension) 
    
    and also be able to capture the dynamic changing events that happens quickly and related to the task it is currently focused on and 
    ignore other distractionslike the visual hinderance or the changing of lightning or so on. 

    Using the MemoryNet for conditioning the unet for predicting noise could take both advantages of the memory and current scenes, and should be 
    able to use the combined information(like the updating using the visual information for memory) to use that as the final condition to 
    plug into the unet for predicting the actions

    """
    def __init__():

        pass

    def forward():
        """
        ## Args: 
        - prev_memory: torch.Tensor
        - interaction: dict(str: torch.Tensor) 
            interaction may contain:
                1. The robot's taken action / actions(sequential) the bot took these steps
                2. Real vision like rgb imgs, the point cloud depth information and so on which has been seen by the bot at this step
                3. New high level decision robot decide to take in the future sequential of actions (periodic)
        - 
        
        """
        pass 

    def compute_loss():

        pass 

