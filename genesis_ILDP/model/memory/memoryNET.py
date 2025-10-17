"""
This file will contain the primitive of memoryNET which is used to grasp the overall understanding of the robot current situation in the latent space
and trained real time with the robot with the visual encoder. 

The net is mainly a sequential model, which takes in the current observation obs for example the img, the agent pos, and update its own representation 
of the feature map just like the temporal features created by the agent_pos and the current_img, and the new encoding will work together as the cond
? to predict the noise(or something else for merging) the training stage includes the random sampling the ratio of when to use the memory and when 
to use combined features for training both the encoder and the visual encoder or train the memory unit only(which require it has a good start from the 
visual encoder information (or just the same time))

problems to solve: what are some good benchmarks showing this is effective in image hindering (long task completion rate and so on)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
