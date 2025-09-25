# This file serve as the basic encoding for the global images being stacked; and provide the API functions that is used for time encoding

import torch
import torch.nn as nn 
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

class global_img_encoding(nn.Module):
    def __init__(self, in_channels, encoded_dim=1024, backbone='custom', pretrained=True):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.backbone_type = backbone

        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            # Adjust first conv layer if input channels != 3
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Replace BatchNorm with GroupNorm for EMA compatibility
            from genesis_ILDP.utils.pytorch_util import replace_submodules
            resnet = replace_submodules(
                root_module=resnet,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features)
            )

            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Output: (B, 512, 1, 1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512, encoded_dim)
        elif backbone == 'custom':
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, padding=1),
                nn.ReLU(), 
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
                nn.ReLU(), 
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
                nn.ReLU(), 
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, encoded_dim)
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone}")

    def forward(self, imgs:torch.Tensor):
        features = self.feature_extractor(imgs)
        if self.backbone_type == 'resnet18':
            features = self.flatten(features)
            features = self.fc(features)
        return features
    
    def get_feature_dim(self, ):
        return self.encoded_dim
    

def time_encoding(t, t_emb_dim=256):
    half_embedding_dim = t_emb_dim // 2
    base_freq = torch.tensor([1 / (10000 ** ((2 * i) / t_emb_dim)) for i in range(half_embedding_dim)])
    
    if isinstance(t, np.ndarray):
        t = to_torch(t)
    
    modified_freq = t * base_freq
    final_encoding = torch.cat([torch.sin(modified_freq), torch.cos(modified_freq)], dim=0).contiguous()

    return final_encoding


class pos_encoding(nn.Module):
    # Input: (B * 2) agent positions
    def __init__(self, encoded_dim = 512, agent_pos_dim = 2):
        super().__init__()
        self.pos_emb_dim = encoded_dim
        self.agent_pos_dim = agent_pos_dim
        self.encoder = nn.Sequential(
            nn.Linear(agent_pos_dim, 2 * self.pos_emb_dim),
            nn.ReLU(),
            nn.Linear(2 * self.pos_emb_dim, self.pos_emb_dim)
        )
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
