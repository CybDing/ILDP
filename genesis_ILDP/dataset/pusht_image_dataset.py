from typing import Dict
import torch
import numpy as np
import copy
from genesis_ILDP.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer # Used for easy reading and saving for zarr file

from genesis_ILDP.dataset.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

from genesis_ILDP.dataset.base_dataset import BaseImageDataset
import matplotlib.pyplot as plt


class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        # Directly read into memory, with dict structure for data alignment. (When store is None for this API function)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            horizon=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            horizon=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def visualize_sequence_images(dataset, idx=0, max_frames=8):
    """
    Visualize images from a sequence in the dataset
    
    Args:
        dataset: PushTImageDataset instance
        idx: sequence index to visualize
        max_frames: maximum number of frames to display
    """
    sample = dataset.__getitem__(idx)
    images = sample['obs']['image']  # Shape: (T, 3, 96, 96)
    agent_pos = sample['obs']['agent_pos']  # Shape: (T, 2)
    actions = sample['action']  # Shape: (T, 2)
    
    T = images.shape[0]
    n_frames = min(T, max_frames)
    
    fig, axes = plt.subplots(2, n_frames//2 if n_frames > 4 else n_frames, 
                            figsize=(15, 8 if n_frames > 4 else 4))
    if n_frames <= 4:
        axes = axes.reshape(1, -1) if n_frames > 1 else [[axes]]
    
    for i in range(n_frames):
        row = i // (n_frames//2) if n_frames > 4 else 0
        col = i % (n_frames//2) if n_frames > 4 else i
        
        # Convert from CHW to HWC format and from tensor to numpy
        img = images[i].permute(1, 2, 0).numpy()  # (96, 96, 3)
        
        # Clip values to [0,1] range just in case
        img = np.clip(img, 0, 1)
        
        axes[row][col].imshow(img)
        axes[row][col].set_title(f'Frame {i}\nAgent: ({agent_pos[i][0]:.3f}, {agent_pos[i][1]:.3f})\n'
                                f'Action: ({actions[i][0]:.3f}, {actions[i][1]:.3f})')
        axes[row][col].axis('off')
    
    # Hide empty subplots
    if n_frames < len(axes.flat):
        for i in range(n_frames, len(axes.flat)):
            axes.flat[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_single_image(image_tensor, title="Image"):
    """
    Display a single image tensor
    
    Args:
        image_tensor: tensor of shape (3, H, W) or (H, W, 3)
        title: title for the plot
    """
    # Handle different input formats
    if len(image_tensor.shape) == 3:
        if image_tensor.shape[0] == 3:  # CHW format
            img = image_tensor.permute(1, 2, 0).numpy()
        else:  # HWC format
            img = image_tensor.numpy()
    else:
        raise ValueError(f"Unexpected image shape: {image_tensor.shape}")
    
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def test():
    import os
    import sys

    zarr_path = '../data/train_data/pusht/pusht_test_0919_3.zarr'
    zarr_path = '../data/train_data/pusht/pusht_train_0920.zarr'
    zarr_path = '../data/train_data/pusht/merged_data_0925.zarr' 
    zarr_path = '../data/train_data/pusht/pusht_cchi_v7_replay.zarr'
    zarr_path = '../data/train_data/pusht/genesis_data_20251019_151046.zarr' 
    
    dataset = PushTImageDataset(zarr_path, horizon=16)
    print("Total sequences stored in Replay buffer: ", len(dataset))
    print(len(dataset.__getitem__(10)['obs']['agent_pos']))
    print(len(dataset.__getitem__(10)['action']))

    print("action sequence:  ", dataset.__getitem__(10)['action'])
    # print("agent_pos sequence:  ", dataset.__getitem__(0)['obs']['agent_pos'])
    print("imgs dim: ", dataset.__getitem__(0)['obs']['image'].shape)
    normalizer = dataset.get_normalizer(mode='limits')

    # print("normalizer stat: \n", normalizer.get_input_stats()['action'], normalizer.get_output_stats())
    print(normalizer['action'].normalize(dataset.__getitem__(100)['action']))
    print(normalizer.normalize({'action':(dataset.__getitem__(100)['action'])}))

    # print(normalizer.unnormalize({'action':torch.Tensor([[0.1, 0]])}))

    # Add visualization after existing test code
    # print("\nVisualizing images...")
    visualize_sequence_images(dataset, idx=830, max_frames=16)
    
    # # Display a single frame
    # sample = dataset.__getitem__(-30)
    # first_image = sample['obs']['image'][0]  # First frame
    # print(first_image)
    # display_single_image(first_image, "First Frame of Sequence")
    
    # print(f"Image tensor shape: {first_image.shape}")
    # print(f"Image value range: [{first_image.min():.3f}, {first_image.max():.3f}]")

if __name__ == "__main__":
    test()