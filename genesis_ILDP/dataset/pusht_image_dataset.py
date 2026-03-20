from typing import Dict
import torch
import numpy as np
import copy
from genesis_ILDP.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer # Used for easy reading and saving for zarr file

from genesis_ILDP.dataset.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_imagenet_normalizer

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
            max_train_episodes=None,
            workspace_limits=None,  # Add workspace limits parameter
            image_normalizer: str = "range"  # "range" ([-1,1]) or "imagenet"
            ):

        super().__init__()
        self.workspace_limits = workspace_limits
        # Directly read into memory, with dict structure for data alignment. (When store is None for this API function)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])

        # Load final_intersection_ratio if available
        try:
            import zarr
            root = zarr.open(zarr_path, mode='r')
            if 'final_intersection_ratio' in root:
                self.final_intersection_ratios = np.array(root['final_intersection_ratio'][:])
            else:
                self.final_intersection_ratios = None
        except Exception as e:
            print(f"Warning: Could not load final_intersection_ratio: {e}")
            self.final_intersection_ratios = None

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
        self.zarr_path = zarr_path
        self.image_normalizer = image_normalizer

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

    def get_normalizer(self, mode='limits', use_image_normalizer=True, **kwargs):
        """
        Get normalizer for the dataset.
        If workspace_limits is provided, use manual normalization instead of data-based.
        """
        from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

        normalizer = LinearNormalizer()

        # Use manual limits if provided
        if self.workspace_limits is not None:
            print("Using manual workspace limits for normalization:")
            print(f"  X: {self.workspace_limits['xlim']}")
            print(f"  Y: {self.workspace_limits['ylim']}")

            # Create manual normalizers for action and agent_pos
            for key in ['action', 'agent_pos']:
                normalizer[key] = self._create_manual_normalizer(
                    self.workspace_limits, mode=mode, **kwargs
                )
        else:
            # Original behavior: fit from data
            print("Computing normalization from data (workspace_limits not provided)")
            data = {
                'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state'][...,:2]
            }
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        if use_image_normalizer:
            if self.image_normalizer == "imagenet":
                normalizer['image'] = get_imagenet_normalizer()
            elif self.image_normalizer == "range":
                normalizer['image'] = get_image_range_normalizer()
            else:
                raise ValueError(f"Unknown image_normalizer option: {self.image_normalizer}")

        return normalizer

    def _create_manual_normalizer(self, limits, mode='limits', output_max=1.0, output_min=-1.0, **kwargs):
        """
        Create a normalizer with manual limits instead of data-based statistics.
        """
        import torch
        from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

        x_min, x_max = limits['xlim']
        y_min, y_max = limits['ylim']

        if mode == 'limits':
            # For 2D data: [x, y]
            input_min = torch.tensor([x_min, y_min], dtype=torch.float32)
            input_max = torch.tensor([x_max, y_max], dtype=torch.float32)
            input_mean = (input_min + input_max) / 2
            input_std = (input_max - input_min) / 4  # Approximate std

            # Compute scale and offset
            input_range = input_max - input_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min

            # Create normalizer manually
            input_stats_dict = {
                'min': input_min,
                'max': input_max,
                'mean': input_mean,
                'std': input_std
            }

            return SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=input_stats_dict
            )
        else:
            raise NotImplementedError(f"Manual normalization for mode={mode} not implemented")

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255 # move channel to the second dimension 

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
    print(images.shape)
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
    zarr_path = '../data/train_data/pusht/genesis_data_20251019_151046.zarr' 
    zarr_path = '../data/train_data/pusht/pusht_cchi_v7_replay.zarr'
    zarr_path = '../data/train_data/pusht/pusht_1027_cleaned.zarr'
    zarr_path = '../data/train_data/pusht/merged_data_0925.zarr' 
    
    dataset = PushTImageDataset(zarr_path, horizon=16)
    print("Total sequences stored in Replay buffer: ", len(dataset))
    print(len(dataset.__getitem__(10)['obs']['agent_pos']))
    print(len(dataset.__getitem__(10)['action']))

    print("action sequence:  ", dataset.__getitem__(10)['action'])
    # print("agent_pos sequence:  ", dataset.__getitem__(0)['obs']['agent_pos'])
    print("imgs range[", torch.max(dataset.__getitem__(0)['obs']['image']), ", ", torch.min(dataset.__getitem__(0)['obs']['image']))
    normalizer = dataset.get_normalizer(mode='limits')

    # print("normalizer stat: \n", normalizer.get_input_stats()['action'], normalizer.get_output_stats())
    print(dataset.__getitem__(10)['action'])
    print(normalizer.normalize({'action':(dataset.__getitem__(10)['action'])}))

    # Access final_intersection_ratio for episode 1
    if dataset.final_intersection_ratios is not None:
        print(f"Episode 1 final intersection ratio: {dataset.final_intersection_ratios[1]}")
        print(f"All final intersection ratios: {dataset.final_intersection_ratios}")
    else:
        print("No final_intersection_ratio data available")

    # print(normalizer.unnormalize({'action':torch.Tensor([[0.1, 0]])}))

    # Add visualization after existing test code
    # print("\nVisualizing images...")
    visualize_sequence_images(dataset, idx=1640, max_frames=16)
    
    # # Display a single frame
    # sample = dataset.__getitem__(-30)
    # first_image = sample['obs']['image'][0]  # First frame
    # print(first_image)
    # display_single_image(first_image, "First Frame of Sequence")
    
    # print(f"Image tensor shape: {first_image.shape}")
    # print(f"Image value range: [{first_image.min():.3f}, {first_image.max():.3f}]")

if __name__ == "__main__":
    test()
