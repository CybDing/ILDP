import torch
import random
import numpy as np
from typing import List


class SmartBatchSampler(torch.utils.data.Sampler):
    """
    Elegant batch sampler that groups sequences by length.
    - Sorts all samples by sequence length
    - Randomly shuffles within same-length groups
    - Creates batches from sorted list (minimizes padding)
    """
    def __init__(self, dataset, batch_size=8, shuffle_same_length=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_same_length = shuffle_same_length

    def __iter__(self):
        # Get all sample indices with their lengths
        indices_with_length = []
        for idx in range(len(self.dataset)):
            chunk_info = self.dataset.sampler.chunks[idx]
            length = chunk_info['chunk_length']
            indices_with_length.append((idx, length))

        # Sort by length
        indices_with_length.sort(key=lambda x: x[1])

        # Group by length and shuffle within groups
        if self.shuffle_same_length:
            grouped = {}
            for idx, length in indices_with_length:
                if length not in grouped:
                    grouped[length] = []
                grouped[length].append(idx)

            # Shuffle within each length group
            for length in grouped:
                random.shuffle(grouped[length])

            # Rebuild sorted list with shuffled groups
            all_indices = []
            for length in sorted(grouped.keys()):
                all_indices.extend(grouped[length])
        else:
            all_indices = [idx for idx, _ in indices_with_length]

        # Generate batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i+self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def pad_sequence_batch(batch: List) -> dict:
    """
    Simple padding to max length in batch.
    Returns dict with padded tensors and mask.
    """
    if len(batch) == 0:
        return {}

    max_T = max(sample['chunk_length'] for sample in batch)
    B = len(batch)

    # Get shapes from first sample
    first_sample = batch[0]
    img = first_sample['obs']['image']
    if isinstance(img, torch.Tensor):
        img_shape = img.shape[1:]
    else:
        img_shape = img.shape[1:] if len(img.shape) > 1 else img.shape

    action_dim = first_sample['action'].shape[-1]

    # Initialize tensors on CPU explicitly to support pin_memory
    padded_imgs = torch.zeros(B, max_T, *img_shape, dtype=torch.float32, device='cpu')
    padded_pos = torch.zeros(B, max_T, 2, dtype=torch.float32, device='cpu')
    padded_actions = torch.zeros(B, max_T, action_dim, dtype=torch.float32, device='cpu')
    padding_masks = torch.zeros(B, max_T, dtype=torch.float32, device='cpu')

    for b, sample in enumerate(batch):
        T = sample['chunk_length']

        # Convert to tensor if needed and ensure on CPU for pin_memory compatibility
        img_data = sample['obs']['image']
        if not isinstance(img_data, torch.Tensor):
            img_data = torch.from_numpy(img_data).float()
        img_data = img_data.cpu()  # Ensure CPU

        pos_data = sample['obs']['agent_pos']
        if not isinstance(pos_data, torch.Tensor):
            pos_data = torch.from_numpy(pos_data).float()
        pos_data = pos_data.cpu()  # Ensure CPU

        action_data = sample['action']
        if not isinstance(action_data, torch.Tensor):
            action_data = torch.from_numpy(action_data).float()
        action_data = action_data.cpu()  # Ensure CPU

        # Fill data
        padded_imgs[b, :T] = img_data
        padded_pos[b, :T] = pos_data
        padded_actions[b, :T] = action_data
        padding_masks[b, :T] = 1.0

    return {
        'obs': {
            'image': padded_imgs,
            'agent_pos': padded_pos
        },
        'action': padded_actions,
        'padding_mask': padding_masks,
        'chunk_length': torch.tensor([s['chunk_length'] for s in batch], dtype=torch.long, device='cpu')
    }