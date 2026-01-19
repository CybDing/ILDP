from typing import Dict
import torch
import numpy as np
import copy
from genesis_ILDP.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer

from diffusion_policy.common.sampler import get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

from genesis_ILDP.dataset.base_dataset import BaseImageDataset


class SequenceChunkSampler:
    """
    Samples variable-length chunks from episodes.
    Strategy: Each epoch iterates through all episodes, sampling non-overlapping chunks
    of random length [min_length, max_length] to ensure ~80% coverage without excessive repetition.
    """
    def __init__(self, replay_buffer, min_chunk_length=2, max_chunk_length=16,
                 pad_before=1, pad_after=7, episode_mask=None):
        self.replay_buffer = replay_buffer
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.pad_before = pad_before
        self.pad_after = pad_after

        if episode_mask is None:
            self.episode_mask = np.ones(self.replay_buffer.n_episodes, dtype=bool)
        else:
            self.episode_mask = episode_mask

        self.episode_ends = self.replay_buffer.episode_ends[:]
        self.chunks = self._generate_chunks()

    def _generate_chunks(self):
        """Generate all chunks for one epoch"""
        chunks = []

        for episode_idx in range(self.replay_buffer.n_episodes):
            if not self.episode_mask[episode_idx]:
                continue

            start_idx = 0 if episode_idx == 0 else self.episode_ends[episode_idx - 1]
            end_idx = self.episode_ends[episode_idx] - 1
            episode_length = end_idx - start_idx + 1

            # Sample non-overlapping chunks within this episode
            cur_pos = 0
            while cur_pos < episode_length:
                remaining = episode_length - cur_pos
                chunk_length = min(
                    np.random.randint(self.min_chunk_length, self.max_chunk_length + 1),
                    remaining
                )

                if chunk_length < self.min_chunk_length:
                    break

                chunk_start = start_idx + cur_pos
                chunk_end = chunk_start + chunk_length - 1

                chunks.append({
                    'episode_idx': episode_idx,
                    'chunk_start': chunk_start,
                    'chunk_end': chunk_end,
                    'chunk_length': chunk_length
                })

                cur_pos += chunk_length

        return chunks

    def sample_chunk(self, idx):
        """Sample a specific chunk by index"""
        chunk_info = self.chunks[idx]
        chunk_start = chunk_info['chunk_start']
        chunk_end = chunk_info['chunk_end']
        chunk_length = chunk_info['chunk_length']

        result = {}
        for key in self.replay_buffer.keys():
            data = self.replay_buffer[key][chunk_start:chunk_end + 1]
            result[key] = data

        result['chunk_length'] = chunk_length
        return result

    def __len__(self):
        return len(self.chunks)

    def regenerate_chunks(self):
        """Regenerate chunks for a new epoch with different random sampling"""
        self.chunks = self._generate_chunks()


class PushTSequenceChunkDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            min_chunk_length=2,
            max_chunk_length=16,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            workspace_limits=None):

        super().__init__()
        self.workspace_limits = workspace_limits

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

        self.sampler = SequenceChunkSampler(
            replay_buffer=self.replay_buffer,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.zarr_path = zarr_path

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceChunkSampler(
            replay_buffer=self.replay_buffer,
            min_chunk_length=self.min_chunk_length,
            max_chunk_length=self.max_chunk_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', use_image_normalizer=True, **kwargs):
        from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

        normalizer = LinearNormalizer()

        if self.workspace_limits is not None:
            for key in ['action', 'agent_pos']:
                normalizer[key] = self._create_manual_normalizer(
                    self.workspace_limits, mode=mode, **kwargs)
        else:
            data = {
                'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state'][...,:2]
            }
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        if use_image_normalizer:
            normalizer['image'] = get_image_range_normalizer()

        return normalizer

    def _create_manual_normalizer(self, limits, mode='limits', output_max=1.0, output_min=-1.0, **kwargs):
        import torch
        from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

        x_min, x_max = limits['xlim']
        y_min, y_max = limits['ylim']

        if mode == 'limits':
            input_min = torch.tensor([x_min, y_min], dtype=torch.float32)
            input_max = torch.tensor([x_max, y_max], dtype=torch.float32)
            input_mean = (input_min + input_max) / 2
            input_std = (input_max - input_min) / 4

            input_range = input_max - input_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min

            input_stats_dict = {
                'min': input_min,
                'max': input_max,
                'mean': input_mean,
                'std': input_std
            }

            return SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=input_stats_dict)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented for manual normalizer")

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_chunk(idx)

        # Convert to torch tensors
        # Image is (T, H, W, C) from replay buffer, convert to (T, C, H, W) for PyTorch
        image = torch.from_numpy(sample['img']).float().permute(0, 3, 1, 2)  # (T, C, H, W)
        agent_pos = torch.from_numpy(sample['state'][:, :2]).float()  # (T, 2)
        action = torch.from_numpy(sample['action']).float()  # (T, action_dim)
        chunk_length = sample['chunk_length']

        obs_dict = {
            'image': image,
            'agent_pos': agent_pos
        }

        return {
            'obs': obs_dict,
            'action': action,
            'chunk_length': chunk_length
        }
