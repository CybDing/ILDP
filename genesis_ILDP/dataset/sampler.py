from diffusion_policy.common.replay_buffer import ReplayBuffer
import numpy as np


# def get_val_mask(n_episodes, val_ratio, seed=0) -> np.ndarray:
#     val_mask = np.zeros(n_episodes, dtype=bool)
#     if val_ratio <= 0:
#         return val_mask

#     # have at least 1 episode for validation, and at least 1 episode for train
#     n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
#     rng = np.random.default_rng(seed=seed)
#     val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
#     val_mask[val_idxs] = True
#     return val_mask
from diffusion_policy.diffusion_policy.common.sampler import get_val_mask, downsample_mask

def create_indices(episode_ends, sequence_length, episode_mask, pad_before, pad_after):
    indices = []
    for episode_idx in range(len(episode_ends)):
        if episode_mask[episode_idx] == 0: continue;
        else:
            if episode_idx == 0: start_idx = 0
            else: start_idx = episode_ends[episode_idx - 1] +  1
            end_idx = episode_ends[episode_idx]

        seq_counts = end_idx - start_idx + pad_before + pad_after - sequence_length + 1
        if seq_counts <= 0: continue
        cur_start_idx = -pad_before
        for _ in range(seq_counts):
            if cur_start_idx < 0: 
                sample_start_idx = -cur_start_idx
                buffer_start_idx = start_idx
            else:
                sample_start_idx = 0
                buffer_start_idx = start_idx + cur_start_idx
            
            cur_end_idx = cur_start_idx + sequence_length - 1
            if cur_end_idx > end_idx:
                sample_end_idx = sequence_length - (cur_end_idx - end_idx) - 1
                buffer_end_idx = end_idx
            else:
                sample_end_idx = sequence_length - 1
                buffer_end_idx = buffer_start_idx + sample_end_idx - sample_start_idx

            indices.append((buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx))

    return indices


class SequenceSampler:
    def __init__(self, zarr_path=None, replay_buffer=None, horizon=16, pad_before=1, pad_after=7, 
                is_padding=True, episode_mask=None):
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        elif zarr_path is not None:
            self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path) # Class Method for initializing the class
        else:
            raise ValueError("Either zarr_path or replay_buffer must be provided")
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        if episode_mask is None:
            self.episode_mask = np.ones(len(episode_mask), dtype=bool)
        else: self.episode_mask = episode_mask
        self.is_padding = is_padding
        self.indices = self.indices_gen()

    def indices_gen(self):
        episode_ends = self.replay_buffer.episode_ends[:]

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=self.horizon,
            episode_mask=self.episode_mask,
            pad_before=self.pad_before,
            pad_after=self.pad_after
        )
        return indices

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        result = dict()
        for key in self.replay_buffer.keys():
            input_arr = self.replay_buffer[key]
            sample = input_arr[buffer_start_idx : buffer_end_idx + 1]

            # Handle padding if needed
            if (sample_start_idx > 0) or (sample_end_idx < self.horizon):
                data = np.zeros((self.horizon,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]  
                if sample_end_idx < self.horizon:
                    data[sample_end_idx:] = sample[-1]  
                data[sample_start_idx:sample_end_idx + 1] = sample
            else:
                data = sample

            result[key] = data
        return result

    def __len__(self):
        return len(self.indices)
