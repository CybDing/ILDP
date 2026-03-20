from typing import Dict, List, Tuple, Optional
import copy

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from genesis_ILDP.dataset.base_dataset import BaseImageDataset
from genesis_ILDP.dataset.sampler import get_val_mask, downsample_mask


class ProgressImageDataset(BaseImageDataset):
    """
    Frame-level dataset that returns (image, progress) pairs.

    Progress label = step_idx / (episode_len - 1) so that the first frame is 0
    and the last frame is 1. If an episode has only one frame, progress is 0.
    """
    def __init__(
        self,
        zarr_path: str,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
        use_image_normalizer: bool = True,
        episode_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['img'])
        self.zarr_path = zarr_path
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.use_image_normalizer = use_image_normalizer

        if episode_mask is None:
            val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed,
            )
        else:
            train_mask = episode_mask
            val_mask = ~episode_mask

        self.train_mask = train_mask
        self.indices = self._build_indices(train_mask)

    def _build_indices(self, episode_mask: np.ndarray) -> List[Tuple[int, int, int, float]]:
        """
        Returns a list of tuples:
        (global_step_idx, episode_len, step_idx, progress_label)
        """
        indices = []
        episode_ends = self.replay_buffer.episode_ends[:]
        start = 0
        for ep_idx, end in enumerate(episode_ends):
            ep_len = end - start
            if episode_mask[ep_idx]:
                denom = max(ep_len - 1, 1)
                for step_idx in range(ep_len):
                    global_idx = start + step_idx
                    progress = step_idx / denom
                    indices.append((global_idx, ep_len, step_idx, progress))
            start = end
        return indices

    def get_validation_dataset(self) -> "ProgressImageDataset":
        val_dataset = copy.copy(self)
        val_dataset.indices = val_dataset._build_indices(~self.train_mask)
        val_dataset.train_mask = ~self.train_mask
        return val_dataset

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        if self.use_image_normalizer:
            normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx, episode_len, step_idx, progress = self.indices[idx]
        img = self.replay_buffer['img'][global_idx]
        image = np.moveaxis(img, -1, 0) / 255.0  # (C, H, W) in [0,1]

        data = {
            "image": torch.from_numpy(image).float(),
            "progress": torch.tensor([progress], dtype=torch.float32),
            "meta": {
                "episode_len": torch.tensor(episode_len, dtype=torch.int32),
                "step_idx": torch.tensor(step_idx, dtype=torch.int32),
            },
        }
        return data
