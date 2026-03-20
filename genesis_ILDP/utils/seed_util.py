"""Unified seed management utility for reproducible experiments."""
import random
import numpy as np
import torch
from typing import Optional


class SeedManager:
    """Centralized seed management for all random number generators."""

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self._generators = dict()  # device -> torch.Generator

    def set_global_seed(self, deterministic: bool = False):
        """Set all global RNG seeds."""
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(self.base_seed)

        if deterministic:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        else:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(False)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True

    def get_generator(self, device: Optional[torch.device] = None) -> torch.Generator:
        """
        Get or create a torch Generator for the specified device.
        Keeps one generator per device to avoid device-mismatch errors.
        """
        if device is None:
            device = torch.device('cpu')
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        key = (device.type, device.index)
        if key not in self._generators:
            try:
                gen = torch.Generator(device=device)
            except TypeError:
                # Some backends only allow 'cpu' or 'cuda'; fall back gracefully
                gen = torch.Generator(device=device.type if device.type in ("cuda", "mps") else "cpu")
            gen.manual_seed(self.base_seed)
            self._generators[key] = gen
        return self._generators[key]

    def worker_init_fn(self, worker_id: int, strategy: str = 'offset'):
        """DataLoader worker initialization function."""
        if strategy == 'offset':
            worker_seed = self.base_seed + worker_id
        elif strategy == 'independent':
            worker_seed = self.base_seed
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def get_env_seeds(self, n_envs: int, offset: int = 0) -> np.ndarray:
        """Generate environment seeds."""
        return np.arange(self.base_seed + offset, self.base_seed + offset + n_envs)

    def get_train_test_seeds(self, n_train: int, n_test: int,
                            train_offset: int = 0, test_offset: int = 10000) -> tuple:
        """Generate train and test environment seeds."""
        train_seeds = self.get_env_seeds(n_train, train_offset)
        test_seeds = self.get_env_seeds(n_test, test_offset)
        return train_seeds, test_seeds
