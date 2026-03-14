#!/usr/bin/env python3
"""
Pre-training entry point for Diffusion Policy.

This script launches pre-training using TrainActionDiffusionImageWorkspace,
which handles dataset loading, normalizer setup, training loop, checkpointing,
and optional environment rollout evaluation.

Usage:
    python pretrain_dp.py --config-name train_action_diffusion_pusht_image

    Override parameters via Hydra CLI:
    python pretrain_dp.py training.num_epochs=100 training.batch_size=64

See config/train/ for available training configurations.
The core training logic is implemented in:
    genesis_ILDP/workspace/train_action_diffusion_image_workspace.py
"""

import sys
import pathlib
import os
import hydra
from omegaconf import OmegaConf

from genesis_ILDP.workspace.train_action_diffusion_image_workspace import TrainActionDiffusionImageWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config/train")),
    config_name="train_action_diffusion_pusht_image"
)
def main(cfg):
    workspace = TrainActionDiffusionImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
