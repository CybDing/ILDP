#!/usr/bin/env python3
"""
Fine-tuning script for pretrained diffusion policy models.

This script loads a pretrained checkpoint and continues training with a new dataset.
It reuses the workspace training loop from train_action_diffusion_image_workspace.py
but replaces the dataset and normalizer.

Usage:
    python FT_DIFF_datasets.py --checkpoint path/to/model.ckpt --output_dir ./ft_output

Example:
    python FT_DIFF_datasets.py \
        --checkpoint ./outputs/best_model.ckpt \
        --output_dir ./outputs/finetuned_model
"""

import sys
import os
import pathlib
import argparse
import hydra
from omegaconf import OmegaConf
import torch
import dill
import copy

from genesis_ILDP.workspace.train_action_diffusion_image_workspace import TrainActionDiffusionImageWorkspace


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
    config_name="finetune_action_diffusion_pusht_image"  # Use fine-tuning config
)
def main(cfg):
    """
    Fine-tune a pretrained model with a new dataset.

    Key points for fine-tuning:
    1. Load pretrained checkpoint (model + EMA weights)
    2. Replace dataset with new fine-tuning dataset
    3. Update normalizer from new dataset
    4. Reset optimizer (or optionally resume optimizer state)
    5. Continue training with existing workspace.run() method
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint (.ckpt)')
    parser.add_argument('--output_dir', required=True, help='Output directory for fine-tuned model')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer state (recommended for fine-tuning)')
    parser.add_argument('--reset_epoch', action='store_true',
                       help='Reset epoch counter to 0 (recommended for fine-tuning)')
    args = parser.parse_args()

    # Ensure checkpoint exists
    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print("FINE-TUNING SETUP")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reset optimizer: {args.reset_optimizer}")
    print(f"Reset epoch: {args.reset_epoch}")
    print("=" * 80)

    # Load pretrained checkpoint
    device = torch.device(cfg.training.device)
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location=device)
    pretrained_cfg = payload['cfg']

    print(f"\nLoaded checkpoint from: {checkpoint_path}")
    print(f"Pretrained config target: {pretrained_cfg._target_}")

    # Create a new config for fine-tuning by merging pretrained config with new config
    # Priority: new cfg > pretrained_cfg (so we can override dataset, training params, etc.)
    ft_cfg = OmegaConf.merge(pretrained_cfg, cfg)

    # Force important overrides for fine-tuning
    ft_cfg._output_dir = args.output_dir
    ft_cfg.training.resume = False  # Don't auto-resume, we're manually loading

    # Optionally update the experiment name to indicate fine-tuning
    if 'exp_name' in ft_cfg:
        ft_cfg.exp_name = f"{ft_cfg.exp_name}_finetune"

    print(f"\nFine-tuning config:")
    print(f"  Dataset: {ft_cfg.task.dataset.zarr_path}")
    print(f"  Num epochs: {ft_cfg.training.num_epochs}")
    print(f"  Learning rate: {ft_cfg.optimizer.lr}")
    print(f"  Batch size: {ft_cfg.training.batch_size}")

    # Initialize workspace with merged config
    workspace = TrainActionDiffusionImageWorkspace(ft_cfg, output_dir=args.output_dir)

    # Load pretrained weights
    # IMPORTANT: Control what gets loaded from checkpoint
    exclude_keys = []
    include_keys = []

    if args.reset_optimizer:
        # Don't load optimizer state - workspace will use fresh optimizer from ft_cfg
        exclude_keys.append('optimizer')
        print("\nResetting optimizer state (workspace will create fresh optimizer from config)")
    else:
        # Load optimizer state from checkpoint (useful for continuing training)
        print("\nLoading optimizer state from checkpoint")

    if args.reset_epoch:
        # Don't load epoch/global_step - start from 0
        print("\nResetting epoch counter to 0")
    else:
        # Load epoch and global_step from checkpoint
        include_keys.extend(['global_step', 'epoch'])
        epoch_num = payload.get('pickles', {}).get('epoch', 0) if 'pickles' in payload else 0
        print(f"\nResuming from epoch {epoch_num}")

    # Load the pretrained payload
    workspace.load_payload(
        payload,
        exclude_keys=exclude_keys,
        include_keys=include_keys if include_keys else None
    )

    # If we reset optimizer, we need to recreate it with new config settings
    if args.reset_optimizer:
        print(f"Creating new optimizer with lr={ft_cfg.optimizer.lr}")
        workspace.optimizer = hydra.utils.instantiate(
            ft_cfg.optimizer, params=workspace.model.parameters()
        )

    print(f"\nLoaded pretrained model weights:")
    print(f"  Model: {type(workspace.model).__name__}")
    if workspace.ema_model is not None:
        print(f"  EMA Model: {type(workspace.ema_model).__name__}")

    # CRITICAL: Update normalizer from new dataset
    # The workspace.run() method will load the dataset and update normalizer automatically
    # But we can verify the dataset path here
    print(f"\nDataset will be loaded from: {ft_cfg.task.dataset.zarr_path}")
    print("Note: Normalizer will be updated from new dataset in workspace.run()")

    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80 + "\n")

    # Run fine-tuning using the existing workspace training loop
    # This will:
    # 1. Load the new dataset from cfg.task.dataset
    # 2. Extract normalizer from new dataset
    # 3. Update model.set_normalizer() and ema_model.set_normalizer()
    # 4. Continue training
    workspace.run()

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETED")
    print("=" * 80)
    print(f"Fine-tuned model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()