import sys
import pathlib
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from typing import Optional

from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from genesis_ILDP.policy.action_diffusion_memory_policy import ActionDiffusionMemoryPolicy
from genesis_ILDP.dataset.base_dataset import BaseImageDataset
from genesis_ILDP.env_runner.base_image_runner import BaseImageRunner
from genesis_ILDP.utils.pytorch_util import dict_apply, optimizer_to
from genesis_ILDP.utils.seed_util import SeedManager
from genesis_ILDP.dataset.smart_batch_sampler import SmartBatchSampler, pad_sequence_batch

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from genesis_ILDP.model.common.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainMemoryDiffusionWorkspace(BaseWorkspace):
    """
    Training workspace for ActionDiffusionMemoryPolicy with MemoryNet.
    Supports variable-length sequence chunks for memory-based training.
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        device = torch.device(cfg.training.device)
        seed = cfg.training.seed
        deterministic = bool(cfg.training.get('deterministic', False))
        worker_seed_strategy = cfg.training.get('worker_seed_strategy', 'offset')

        self.seed_manager = SeedManager(base_seed=seed)
        self.seed_manager.set_global_seed(deterministic=deterministic)

        # Model generator for inference (on device)
        self.generator = self.seed_manager.get_generator(device)

        self.use_generator = cfg.training.get('use_generator', True)
        self.worker_init_fn = lambda worker_id: self.seed_manager.worker_init_fn(worker_id, worker_seed_strategy)

        self.model: ActionDiffusionMemoryPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ActionDiffusionMemoryPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

        print(f"ActionDiffusionMemoryPolicy workspace initialized:")
        print(f"- Model: {type(self.model).__name__}")
        print(f"- Device: {cfg.training.device}")
        print(f"- Use EMA: {cfg.training.use_ema}")
        print(f"- Seed: {seed} ({'deterministic' if deterministic else 'non-deterministic'})")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset), f"Expected BaseImageDataset, got {type(dataset)}"

        dataloader_kwargs = dict(cfg.dataloader)
        val_dataloader_kwargs = dict(cfg.val_dataloader)

        device = torch.device(cfg.training.device)

        def to_device_safe(x):
            if hasattr(x, 'to'):
                if device.type == 'mps' and getattr(x, 'dtype', None) == torch.float64:
                    x = x.to(torch.float32)
                return x.to(device, non_blocking=True)
            return x

        # Use SmartBatchSampler for efficient length-based batching
        # This groups similar-length sequences together to minimize padding waste
        batch_size = dataloader_kwargs.pop('batch_size', 8)
        dataloader_kwargs.pop('shuffle', None)  # Not needed with batch sampler
        val_dataloader_kwargs.pop('shuffle', None)
        
        train_batch_sampler = SmartBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle_same_length=True
        )
        
        train_dataloader = DataLoader(
            dataset,
            batch_sampler=train_batch_sampler,
            worker_init_fn=self.worker_init_fn if dataloader_kwargs.get('num_workers', 0) > 0 else None,
            collate_fn=pad_sequence_batch,  # Pads to max length in batch
            **dataloader_kwargs
        )

        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()

        # Val dataloader also uses batching for efficiency
        val_batch_size = val_dataloader_kwargs.pop('batch_size', batch_size)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            worker_init_fn=self.worker_init_fn if val_dataloader_kwargs.get('num_workers', 0) > 0 else None,
            collate_fn=pad_sequence_batch,
            **val_dataloader_kwargs
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner: BaseImageRunner = None
        if 'env_runner' in cfg.task:
            try:
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=self.output_dir)
                assert isinstance(env_runner, BaseImageRunner)
                print(f"Environment runner configured: {type(env_runner).__name__}")
            except Exception as e:
                print(f"Warning: Could not initialize env_runner: {e}")
                print("Training will continue without environment evaluation")

        wandb_run = None
        if cfg.logging.mode != 'disabled':
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                    "model_type": "ActionDiffusionMemoryPolicy"
                }
            )
        else:
            print("Wandb logging disabled")

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        print(f"Starting training for {cfg.training.num_epochs} epochs...")

        # Training loop
        for epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()

            # Regenerate chunks for new epoch (different random sampling)
            if hasattr(dataset, 'sampler') and hasattr(dataset.sampler, 'regenerate_chunks'):
                dataset.sampler.regenerate_chunks()

            # Train
            self.model.train()
            train_losses = []

            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                          leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                for batch_idx, batch in enumerate(tepoch):
                    # Batch is now a dict with padded tensors: (B, T, ...)
                    # Convert to device
                    batch = dict_apply(batch, to_device_safe)

                    # Compute loss for the entire batch
                    loss = self.model.compute_loss(batch)
                    loss = loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    train_losses.append(loss.item() * cfg.training.gradient_accumulate_every)

                    # Gradient accumulation step
                    if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            cfg.training.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        # EMA update
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        self.global_step += 1

                    # Logging
                    tepoch.set_postfix(loss=train_losses[-1], lr=self.optimizer.param_groups[0]['lr'])

                    if cfg.training.max_train_steps is not None:
                        if batch_idx >= (cfg.training.max_train_steps - 1):
                            break

            # Average train loss
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # Validation
            if (epoch_idx % cfg.training.val_every) == 0:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                  leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                        for batch_idx, batch in enumerate(tepoch):
                            # Batch is now a dict with padded tensors
                            batch = dict_apply(batch, to_device_safe)
                            loss = self.model.compute_loss(batch)
                            val_losses.append(loss.item())
                            tepoch.set_postfix(loss=loss.item())

                            if cfg.training.max_val_steps is not None:
                                if batch_idx >= (cfg.training.max_val_steps - 1):
                                    break

                val_loss = np.mean(val_losses)
                step_log['val_loss'] = val_loss

            # Rollout evaluation
            if (epoch_idx % cfg.training.rollout_every) == 0 and env_runner is not None:
                policy_to_eval = self.model
                if cfg.training.use_ema:
                    policy_to_eval = self.ema_model

                policy_to_eval.eval()

                with torch.no_grad():
                    runner_log = env_runner.run(policy_to_eval, wandb_run=wandb_run)

                step_log.update(runner_log)

            # Checkpointing
            if (epoch_idx % cfg.training.checkpoint_every) == 0:
                checkpoint_path = pathlib.Path(self.output_dir) / 'checkpoints' / f'epoch_{epoch_idx}.ckpt'
                self.save_checkpoint(path=checkpoint_path)

                # TopK checkpointing
                if 'val_loss' in step_log:
                    topk_manager.save_topk(
                        self.save_checkpoint,
                        step_log['val_loss'],
                        epoch_idx
                    )

            # Logging
            if wandb_run is not None:
                wandb_run.log(step_log, step=self.global_step)

            self.epoch += 1

        if wandb_run is not None:
            wandb_run.finish()

        print("Training completed!")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg: OmegaConf):
    workspace = TrainMemoryDiffusionWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
