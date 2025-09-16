if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

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

from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from genesis_ILDP.policy.action_diffusion_image_policy import ActionDiffusionImagePolicy
from genesis_ILDP.dataset.base_dataset import BaseImageDataset
from genesis_ILDP.env_runner.base_image_runner import BaseImageRunner
from genesis_ILDP.utils.pytorch_util import dict_apply, optimizer_to

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from genesis_ILDP.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainActionDiffusionImageWorkspace(BaseWorkspace):
    """
    Training workspace specifically for ActionDiffusionImagePolicy.
    Uses only genesis_ILDP components with minimal diffusion_policy dependencies.
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ActionDiffusionImagePolicy = hydra.utils.instantiate(cfg.policy)
        
        self.ema_model: ActionDiffusionImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        print(f"ActionDiffusionImagePolicy workspace initialized:")
        print(f"- Model type: {type(self.model).__name__}")
        print(f"- Device: {cfg.training.device}")
        print(f"- Use EMA: {cfg.training.use_ema}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset), f"Expected BaseImageDataset, got {type(dataset)}"

        # Handle MPS device compatibility for DataLoader
        dataloader_kwargs = dict(cfg.dataloader)
        val_dataloader_kwargs = dict(cfg.val_dataloader)

        device = torch.device(cfg.training.device)
        seed = cfg.training.seed
        if device.type == 'mps' and dataloader_kwargs.get('shuffle', False):
            # MPS has compatibility issues with DataLoader shuffling
            # Disable shuffling for MPS devices to avoid generator conflicts
            dataloader_kwargs['shuffle'] = False
            print("Warning: Disabled DataLoader shuffle for MPS device compatibility")

        train_dataloader = DataLoader(dataset, **dataloader_kwargs)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dataloader_kwargs)

        # *** IMPORTANT: Set normalizer on model ***
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner (optional, for rollout evaluation)
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

        # configure logging
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
                    "model_type": "ActionDiffusionImagePolicy"
                }
            )
        else:
            print("Wandb logging disabled")

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Set generator device for MPS compatibility
        if device.type == 'mps':
            # For MPS, we need to ensure the generator is on the correct device
            # But PyTorch DataLoader samplers don't support MPS generators yet
            # So we'll need to handle this differently
            print("Warning: MPS device detected. DataLoader may have compatibility issues.")
            print("Consider using device='cpu' if you encounter random sampling errors.")
            print("Note: MPS has known issues with certain convolution backward passes.")
            print("If you encounter 'view size is not compatible' errors, try using device='cpu'.")

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        print(f"Starting training for {cfg.training.num_epochs} epochs...")

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss using ActionDiffusionImagePolicy
                        try:
                            # Enable anomaly detection to pinpoint the exact .view() call
                            torch.autograd.set_detect_anomaly(True)
                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()
                        except RuntimeError as e:
                            if "view size is not compatible" in str(e):
                                print(f"RuntimeError during backward pass: {e}")
                                print("This suggests a .view() call on a non-contiguous tensor in the compute_loss method")
                                print("Check ActionDiffusionImagePolicy.compute_loss() and its called methods")
                                # Print batch shapes for debugging
                                for key, value in batch.items():
                                    if hasattr(value, 'shape'):
                                        print(f"batch['{key}'].shape: {value.shape}")
                                        print(f"batch['{key}'].is_contiguous(): {value.is_contiguous()}")
                                raise
                            else:
                                raise

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout (only if env_runner is configured)
                if env_runner is not None and (self.epoch % cfg.training.rollout_every) == 0:
                    try:
                        print(f"Running rollout evaluation at epoch {self.epoch}")
                        runner_log = env_runner.run(policy)
                        # log all rollout metrics
                        step_log.update(runner_log)
                        print(f"Rollout completed: {list(runner_log.keys())}")
                    except Exception as e:
                        print(f"Warning: Rollout evaluation failed: {e}")

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss.item())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = np.mean(val_losses)
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            print(f"Validation loss: {val_loss:.4f}")

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        try:
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            gt_action = batch['action']
                            
                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            print(f"Sample MSE: {mse.item():.4f}")
                            
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                        except Exception as e:
                            print(f"Warning: Sampling evaluation failed: {e}")
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                        print(f"Saved top-k checkpoint: {topk_ckpt_path}")

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        print(f"Training completed after {cfg.training.num_epochs} epochs!")
        if wandb_run is not None:
            wandb_run.finish()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_action_diffusion_pusht_image")
def main(cfg):
    workspace = TrainActionDiffusionImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()