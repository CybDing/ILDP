# if __name__ == "__main__":
#     ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
#     sys.path.append(ROOT_DIR)
#     os.chdir(ROOT_DIR)

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
from genesis_ILDP.policy.pretrain.action_diffusion_image_policy import ActionDiffusionImagePolicy
from genesis_ILDP.model.discriminator.stage_discriminator import StageDiscriminator
from genesis_ILDP.dataset.base_dataset import BaseImageDataset
from genesis_ILDP.env_runner.base_image_runner import BaseImageRunner
from genesis_ILDP.utils.pytorch_util import dict_apply, optimizer_to
from genesis_ILDP.utils.seed_util import SeedManager

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from genesis_ILDP.model.common.ema_model import EMAModel
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

        device = torch.device(cfg.training.device)
        seed = cfg.training.seed
        deterministic = bool(cfg.training.get('deterministic', False))
        worker_seed_strategy = cfg.training.get('worker_seed_strategy', 'offset')

        self.seed_manager = SeedManager(base_seed=seed)
        self.seed_manager.set_global_seed(deterministic=deterministic)
        # Separate generators: CPU for DataLoader shuffling, device for sampling
        self.data_generator = self.seed_manager.get_generator(torch.device('cpu'))
        self.sample_generator = self.seed_manager.get_generator(device)

        self.use_generator = cfg.training.get('use_generator', True)
        self.worker_init_fn = lambda worker_id: self.seed_manager.worker_init_fn(worker_id, worker_seed_strategy)

        self.model: ActionDiffusionImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ActionDiffusionImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Optional progress predictor (pretrained or to be fine-tuned separately)
        self.progress_predictor: StageDiscriminator | None = None
        pp_cfg = cfg.get('progress_predictor', None)
        if pp_cfg is not None and pp_cfg.get('enable', False):
            self.progress_predictor = hydra.utils.instantiate(pp_cfg.model)
            if pp_cfg.get('ckpt_path', None):
                state = torch.load(pp_cfg.ckpt_path, map_location='cpu')
                # allow both full payload and bare state_dict
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                self.progress_predictor.load_state_dict(state)
                print(f"Loaded progress predictor weights from {pp_cfg.ckpt_path}")
            self.progress_predictor.to(device)
            if pp_cfg.get('freeze', True):
                for p in self.progress_predictor.parameters():
                    p.requires_grad = False
            print("Progress predictor initialized for runner.")

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

        print(f"ActionDiffusionImagePolicy workspace initialized:")
        print(f"- Model: {type(self.model).__name__}")
        print(f"- Device: {cfg.training.device}")
        print(f"- Use EMA: {cfg.training.use_ema}")
        print(f"- Seed: {seed} ({'deterministic' if deterministic else 'non-deterministic'})")

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

        # Handle DataLoader configuration
        dataloader_kwargs = dict(cfg.dataloader)
        val_dataloader_kwargs = dict(cfg.val_dataloader)

        device = torch.device(cfg.training.device)

        def to_device_safe(x):
            if hasattr(x, 'to'):
                if device.type == 'mps' and getattr(x, 'dtype', None) == torch.float64:
                    x = x.to(torch.float32)
                return x.to(device, non_blocking=True)
            return x
        
        # DataLoader: use CPU generator when shuffling to avoid device mismatch
        if dataloader_kwargs.get('shuffle', False):
            dataloader_kwargs['generator'] = self.data_generator

        train_dataloader = DataLoader(
            dataset,
            worker_init_fn=self.worker_init_fn if dataloader_kwargs.get('num_workers', 0) > 0 else None,
            **dataloader_kwargs
        )

        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()

        # Validation DataLoader (same as train)
        if val_dataloader_kwargs.get('shuffle', False) and self.use_generator:
            val_dataloader_kwargs['generator'] = self.data_generator

        val_dataloader = DataLoader(
            val_dataset,
            worker_init_fn=self.worker_init_fn if val_dataloader_kwargs.get('num_workers', 0) > 0 else None,
            **val_dataloader_kwargs
        )

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

                # attach optional progress predictor for early-stop evaluation
                if hasattr(env_runner, 'progress_predictor'):
                    env_runner.progress_predictor = self.progress_predictor
                    env_runner.progress_threshold = cfg.task.env_runner.get('progress_threshold', env_runner.progress_threshold)
                    env_runner.progress_min_steps = cfg.task.env_runner.get('progress_min_steps', env_runner.progress_min_steps)

                # Try to align environment seeds with training seed for reproducibility
                if hasattr(env_runner, "_update_seeds"):
                    env_runner._update_seeds(
                        seed_train=cfg.task.env_runner.get('train_start_seed', None),
                        seed_test=cfg.task.env_runner.get('test_start_seed', None),
                    )
                elif hasattr(env_runner, "seed_train") and hasattr(env_runner, "seed_test"):
                    env_runner.seed_train = cfg.task.env_runner.get('train_start_seed', env_runner.seed_train)
                    env_runner.seed_test = cfg.task.env_runner.get('test_start_seed', env_runner.seed_test)

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

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

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
                        batch = dict_apply(batch, to_device_safe)
                        
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
                        
                        if cfg.training.use_ema:
                            ema.step(self.model)

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

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if env_runner is not None and (self.epoch % cfg.training.rollout_every) == 0:
                    try:
                        print(f"Running rollout evaluation at epoch {self.epoch}")

                        eval_generator = self.sample_generator if self.use_generator else None

                        runner_log = env_runner.run(policy, generator=eval_generator, wandb_run=wandb_run)
                        # runner_log is JSON-safe (no videos), merge into step_log
                        step_log.update(runner_log)
                        print(f"Rollout completed: {list(runner_log.keys())}")
                    except Exception as e:
                        print(f"Warning: Rollout evaluation failed: {e}")
                        # Provide a default test_mean_score to prevent KeyError
                        if 'test_mean_score' not in step_log:
                            step_log['test_mean_score'] = 0.0
                            print("Added default test_mean_score=0.0 due to rollout failure")

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, to_device_safe)
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

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        try:
                            batch = dict_apply(train_sampling_batch, lambda x: to_device_safe(x))
                            obs_dict = batch['obs']
                            gt_action = batch['action']

                            print(f"Full batch shapes - obs image: {obs_dict['image'].shape}, obs agent_pos: {obs_dict['agent_pos'].shape}, gt_action: {gt_action.shape}")

                            n_obs_steps = getattr(self.model, 'n_obs_steps', 2)

                            obs_dict_sliced = {
                                'image': obs_dict['image'][:, :n_obs_steps],
                                'agent_pos': obs_dict['agent_pos'][:, :n_obs_steps]
                            }

                            sample_generator = self.sample_generator if self.use_generator else None

                            result = policy.predict_action(obs_dict_sliced, generator=sample_generator)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action[:, :, :2], gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            print(f"Sample MSE: {mse.item():.4f}")
                        except Exception as e:
                            print(f"Warning: Sampling evaluation failed: {e}")
                            import traceback
                            traceback.print_exc()  # Add this for better debugging
                
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
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config/train")),
    config_name="train_action_diffusion_pusht_image")
def main(cfg):
    workspace = TrainActionDiffusionImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
