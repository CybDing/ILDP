import os
import pathlib
import copy
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from genesis_ILDP.model.discriminator.stage_discriminator import StageDiscriminator
from genesis_ILDP.dataset.base_dataset import BaseImageDataset
from genesis_ILDP.utils.seed_util import SeedManager


class TrainProgressPredictorWorkspace(BaseWorkspace):
    """
    Standalone workspace to train the StageDiscriminator / progress predictor.
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)

        self.device = torch.device(cfg.training.device)

        # unified seed control
        self.seed_manager = SeedManager(base_seed=cfg.training.seed)
        self.seed_manager.set_global_seed(deterministic=cfg.training.get('deterministic', False))
        # CPU generator for DataLoader; device generator unused here but kept for symmetry
        self.data_generator = self.seed_manager.get_generator(torch.device('cpu'))
        worker_seed_strategy = cfg.training.get('worker_seed_strategy', 'offset')
        self.worker_init_fn = lambda wid: self.seed_manager.worker_init_fn(wid, worker_seed_strategy)

        self.use_amp = cfg.training.get('use_amp', False)
        self.grad_clip_norm = cfg.training.get('grad_clip_norm', None)
        self.global_step = 0
        self.epoch = 0

        # build model + (optional) external vision encoder
        vision_encoder = None
        if cfg.get('vision_encoder') is not None:
            vision_encoder = hydra.utils.instantiate(cfg.vision_encoder)

        self.model: StageDiscriminator = hydra.utils.instantiate(
            cfg.model,
            vision_encoder=vision_encoder,
        )
        self.model.to(self.device)

        # optimizers (keep references on the model for BaseWorkspace checkpointing)
        vit_params = []
        if not self.model.freeze_encoder and hasattr(self.model.encoder, "parameters"):
            vit_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        self.vit_optimizer = hydra.utils.instantiate(cfg.optimizer.vit, params=vit_params) if vit_params else None
        self.mlp_optimizer = hydra.utils.instantiate(cfg.optimizer.mlp, params=self.model.output_mlp.parameters())

        self.model.vit_optimizer = self.vit_optimizer
        self.model.mlp_optimizer = self.mlp_optimizer

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _to_device(self, batch):
        def _move(x):
            if hasattr(x, "to"):
                return x.to(self.device, non_blocking=True)
            return x
        return {k: _move(v) for k, v in batch.items()}

    def _train_one_epoch(self, dataloader, cfg):
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, disable=cfg.training.get('tqdm_disable', False))
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["progress"].to(self.device, non_blocking=True).squeeze(-1)

            if self.vit_optimizer is not None:
                self.vit_optimizer.zero_grad(set_to_none=True)
            if self.mlp_optimizer is not None:
                self.mlp_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                preds = self.model(images)
                loss = F.mse_loss(preds, targets)

            if self.use_amp:
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.mlp_optimizer)
                    if self.vit_optimizer is not None:
                        self.scaler.unscale_(self.vit_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.scale(loss).backward()
                if self.vit_optimizer is not None:
                    self.scaler.step(self.vit_optimizer)
                self.scaler.step(self.mlp_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.vit_optimizer is not None:
                    self.vit_optimizer.step()
                self.mlp_optimizer.step()

            mae = torch.mean(torch.abs(preds.detach() - targets)).item()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_mae += mae * bs
            total_samples += bs
            self.global_step += 1

            if self.global_step % cfg.training.log_every == 0:
                pbar.set_description(f"step {self.global_step} | loss {loss.item():.4f} | mae {mae:.4f}")

        avg_loss = total_loss / max(total_samples, 1)
        avg_mae = total_mae / max(total_samples, 1)
        return {"loss": avg_loss, "mae": avg_mae}

    @torch.no_grad()
    def _validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["progress"].to(self.device, non_blocking=True).squeeze(-1)
            preds = self.model(images)
            loss = F.mse_loss(preds, targets)
            mae = torch.mean(torch.abs(preds - targets)).item()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_mae += mae * bs
            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        avg_mae = total_mae / max(total_samples, 1)
        return {"loss": avg_loss, "mae": avg_mae}

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume if configured
        if cfg.training.resume:
            ckpt_path = self.get_checkpoint_path()
            if ckpt_path.is_file():
                print(f"Resuming from checkpoint: {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        dataloader_kwargs = dict(cfg.dataloader)
        if dataloader_kwargs.get('shuffle', False):
            dataloader_kwargs['generator'] = self.data_generator
        dataloader = DataLoader(
            dataset,
            worker_init_fn=self.worker_init_fn if dataloader_kwargs.get('num_workers', 0) > 0 else None,
            **dataloader_kwargs
        )

        val_dataset = dataset.get_validation_dataset()
        val_kwargs = dict(cfg.val_dataloader)
        if val_kwargs.get('shuffle', False):
            val_kwargs['generator'] = self.data_generator
        val_dataloader = DataLoader(
            val_dataset,
            worker_init_fn=self.worker_init_fn if val_kwargs.get('num_workers', 0) > 0 else None,
            **val_kwargs
        )

        best_val = float("inf")

        for epoch in range(self.epoch, cfg.training.num_epochs):
            train_metrics = self._train_one_epoch(dataloader, cfg)
            val_metrics = self._validate(val_dataloader)

            print(
                f"[Epoch {epoch+1}/{cfg.training.num_epochs}] "
                f"train_loss={train_metrics['loss']:.4f}, train_mae={train_metrics['mae']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}, val_mae={val_metrics['mae']:.4f}"
            )

            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                ckpt_path = self.save_checkpoint(tag=f"epoch_{epoch+1}")
                print(f"Checkpoint saved: {ckpt_path}")

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                ckpt_path = self.save_checkpoint(tag="best")
                print(f"New best checkpoint saved: {ckpt_path}")

            self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config/train")),
    config_name="train_progress_predictor",
)
def main(cfg):
    workspace = TrainProgressPredictorWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
