"""
Basic evaluation:
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt

With custom eval config:
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt eval=custom_eval

Override specific parameters:
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt \
        eval.device=cuda:1 \
        eval.env_runner.n_test=50 \
        eval.env_runner.max_steps=500

"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import hydra
import torch
import dill
import wandb
import json
import numpy as np
from omegaconf import OmegaConf, DictConfig
from genesis_ILDP.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path="../config/eval",
    config_name="eval_checkpoint_mps"
)
# could use -cn for overriding the config name
def main(eval_cfg: DictConfig) -> None:
    # Override Hydra's working directory to stay in original dir
    os.chdir(hydra.utils.get_original_cwd())
    """
    Main evaluation function using Hydra configuration.

    Args:
        eval_cfg: Hydra configuration from config/eval/default.yaml
    """
    # Extract checkpoint path
    checkpoint_path = eval_cfg.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Setup output directory
    output_dir = eval_cfg.output_dir
    if output_dir is None:
        # Use Hydra's automatic output dir
        from hydra.core.hydra_config import HydraConfig
        output_dir = HydraConfig.get().runtime.output_dir

    if os.path.exists(output_dir) and eval_cfg.get('confirm_overwrite', False):
        import click
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load checkpoint with device mapping
    target_device = torch.device(eval_cfg.device)
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Target device: {target_device}")

    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location=target_device)
    checkpoint_cfg = payload['cfg']

    # Merge checkpoint config with eval overrides
    # Priority: eval_cfg.overrides > checkpoint_cfg
    if 'overrides' in eval_cfg and eval_cfg.overrides is not None:
        print("\n=== Applying config overrides ===")

        OmegaConf.set_struct(checkpoint_cfg, False) # enable hydra adding new keys from override
        checkpoint_cfg = OmegaConf.merge(checkpoint_cfg, eval_cfg.overrides)
        OmegaConf.set_struct(checkpoint_cfg, True)

        print(OmegaConf.to_yaml(eval_cfg.overrides))

    # Initialize workspace
    cls = hydra.utils.get_class(checkpoint_cfg._target_)
    workspace = cls(checkpoint_cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Get policy from workspace
    policy = workspace.model
    if eval_cfg.use_ema and hasattr(workspace, 'ema_model'):
        print("Using EMA model")
        policy = workspace.ema_model

    policy.to(target_device)
    policy.eval()

    if 'env_runner' in eval_cfg and eval_cfg.env_runner is not None:
        print("\n=== Using custom env_runner from eval config ===")
        runner_config = eval_cfg.env_runner
    else:
        print("\n=== Using env_runner from checkpoint config ===")
        runner_config = checkpoint_cfg.task.env_runner

    runner_config = OmegaConf.merge(runner_config, {'output_dir': output_dir})


    print("\n=== Env Runner Configuration ===")
    print(OmegaConf.to_yaml(runner_config))

    env_runner = hydra.utils.instantiate(runner_config)
    print("\n=== Starting evaluation ===")
    runner_log = env_runner.run(policy)

    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif hasattr(value, 'item'):
            json_log[key] = float(value.item())
        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            json_log[key] = float(value)
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    print(f"\n=== Evaluation completed ===")
    print(f"Results saved to: {out_path}")
    print(f"\nKey metrics:")
    for key, value in json_log.items():
        if 'mean_score' in key or 'success_rate' in key:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
