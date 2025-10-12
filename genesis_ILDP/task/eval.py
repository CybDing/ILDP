"""

legacy version model eval

python eval.py --checkpoint ./data/checkpoints.ckpt -o ../data/pusht_eval_output -d mps:0

Example with custom runner parameters:
python eval.py --checkpoint ./data/checkpoints.ckpt -o ../data/pusht_eval_output -d mps:0 \
    --n_test 20 --n_test_vis 5 --n_train 10 --n_train_vis 3 \
    --test_start_seed 50000 --train_start_seed 1000 --max_steps 300 --fps 60 --enable_render

Quick test with good video speed:
python eval.py --checkpoint ./data/checkpoints.ckpt -o ../data/pusht_eval_output -d mps:0 \
    --n_test 1 --n_test_vis 1 --max_steps 200 --fps 30
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import numpy as np
from genesis_ILDP.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_test', default=None, type=int, help='Number of test episodes (overrides config)')
@click.option('--n_test_vis', default=None, type=int, help='Number of test episodes with visualization (overrides config)')
@click.option('--n_train', default=0, type=int, help='Number of train episodes (overrides config)')
@click.option('--n_train_vis', default=0, type=int, help='Number of train episodes with visualization (overrides config)')
@click.option('--test_start_seed', default=50023, type=int, help='Starting seed for test episodes (overrides config)')
@click.option('--train_start_seed', default=0, type=int, help='Starting seed for train episodes (overrides config)')
@click.option('--max_steps', default=600, type=int, help='Maximum steps per episode (overrides config)')
@click.option('--fps', default=30, type=int, help='Video FPS for rendering (overrides config)')
@click.option('--enable_render/--no_render', default=None, help='Enable/disable rendering (overrides config)')
def main(checkpoint, output_dir, device, n_test, n_test_vis, n_train, n_train_vis, test_start_seed, train_start_seed, max_steps, fps, enable_render):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint with device mapping for cross-platform compatibility
    target_device = torch.device(device)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill, map_location=target_device)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    # Device is already set as target_device above
    policy.to(target_device)
    policy.eval()

    # Configure env_runner parameters (override config with command line args if provided)
    runner_params = {}
    if n_test is not None:
        runner_params['n_test'] = n_test
        print(f"Overriding n_test: {n_test}")
    if n_test_vis is not None:
        runner_params['n_test_vis'] = n_test_vis
        print(f"Overriding n_test_vis: {n_test_vis}")
    if n_train is not None:
        runner_params['n_train'] = n_train
        print(f"Overriding n_train: {n_train}")
    if n_train_vis is not None:
        runner_params['n_train_vis'] = n_train_vis
        print(f"Overriding n_train_vis: {n_train_vis}")
    if test_start_seed is not None:
        runner_params['test_start_seed'] = test_start_seed
        print(f"Overriding test_start_seed: {test_start_seed}")
    if train_start_seed is not None:
        runner_params['train_start_seed'] = train_start_seed
        print(f"Overriding train_start_seed: {train_start_seed}")
    if max_steps is not None:
        runner_params['max_steps'] = max_steps
        print(f"Overriding max_steps: {max_steps}")
    if fps is not None:
        runner_params['fps'] = fps
        print(f"Overriding fps: {fps}")
    if enable_render is not None:
        runner_params['enable_render'] = enable_render
        print(f"Overriding enable_render: {enable_render}")

    # Always override output_dir
    runner_params['output_dir'] = output_dir

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        **runner_params)
    
    runner_log = env_runner.run(policy)

    # Convert runner_log to JSON-serializable format
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

    # Save results to JSON
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    print(f"Results saved to: {out_path}")

if __name__ == '__main__':
    main()
