"""
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

def analyze_evaluation_results(runner_log):
    """
    Comprehensive statistical analysis of evaluation results
    """
    analysis = {}

    # Helper function to safely process rewards
    def process_rewards(rewards, prefix):
        if not rewards or (isinstance(rewards, (list, np.ndarray)) and len(rewards) == 0):
            return  # Skip if no rewards

        if isinstance(rewards, (list, np.ndarray)) and len(rewards) > 1:
            rewards = np.array(rewards)
            analysis[f'{prefix}_reward_mean'] = float(np.mean(rewards))
            analysis[f'{prefix}_reward_std'] = float(np.std(rewards))
            analysis[f'{prefix}_reward_min'] = float(np.min(rewards))
            analysis[f'{prefix}_reward_max'] = float(np.max(rewards))
            analysis[f'{prefix}_reward_median'] = float(np.median(rewards))
        elif len(rewards) == 1 or not isinstance(rewards, (list, np.ndarray)):
            # Single episode case
            reward_val = float(rewards[0]) if isinstance(rewards, (list, np.ndarray)) else float(rewards)
            analysis[f'{prefix}_reward_mean'] = reward_val
            analysis[f'{prefix}_reward_std'] = 0.0
            analysis[f'{prefix}_reward_min'] = reward_val
            analysis[f'{prefix}_reward_max'] = reward_val
            analysis[f'{prefix}_reward_median'] = reward_val

    # Process test and train rewards
    process_rewards(runner_log.get('test/mean_score'), 'test')
    process_rewards(runner_log.get('train/mean_score'), 'train')

    # Completion timestep and success rate statistics
    for prefix in ['test', 'train']:
        completion_key = f'{prefix}/completion_timesteps'
        if completion_key in runner_log:
            completion_times = runner_log[completion_key]
            if completion_times and len(completion_times) > 0:
                completion_times = np.array(completion_times)
                analysis[f'{prefix}_completion_mean'] = float(np.mean(completion_times))
                analysis[f'{prefix}_completion_std'] = float(np.std(completion_times))
                analysis[f'{prefix}_completion_min'] = float(np.min(completion_times))
                analysis[f'{prefix}_completion_max'] = float(np.max(completion_times))
                analysis[f'{prefix}_completion_median'] = float(np.median(completion_times))

        # Success rate statistics
        success_key = f'{prefix}/success_rate'
        if success_key in runner_log:
            analysis[f'{prefix}_success_rate'] = float(runner_log[success_key])

    return analysis

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_test', default=None, type=int, help='Number of test episodes (overrides config)')
@click.option('--n_test_vis', default=None, type=int, help='Number of test episodes with visualization (overrides config)')
@click.option('--n_train', default=0, type=int, help='Number of train episodes (overrides config)')
@click.option('--n_train_vis', default=0, type=int, help='Number of train episodes with visualization (overrides config)')
@click.option('--test_start_seed', default=50192, type=int, help='Starting seed for test episodes (overrides config)')
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

    # Perform comprehensive statistical analysis
    enhanced_stats = analyze_evaluation_results(runner_log)

    # Combine original log with enhanced statistics
    combined_log = dict(runner_log)
    combined_log.update(enhanced_stats)

    # dump log to json
    json_log = dict()
    for key, value in combined_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif hasattr(value, 'item'):  # Handle numpy scalars
            json_log[key] = float(value.item()) if hasattr(value, 'item') else float(value)
        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            json_log[key] = float(value)
        else:
            json_log[key] = value

    # Print comprehensive statistics summary
    print("\n=== Evaluation Results Summary ===")
    if 'test_reward_mean' in json_log:
        print(f"Test Reward: {json_log['test_reward_mean']:.3f} ± {json_log.get('test_reward_std', 0):.3f}")
        if 'test_completion_mean' in json_log:
            print(f"Test Completion Time: {json_log['test_completion_mean']:.1f} ± {json_log.get('test_completion_std', 0):.1f} steps")
        if 'test_success_rate' in json_log:
            print(f"Test Success Rate: {json_log['test_success_rate']:.1%}")

    if 'train_reward_mean' in json_log:
        print(f"Train Reward: {json_log['train_reward_mean']:.3f} ± {json_log.get('train_reward_std', 0):.3f}")
        if 'train_completion_mean' in json_log:
            print(f"Train Completion Time: {json_log['train_completion_mean']:.1f} ± {json_log.get('train_completion_std', 0):.1f} steps")
        if 'train_success_rate' in json_log:
            print(f"Train Success Rate: {json_log['train_success_rate']:.1%}")

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    print(f"\nDetailed results saved to: {out_path}")

if __name__ == '__main__':
    main()
