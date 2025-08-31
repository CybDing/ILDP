"""
Usage:
python eval-test.py    
--config ./data/config/eval_pushT_dp_cnn.yaml
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
from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

# @click.command()
# @click.option('-c', '--checkpoint', required=True)
# @click.option('-o', '--output_dir', required=True)
# @click.option('-d', '--device', default='cuda:0')

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config')),
    config_name="eval_pushT_dp_cnn"  

)
def main(cfg: OmegaConf):
    if os.path.exists(cfg.output_dir):
        click.confirm(f"Output path {cfg.output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(cfg.checkpoint_path, 'rb'), pickle_module=dill)

    # cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=cfg.output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None) 
    
    # get policy from workspace
    policy = hydra.utils.instantiate(cfg.policy)
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=cfg.output_dir)
    runner_log = env_runner.run(policy)

    print(runner_log)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(cfg.output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
