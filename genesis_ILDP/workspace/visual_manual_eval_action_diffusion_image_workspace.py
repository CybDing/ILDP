if __name__ == "__main__":
    import sys
    import pathlib
    import os
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import numpy as np
import wandb
import json

from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from genesis_ILDP.policy.pretrain.action_diffusion_image_policy import ActionDiffusionImagePolicy
from genesis_ILDP.env_runner.base_image_runner import BaseImageRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)


class EvalActionDiffusionImageWorkspace(BaseWorkspace):
    """
    Evaluation workspace specifically for ActionDiffusionImagePolicy.
    """
    # include_keys = ['global_step', 'epoch']
    include_keys = None # eval does not need training information 

    def __init__(self, cfg: OmegaConf, output_dir=None, enable_render=True):
        """
        Initialize eval workspace by loading checkpoint.

        Args:
            cfg: Eval config containing checkpoint_path and runtime overrides
            output_dir: Output directory for results
        """
        # Load checkpoint to get model config
        import dill
        checkpoint_path = cfg.get('checkpoint_path')
        if checkpoint_path is None:
            raise ValueError("cfg must contain 'checkpoint_path' for evaluation")

        device = cfg.get('device', 'cuda:0')
        payload = torch.load(checkpoint_path, pickle_module=dill, map_location=device)

        # Use checkpoint's config for model/policy/dataset
        model_cfg = payload['cfg']

        # Override device and output_dir from eval config
        if 'device' in cfg:
            model_cfg.training.device = cfg.device

        # Store both configs
        super().__init__(model_cfg, output_dir=output_dir)
        self.eval_cfg = cfg  # Store eval-specific overrides
        self.device = self.eval_cfg.device
        
        self.eval_cfg.env_runner.enable_render = enable_render # control the render option

        self.model: ActionDiffusionImagePolicy = hydra.utils.instantiate(model_cfg.policy)

        # Determine which weights to load: EMA or regular
        use_ema = model_cfg.training.get('use_ema', False)

        # Load the appropriate weights from checkpoint
        if use_ema and 'ema_model' in payload['state_dicts']:
            # Load EMA weights into the model
            self.model.load_state_dict(payload['state_dicts']['ema_model'])
            print(f"EvalActionDiffusionImageWorkspace initialized:")
            print(f"- Checkpoint: {checkpoint_path}")
            print(f"- Model type: {type(self.model).__name__}")
            print(f"- Device: {model_cfg.training.device}")
            print(f"- Loaded weights: EMA model")
        else:

            self.model.load_state_dict(payload['state_dicts']['model'])
            print(f"EvalActionDiffusionImageWorkspace initialized:")
            print(f"- Checkpoint: {checkpoint_path}")
            print(f"- Model type: {type(self.model).__name__}")
            print(f"- Device: {model_cfg.training.device}")
            print(f"- Loaded weights: Regular model")

        # Load other state if needed (global_step, epoch, etc.)
        if self.include_keys is not None:
            for key in self.include_keys:
                if key in payload.get('pickles', {}):
                    self.__dict__[key] = dill.loads(payload['pickles'][key])

    def run(self, policy=None, runner_params=None):
        """
        Run evaluation with the given policy and runner parameters.

        Args:
            policy: The policy to evaluate. If None, uses self.model or self.ema_model
            runner_params: Dictionary of parameters to override env_runner config

        Returns:
            runner_log: Dictionary containing evaluation results
        """
        # self.cfg is the checkpoint's config (model/policy/dataset)
        cfg = copy.deepcopy(self.cfg)

        # Setup normalizer from dataset (needed for policy to work correctly)
        from genesis_ILDP.dataset.base_dataset import BaseImageDataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()

        # Set normalizer for the model
        self.model.set_normalizer(normalizer)

        # Get policy (use provided or default to self.model which has correct weights)
        if policy is None:
            policy = self.model

        device = torch.device(self.device)
        policy.to(device)
        policy.eval()

        # Configure env_runner with overrides
        # Priority: runner_params > eval_cfg.env_runner > checkpoint config
        env_runner_config = cfg.task.env_runner

        # Apply eval config overrides first
        if hasattr(self, 'eval_cfg') and 'env_runner' in self.eval_cfg:
            env_runner_config = OmegaConf.merge(env_runner_config, self.eval_cfg.env_runner)

        # Then apply runtime overrides
        if runner_params is not None:
            env_runner_config = OmegaConf.merge(env_runner_config, runner_params)

        # Instantiate environment runner
        env_runner: BaseImageRunner = hydra.utils.instantiate(
            env_runner_config,
            output_dir=self.output_dir
        )
        assert isinstance(env_runner, BaseImageRunner)

        print(f"Running evaluation with {type(env_runner).__name__}")

        # Run evaluation
        with torch.no_grad():
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
        out_path = pathlib.Path(self.output_dir).joinpath('eval_log.json')
        with open(out_path, 'w') as f:
            json.dump(json_log, f, indent=2, sort_keys=True)
        print(f"Results saved to: {out_path}")

        return runner_log


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="eval_workspace_demo")
def main(cfg):
    """
    Standalone evaluation script using eval config.

    The config (eval_workspace_demo.yaml) should contain:
    - checkpoint_path: path to trained checkpoint
    - device: evaluation device
    - output_dir: where to save results
    - env_runner: runtime parameter overrides

    All model/policy/dataset settings are loaded from checkpoint automatically.
    """
    import pathlib
    output_dir = pathlib.Path(cfg.get('output_dir', './eval_output'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create workspace - this automatically loads checkpoint
    workspace = EvalActionDiffusionImageWorkspace(cfg, output_dir=str(output_dir))

    # Run evaluation (uses eval_cfg overrides from __init__)
    runner_log = workspace.run()

    print("\nEvaluation Results:")
    for key, value in runner_log.items():
        if not isinstance(value, wandb.sdk.data_types.video.Video):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
