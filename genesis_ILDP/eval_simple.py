"""
Simple Standalone Evaluation Script
Usage:
    python eval_simple.py --config-path config --config-name eval_action_diffusion_pusht_image checkpoint_path=./path/to/checkpoint.ckpt

Features:
    - Lightweight evaluation without training workspace
    - Direct Hydra instantiation of policy and environment
    - Configurable seeds for reproducible evaluation
    - Clean separation from training code
"""

import sys
import os
import pathlib
import hydra
import torch
from omegaconf import DictConfig
import logging

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="eval_action_diffusion_pusht_image", version_base=None)
def main(cfg: DictConfig) -> None:
    """Simple evaluation using direct Hydra instantiation."""

    logger.info("=== SIMPLE EVALUATION ===")
    logger.info(f"Checkpoint: {cfg.checkpoint_path}")
    logger.info(f"Device: {cfg.training.device}")

    # Check checkpoint exists
    if not os.path.exists(cfg.checkpoint_path):
        logger.error(f"Checkpoint not found: {cfg.checkpoint_path}")
        return

    # 1. Instantiate policy directly using Hydra
    logger.info("Instantiating policy...")
    policy = hydra.utils.instantiate(cfg.policy)
    logger.info(f"Policy created: {type(policy).__name__}")

    # 2. Load checkpoint
    logger.info("Loading checkpoint...")
    payload = torch.load(cfg.checkpoint_path, map_location=cfg.training.device)

    if cfg.use_ema and 'ema_model_state' in payload:
        logger.info("Loading EMA weights")
        # Create EMA model and load state
        ema_model = hydra.utils.instantiate(cfg.ema)
        ema_model.load_state_dict(payload['ema_model_state'])
        policy = ema_model
    else:
        logger.info("Loading regular model weights")
        policy.load_state_dict(payload['state_dict'])

    policy.eval()
    policy = policy.to(cfg.training.device)

    # 3. Instantiate environment runner directly using Hydra
    logger.info("Instantiating environment runner...")
    env_runner = hydra.utils.instantiate(cfg.task.env_runner)
    logger.info(f"Environment runner created: {env_runner.n_envs} envs, {env_runner.n_test} test episodes")

    # 4. Run evaluation with specified seeds
    logger.info("Running evaluation...")
    with torch.no_grad():
        results = env_runner.run(policy)

    # 5. Print and save results
    logger.info("=== RESULTS ===")
    for key, value in results.items():
        if 'mean_score' in key:
            logger.info(f"{key}: {value:.4f}")

    # Save results
    output_dir = pathlib.Path(cfg.multi_run.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "eval_results.txt"
    with open(results_file, 'w') as f:
        f.write("=== SIMPLE EVALUATION RESULTS ===\n")
        f.write(f"Checkpoint: {cfg.checkpoint_path}\n")
        f.write(f"Episodes: {env_runner.n_test}\n")
        f.write(f"Seeds: {env_runner.seed_test} to {env_runner.seed_test + env_runner.n_test}\n\n")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")

    logger.info(f"Results saved: {results_file}")
    logger.info("=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()