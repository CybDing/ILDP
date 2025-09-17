#!/usr/bin/env python3
"""
Script to adapt a trained diffusion policy model to a new environment.

Usage:
    python adapt_trained_model.py --checkpoint path/to/model.ckpt --config path/to/config.yaml
"""

import argparse
import hydra
from omegaconf import OmegaConf
import torch

from genesis_ILDP.policy.action_diffusion_image_policy import ActionDiffusionImagePolicy
from genesis_ILDP.utils.domain_adaptation import DomainAdapter
from genesis_ILDP.env_runner.pusht_image_runner import PushTImageRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', required=True, help='Path to model config')
    parser.add_argument('--new_env_config', required=True, help='Path to new environment config')
    args = parser.parse_args()

    # Load trained model
    cfg = OmegaConf.load(args.config)
    model = hydra.utils.instantiate(cfg.policy)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    print(f"Loaded trained model from {args.checkpoint}")

    # Load new environment
    new_env_cfg = OmegaConf.load(args.new_env_config)
    new_env_runner = hydra.utils.instantiate(new_env_cfg.env_runner)

    print(f"Created new environment: {type(new_env_runner).__name__}")

    # Perform domain adaptation
    print("Starting domain adaptation...")

    # 1. Create adapter
    adapter = DomainAdapter(model.normalizer)

    # 2. Collect sample data from new environment
    print("Collecting sample data from new environment...")
    new_env_data = adapter.collect_env_data_sample(new_env_runner.env, n_samples=200)

    print(f"Collected data shapes:")
    print(f"  Actions: {new_env_data['action'].shape}")
    print(f"  Agent positions: {new_env_data['agent_pos'].shape}")

    # 3. Create adapted normalizer
    print("Creating adapted normalizer...")
    adapted_normalizer = adapter.create_adapted_normalizer(new_env_data)

    # 4. Set new normalizer on model
    model.set_normalizer(adapted_normalizer)

    print("Domain adaptation complete!")

    # 5. Test adapted model
    print("Testing adapted model...")
    try:
        test_results = new_env_runner.run(model)
        print(f"Test results: {test_results}")
        print("Model successfully adapted and tested!")
    except Exception as e:
        print(f"Test failed: {e}")
        print("You may need to further adjust the adaptation parameters.")

    # 6. Save adapted model
    output_path = args.checkpoint.replace('.ckpt', '_adapted.ckpt')
    adapted_checkpoint = {
        'model': model.state_dict(),
        'normalizer': adapted_normalizer.state_dict(),
        'original_config': cfg,
        'adaptation_info': {
            'original_action_range': str(adapter.original_normalizer['action'].params_dict),
            'new_action_range': str(adapted_normalizer['action'].params_dict),
            'adaptation_samples': new_env_data['action'].shape[0]
        }
    }

    torch.save(adapted_checkpoint, output_path)
    print(f"Saved adapted model to {output_path}")

if __name__ == "__main__":
    main()