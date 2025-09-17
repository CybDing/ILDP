"""
Domain adaptation utilities for transferring trained models between different environments.
"""

import numpy as np
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer

class DomainAdapter:
    """Adapt trained models to new environments with different action/observation ranges."""

    def __init__(self, original_normalizer):
        self.original_normalizer = original_normalizer

    def create_adapted_normalizer(self, new_env_data):
        """
        Create a new normalizer adapted to the target environment.

        Args:
            new_env_data: dict with 'action' and 'agent_pos' from target environment
        """
        adapted_normalizer = LinearNormalizer()

        # Fit to new environment data
        adapted_normalizer.fit(
            data=new_env_data,
            last_n_dims=1,
            mode='limits',  # Use limits mode for consistent scaling
            output_max=1.0,
            output_min=-1.0
        )

        # Keep image normalization the same (images should be environment-agnostic)
        if 'image' in self.original_normalizer.params_dict:
            adapted_normalizer['image'] = self.original_normalizer['image']

        return adapted_normalizer

    def collect_env_data_sample(self, env, n_samples=100):
        """
        Collect sample data from new environment to fit normalizer.

        Args:
            env: The target environment
            n_samples: Number of samples to collect
        """
        actions = []
        agent_positions = []

        for _ in range(n_samples):
            obs = env.reset()

            # Random exploration to sample the action/position space
            for _ in range(10):  # 10 steps per episode
                # Sample random action within env bounds
                action = env.action_space.sample()
                obs, _, done, _ = env.step(action)

                actions.append(action)
                agent_positions.append(obs['agent_pos'])

                if done:
                    break

        return {
            'action': np.array(actions),
            'agent_pos': np.array(agent_positions)
        }

    def remap_action_ranges(self,
                           source_min, source_max,
                           target_min, target_max):
        """
        Create a function to remap actions from source range to target range.

        Returns a function that can be applied to actions before sending to environment.
        """
        def remap_fn(actions):
            # Normalize to [0, 1]
            normalized = (actions - source_min) / (source_max - source_min)
            # Scale to target range
            remapped = normalized * (target_max - target_min) + target_min
            return remapped

        return remap_fn

# Usage example:
def adapt_model_to_new_env(trained_model, new_env):
    """
    Complete pipeline to adapt a trained model to a new environment.
    """
    # 1. Get original normalizer
    original_normalizer = trained_model.normalizer

    # 2. Create adapter
    adapter = DomainAdapter(original_normalizer)

    # 3. Collect sample data from new environment
    new_env_data = adapter.collect_env_data_sample(new_env)

    # 4. Create adapted normalizer
    adapted_normalizer = adapter.create_adapted_normalizer(new_env_data)

    # 5. Set new normalizer on model
    trained_model.set_normalizer(adapted_normalizer)

    print("Model successfully adapted to new environment!")
    print(f"Original action range: {original_normalizer['action'].params_dict}")
    print(f"New action range: {adapted_normalizer['action'].params_dict}")

    return trained_model