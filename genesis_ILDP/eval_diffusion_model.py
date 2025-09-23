"""
Enhanced Evaluation Script for Diffusion Policy Model
Usage:
    python eval_diffusion_model.py --config-path config --config-name eval_diffusion_model

Features:
    - Changeable model checkpoints
    - Action prediction visualization
    - Real-time environment rendering
    - Detailed logging and metrics
    - Video recording capabilities
"""

import sys
import os
import pathlib
import hydra
import torch
import dill
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from omegaconf import OmegaConf
from typing import Dict, List
import cv2

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from genesis_ILDP.policy.action_diffusion_image_policy import ActionDiffusionImagePolicy
from genesis_ILDP.env_runner.pusht_image_runner import PushTImageRunner
from genesis_ILDP.env.pushT_env import PushTEnv
from genesis_ILDP.gym_util.multistep_wrapper_parallel import MultiStepWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DiffusionModelEvaluator:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.output_dir = pathlib.Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.policy = None
        self.env = None
        self.action_history = []
        self.observation_history = []
        self.reward_history = []

        print(f"Initializing DiffusionModelEvaluator...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")

    def load_model(self):
        """Load model from checkpoint"""
        print(f"Loading model from: {self.cfg.checkpoint_path}")

        if not os.path.exists(self.cfg.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.cfg.checkpoint_path}")

        # Load checkpoint
        payload = torch.load(self.cfg.checkpoint_path, pickle_module=dill, map_location=self.device)

        # Instantiate policy
        self.policy = hydra.utils.instantiate(self.cfg.policy)

        # Load policy state from checkpoint
        if 'model' in payload:
            self.policy.load_state_dict(payload['model'])
            print("Loaded model state from checkpoint")
        elif 'state_dict' in payload:
            self.policy.load_state_dict(payload['state_dict'])
            print("Loaded state_dict from checkpoint")
        else:
            # Try loading from workspace if available
            if 'ema_model' in payload and self.cfg.use_ema:
                self.policy.load_state_dict(payload['ema_model'])
                print("Loaded EMA model from checkpoint")
            else:
                print("Warning: Could not find model weights in checkpoint")

        self.policy.to(self.device)
        self.policy.eval()
        print("Model loaded and set to evaluation mode")

    def setup_environment(self):
        """Setup PushT environment"""
        print("Setting up PushT environment...")

        # Create base environment
        base_env = PushTEnv(
            render_size=self.cfg.env.image_shape[1:],  # (96, 96)
            xlim=self.cfg.env.xlim,
            ylim=self.cfg.env.ylim,
            fps=self.cfg.env.fps,
            show_fps=self.cfg.env.show_fps
        )

        # Wrap with MultiStepWrapper
        self.env = MultiStepWrapper(
            base_env,
            n_obs_steps=self.cfg.n_obs_steps,
            n_action_steps=self.cfg.n_action_steps,
            max_episode_steps=self.cfg.env.max_steps,
            n_envs=self.cfg.env.n_envs
        )

        # Initialize environment
        self.env.start(
            n_envs=self.cfg.env.n_envs,
            show_interact_viewer=self.cfg.env.show_viewer,
            env_separate=True,
            seed=self.cfg.env.seeds
        )

        print(f"Environment setup complete with {self.cfg.env.n_envs} environments")

    def visualize_action_prediction(self, obs_dict, action_pred, step_idx):
        """Visualize action predictions and observations"""
        if not self.cfg.visualization.save_plots:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot current observation image
        if 'image' in obs_dict:
            img = obs_dict['image'][0, -1].cpu().numpy()  # Last observation step
            if img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Current Observation')
            axes[0, 0].axis('off')

        # Plot agent position history
        if hasattr(self, 'observation_history') and len(self.observation_history) > 0:
            agent_positions = [obs['agent_pos'][0, -1].cpu().numpy() for obs in self.observation_history[-10:]]
            if agent_positions:
                positions = np.array(agent_positions)
                axes[0, 1].plot(positions[:, 0], positions[:, 1], 'b-o', label='Agent Path')
                axes[0, 1].set_title('Agent Position History')
                axes[0, 1].legend()
                axes[0, 1].grid(True)

        # Plot predicted actions
        actions = action_pred[0].cpu().numpy()  # First environment
        axes[0, 2].plot(actions[:, 0], label='X Actions', marker='o')
        axes[0, 2].plot(actions[:, 1], label='Y Actions', marker='s')
        if actions.shape[1] > 2:
            axes[0, 2].plot(actions[:, 2], label='Z Actions', marker='^')
        axes[0, 2].set_title('Predicted Action Sequence')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Plot action trajectory in 2D space
        axes[1, 0].plot(actions[:, 0], actions[:, 1], 'r-o', label='Action Trajectory')
        axes[1, 0].set_title('Action Trajectory (X-Y)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axis('equal')

        # Plot action magnitudes
        action_mags = np.linalg.norm(actions[:, :2], axis=1)
        axes[1, 1].plot(action_mags, 'g-o')
        axes[1, 1].set_title('Action Magnitudes')
        axes[1, 1].grid(True)

        # Plot reward history
        if len(self.reward_history) > 0:
            axes[1, 2].plot(self.reward_history, 'purple')
            axes[1, 2].set_title('Reward History')
            axes[1, 2].grid(True)

        plt.tight_layout()
        plot_path = self.output_dir / f'action_prediction_step_{step_idx:04d}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    def log_action_details(self, obs_dict, action_pred, step_idx):
        """Log detailed action information"""
        if not self.cfg.logging.verbose:
            return

        print(f"\n=== Step {step_idx} Action Details ===")

        # Log observation info
        if 'agent_pos' in obs_dict:
            agent_pos = obs_dict['agent_pos'][0, -1].cpu().numpy()
            print(f"Current agent position: [{agent_pos[0]:.4f}, {agent_pos[1]:.4f}]")

        # Log action predictions
        actions = action_pred[0].cpu().numpy()
        print(f"Predicted actions shape: {actions.shape}")
        print(f"Action sequence:")
        for i, action in enumerate(actions):
            if len(action) == 2:
                print(f"  Action {i}: [{action[0]:.4f}, {action[1]:.4f}]")
            else:
                print(f"  Action {i}: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")

        # Log action statistics
        action_mean = np.mean(actions, axis=0)
        action_std = np.std(actions, axis=0)
        action_range = np.ptp(actions, axis=0)

        print(f"Action statistics:")
        print(f"  Mean: {action_mean}")
        print(f"  Std:  {action_std}")
        print(f"  Range: {action_range}")

    def run_evaluation(self):
        """Main evaluation loop"""
        print("\n" + "="*50)
        print("Starting Evaluation")
        print("="*50)

        # Load model and setup environment
        self.load_model()
        self.setup_environment()

        # Initialize tracking variables
        total_episodes = 0
        total_steps = 0
        episode_rewards = []
        episode_lengths = []

        # Enable recording if requested
        if self.cfg.visualization.record_video:
            self.env.start_recording()

        try:
            for episode in range(self.cfg.num_episodes):
                print(f"\n--- Episode {episode + 1}/{self.cfg.num_episodes} ---")

                # Reset environment
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                done = False

                self.action_history = []
                self.observation_history = []
                self.reward_history = []

                while not done and episode_length < self.cfg.env.max_steps:
                    # Prepare observation for policy
                    obs_dict = dict(obs)
                    if 'envs_idx' in obs_dict:
                        del obs_dict['envs_idx']

                    # Convert to tensors and move to device
                    for key, value in obs_dict.items():
                        if isinstance(value, np.ndarray):
                            obs_dict[key] = torch.from_numpy(value).to(self.device)

                    # Store observation
                    self.observation_history.append(obs_dict)

                    # Predict action
                    with torch.no_grad():
                        action_result = self.policy.predict_action(obs_dict)
                        action = action_result['action']
                        action_pred = action_result['action_pred']

                    # Store action
                    self.action_history.append(action)

                    # Log and visualize
                    self.log_action_details(obs_dict, action_pred, total_steps)
                    if total_steps % self.cfg.visualization.plot_every == 0:
                        self.visualize_action_prediction(obs_dict, action_pred, total_steps)

                    # Execute action in environment
                    obs, reward, done, info, env_status = self.env.step(action.cpu().numpy())

                    # Track metrics
                    if isinstance(reward, torch.Tensor):
                        reward_val = reward[0].item()
                    else:
                        reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward

                    episode_reward += reward_val
                    self.reward_history.append(reward_val)
                    episode_length += 1
                    total_steps += 1

                    # Print step info
                    if self.cfg.logging.verbose:
                        print(f"Step {episode_length}: reward={reward_val:.4f}")

                # Episode finished
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                total_episodes += 1

                print(f"Episode {episode + 1} finished:")
                print(f"  Length: {episode_length}")
                print(f"  Total reward: {episode_reward:.4f}")
                print(f"  Average reward: {episode_reward/episode_length:.4f}")

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")

        finally:
            # Stop recording and save results
            if self.cfg.visualization.record_video:
                video_path = self.output_dir / 'evaluation_video.mp4'
                self.env.stop_recording(str(video_path))
                print(f"Video saved to: {video_path}")

            # Save evaluation results
            self.save_results(episode_rewards, episode_lengths, total_steps)

    def save_results(self, episode_rewards, episode_lengths, total_steps):
        """Save evaluation results to JSON"""
        results = {
            'num_episodes': len(episode_rewards),
            'total_steps': total_steps,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'std_episode_reward': np.std(episode_rewards) if episode_rewards else 0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
            'success_rate': sum(1 for r in episode_rewards if r > self.cfg.success_threshold) / len(episode_rewards) if episode_rewards else 0,
            'checkpoint_path': self.cfg.checkpoint_path,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }

        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"Episodes completed: {results['num_episodes']}")
        print(f"Total steps: {results['total_steps']}")
        print(f"Mean episode reward: {results['mean_episode_reward']:.4f} ± {results['std_episode_reward']:.4f}")
        print(f"Mean episode length: {results['mean_episode_length']:.2f}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Results saved to: {results_path}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")),
    config_name="eval_diffusion_model"
)
def main(cfg: OmegaConf):
    evaluator = DiffusionModelEvaluator(cfg)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()