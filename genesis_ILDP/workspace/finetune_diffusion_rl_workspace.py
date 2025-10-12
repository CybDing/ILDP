"""
Fine-tuning workspace for diffusion policy with PPO-based RL.

Architecture:
1. Load pretrained policy -> convert to FTActionDiffusionImagePolicy
2. Create Critic (Q-network) for value estimation
3. Create PPODiffusion trainer
4. Training loop:
   - Collect trajectories (TrajsCollector)
   - Compute old_values (Critic)
   - Compute new_log_prob(Actor): Compute new_prob with given obs and old predicted noise probability under the new up_to_date policy
   - PPO loss calculation (PPODiffusion): compute ploss for policy model and vloss for value inside the critic model
"""

if __name__ == "__main__":
    import sys
    import pathlib
    import os
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent) # change to the ILDP folder as parent folder
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import dill
import wandb
import copy

from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from genesis_ILDP.agent.FT_trajs_collector import TrajsCollector
from genesis_ILDP.policy.FT_action_diffusion_image_policy import FTActionDiffusionImagePolicy
from genesis_ILDP.model.rl.QNet import Critic
from genesis_ILDP.model.rl.PPODiffusion import PPODiffusion
from genesis_ILDP.utils.pytorch_util import dict_apply


class ReplayBuffer:
    """
    Replay buffer for storing trajectories from TrajsCollector.
    Stores: obs, actions (diffusion chains), rewards, episode_ends, dones
    """

    def __init__(self, max_episodes=1000, device='cuda:0'):
        self.max_episodes = max_episodes
        self.device = device
        self.buffer = {
            'obs': {'image': [], 'agent_pos': []},
            'action': [],  # Normalized diffusion actions (global_steps, diff_steps+1, horizon, Da)
            'reward': [],
            'episode_ends': [],  # Cumulative episode end indices
            'done': []  # Per-step done flags
        }
        self.num_episodes = 0

    def add_trajectories(self, episode_buffer):
        """Add trajectories from TrajsCollector."""
        if len(episode_buffer['obs']['img']) == 0:
            return

        new_episodes = len(episode_buffer['episode_ends'])

        # Transfer to device
        new_obs_img = episode_buffer['obs']['img'].to(self.device)
        new_obs_pos = episode_buffer['obs']['agent_pos'].to(self.device)
        new_action = episode_buffer['action'].to(self.device)
        new_reward = episode_buffer['reward'].to(self.device)

        # Update episode_ends with offset
        offset = self.buffer['episode_ends'][-1] if len(self.buffer['episode_ends']) > 0 else 0
        new_ends = [e + offset for e in episode_buffer['episode_ends']]

        # Create done flags (1 at episode ends, 0 otherwise)
        total_steps = new_obs_img.shape[0]
        done_flags = torch.zeros(total_steps, device=self.device)
        for end_idx in episode_buffer['episode_ends']:
            done_flags[end_idx - 1] = 1.0  # Mark last step of each episode

        # Concatenate to buffer
        if len(self.buffer['obs']['image']) == 0:
            self.buffer['obs']['image'] = new_obs_img
            self.buffer['obs']['agent_pos'] = new_obs_pos
            self.buffer['action'] = new_action
            self.buffer['reward'] = new_reward
            self.buffer['done'] = done_flags
        else:
            self.buffer['obs']['image'] = torch.cat([self.buffer['obs']['image'], new_obs_img], dim=0)
            self.buffer['obs']['agent_pos'] = torch.cat([self.buffer['obs']['agent_pos'], new_obs_pos], dim=0)
            self.buffer['action'] = torch.cat([self.buffer['action'], new_action], dim=0)
            self.buffer['reward'] = torch.cat([self.buffer['reward'], new_reward], dim=0)
            self.buffer['done'] = torch.cat([self.buffer['done'], done_flags], dim=0)

        self.buffer['episode_ends'].extend(new_ends)
        self.num_episodes += new_episodes

        # Trim to max episodes
        if self.num_episodes > self.max_episodes:
            self._trim_old_episodes()

        print(f"Buffer: {self.num_episodes} episodes, {len(self.buffer['reward'])} steps")

    def _trim_old_episodes(self):
        """Remove oldest episodes to maintain max_episodes."""
        excess_episodes = self.num_episodes - self.max_episodes

        # Find cutoff index
        cutoff_idx = self.buffer['episode_ends'][excess_episodes - 1]

        # Trim data
        self.buffer['obs']['image'] = self.buffer['obs']['image'][cutoff_idx:]
        self.buffer['obs']['agent_pos'] = self.buffer['obs']['agent_pos'][cutoff_idx:]
        self.buffer['action'] = self.buffer['action'][cutoff_idx:]
        self.buffer['reward'] = self.buffer['reward'][cutoff_idx:]
        self.buffer['done'] = self.buffer['done'][cutoff_idx:]

        # Update episode_ends
        new_ends = []
        for end in self.buffer['episode_ends'][excess_episodes:]:
            new_ends.append(end - cutoff_idx)
        self.buffer['episode_ends'] = new_ends

        self.num_episodes = self.max_episodes

    def sample_batch(self, batch_size):
        """Sample random batch of steps."""
        total_steps = len(self.buffer['reward'])
        if total_steps == 0:
            return None

        # Random sample indices
        indices = torch.randint(0, total_steps, (batch_size,), device=self.device)

        batch = {
            'obs': {
                'image': self.buffer['obs']['image'][indices],
                'agent_pos': self.buffer['obs']['agent_pos'][indices]
            },
            'action': self.buffer['action'][indices],  # (B, diff_steps+1, horizon, Da)
            'reward': self.buffer['reward'][indices],
            'done': self.buffer['done'][indices]
        }
        return batch, indices

    def get_all_data(self):
        """Return all data for advantage computation."""
        return {
            'obs': self.buffer['obs'],
            'action': self.buffer['action'],
            'reward': self.buffer['reward'],
            'done': self.buffer['done'],
            'episode_ends': self.buffer['episode_ends']
        }

    def __len__(self):
        return len(self.buffer['reward'])


class FineTuneDiffusionRLWorkspace(BaseWorkspace):
    """Fine-tuning workspace for diffusion policy with PPO."""

    include_keys = ['global_step', 'epoch', 'rollout_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self.device = torch.device(cfg.training.device)

        # Load pretrained policy and convert to FT version
        self._load_and_convert_policy()

        # Create critic
        self._create_critic()

        # Create PPO trainer
        self._create_ppo_trainer()

        # Create trajectory collector
        self._create_trajs_collector()

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            max_episodes=cfg.training.get('max_replay_episodes', 1000),
            device=self.device
        )

        # Cache for old_logprob and old_value
        self.old_logprob_cache = None
        self.old_value_cache = None
        self.advantages_cache = None

        # Setup optimizers
        self._setup_optimizers()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.rollout_step = 0

        print(f"\n{'='*60}")
        print("FineTune RL Workspace Initialized")
        print(f"{'='*60}")
        print(f"  Policy: {type(self.policy).__name__}")
        print(f"  Critic: layers={cfg.critic.layers_Sdim}")
        print(f"  PPO: max_diff_idx={cfg.ppo.max_FTdiff_idx}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

    def _load_and_convert_policy(self):
        """Load pretrained policy and convert to FTActionDiffusionImagePolicy."""
        checkpoint_path = self.cfg.checkpoint_path
        payload = torch.load(checkpoint_path, pickle_module=dill, map_location=self.device)

        # Get model config
        model_cfg = payload['cfg']

        # Create FTActionDiffusionImagePolicy with same config
        # Important: use FT version which has get_logprob()
        ft_policy_cfg = copy.deepcopy(model_cfg.policy)
        ft_policy_cfg._target_ = 'genesis_ILDP.policy.FT_action_diffusion_image_policy.FTActionDiffusionImagePolicy'

        self.policy: FTActionDiffusionImagePolicy = hydra.utils.instantiate(ft_policy_cfg)

        # Load pretrained weights (EMA if available)
        if 'ema_model' in payload and payload['ema_model'] is not None:
            self.policy.load_state_dict(payload['ema_model'])
            print("Loaded EMA weights from checkpoint")
        else:
            self.policy.load_state_dict(payload['state_dicts']['model'])
            print("Loaded regular weights from checkpoint")

        self.policy.to(self.device)

        # Load normalizer from dataset (same as TrajsCollector)
        from genesis_ILDP.dataset.base_dataset import BaseImageDataset
        dataset: BaseImageDataset = hydra.utils.instantiate(model_cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        self.policy.set_normalizer(normalizer)

        # Store config for other components
        self.model_cfg = model_cfg

    def _create_critic(self):
        """Create Critic and adapt encoder from policy."""
        self.critic = Critic(
            shape_meta=self.model_cfg.policy.shape_meta,
            layers_Sdim=self.cfg.critic.layers_Sdim,
            n_obs=self.cfg.critic.n_obs,
            img_feature_dim=self.cfg.critic.get('img_feature_dim', 1024),
            enable_crop=self.cfg.critic.get('enable_crop', True),
            crop_shape=self.cfg.critic.get('crop_shape', (3, 76, 76))
        )

        # Adapt encoder from policy (freeze by default)
        self.critic.adapt_obs_encoder(
            img_encoder=self.policy.imgs_encoding_net,
            agent_pos_encoder=None,
            train_encoder=self.cfg.critic.get('train_encoder', False)
        )

        # Share normalizer with policy
        self.critic.normalizer = self.policy.normalizer

        self.critic.to(self.device)
        print("Critic created and encoder adapted from policy")

    def _create_ppo_trainer(self):
        """Create PPODiffusion trainer."""
        self.ppo_trainer = PPODiffusion(
            max_FTdiff_idx=self.cfg.ppo.max_FTdiff_idx,
            clip_base_ploss=self.cfg.ppo.clip_base_ploss,
            clip_rate_ploss=self.cfg.ppo.clip_rate_ploss,
            clip_vloss=self.cfg.ppo.get('clip_vloss', None),
            clip_advantage_upper_quantile=self.cfg.ppo.clip_advantage_upper_quantile,
            clip_advantage_lower_quantile=self.cfg.ppo.clip_advantage_lower_quantile,
            advantage_decay_coeff=self.cfg.ppo.advantage_decay_coeff
        )

        # Load policy into PPO trainer
        self.ppo_trainer.load_policy(self.policy)
        print("PPO trainer created")

    def _create_trajs_collector(self):
        """Create TrajsCollector for trajectory collection."""
        # Create minimal eval_cfg for TrajsCollector
        eval_cfg = OmegaConf.create({
            'checkpoint_path': self.cfg.checkpoint_path,
            'device': str(self.device)
        })

        self.trajs_collector = TrajsCollector(
            eval_cfg=eval_cfg,
            seed=self.cfg.training.get('rollout_seed', 50000),
            n_episodes=self.cfg.training.n_rollout_episodes,
            max_step=self.cfg.training.get('max_rollout_steps', 500)
        )

        # Update collector's policy to our FT policy
        self.trajs_collector.policy = self.policy

        print(f"TrajsCollector created: {self.cfg.training.n_rollout_episodes} episodes/rollout")

    def _setup_optimizers(self):
        """Setup optimizers for policy and critic."""
        # Policy optimizer
        policy_lr = self.cfg.optimizer.policy_lr
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=policy_lr
        )

        # Critic optimizer (only Q-network parameters, encoder frozen)
        critic_lr = self.cfg.optimizer.critic_lr
        self.critic_optimizer = torch.optim.Adam(
            self.critic.Qnet.parameters(),
            lr=critic_lr
        )

        print(f"Optimizers: policy_lr={policy_lr}, critic_lr={critic_lr}")

    def _compute_gae_advantages(self, gamma=0.99, gae_lambda=0.95):
        """
        Compute GAE advantages for all data in buffer.

        GAE formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        all_data = self.replay_buffer.get_all_data()

        if len(all_data['reward']) == 0:
            return None, None

        # Compute values for all states
        with torch.no_grad():
            self.critic.eval()
            values = self.critic(all_data['obs']).squeeze(-1)  # (total_steps,)

        rewards = all_data['reward']
        dones = all_data['done']
        total_steps = len(rewards)

        # Compute advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Iterate through each episode
        episode_starts = [0] + all_data['episode_ends'][:-1]
        episode_ends = all_data['episode_ends']

        for start_idx, end_idx in zip(episode_starts, episode_ends):
            ep_len = end_idx - start_idx

            # GAE computation for this episode
            gae = 0
            for t in reversed(range(ep_len)):
                global_t = start_idx + t

                if t == ep_len - 1:
                    # Last step of episode
                    next_value = 0.0
                    delta = rewards[global_t] - values[global_t]
                else:
                    next_value = values[global_t + 1]
                    delta = rewards[global_t] + gamma * next_value - values[global_t]

                # GAE accumulation
                gae = delta + gamma * gae_lambda * gae
                advantages[global_t] = gae
                returns[global_t] = gae + values[global_t]

        # Normalize advantages (important for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, values

    def _compute_old_logprob_and_value(self, batch_size=256):
        """
        Compute old_logprob and old_value for buffer data.
        These are computed BEFORE policy update for PPO.
        """
        all_data = self.replay_buffer.get_all_data()
        total_steps = len(all_data['reward'])

        if total_steps == 0:
            return None, None

        old_logprobs = []
        old_values = []

        # Process in batches
        for start_idx in range(0, total_steps, batch_size):
            end_idx = min(start_idx + batch_size, total_steps)

            batch_obs = {
                'image': all_data['obs']['image'][start_idx:end_idx],
                'agent_pos': all_data['obs']['agent_pos'][start_idx:end_idx]
            }
            batch_action = all_data['action'][start_idx:end_idx]  # (B, diff_steps+1, horizon, Da)

            with torch.no_grad():
                self.policy.eval()
                self.critic.eval()

                # Sample random denoising indices for each step
                B = batch_action.shape[0]
                max_idx = min(self.cfg.ppo.max_FTdiff_idx, batch_action.shape[1] - 2)  # -2 for chain_before/next
                denoising_indices = torch.randint(0, max_idx + 1, (B,), device=self.device)

                # Extract chain_before and chain_next based on denoising_idx
                chain_before = torch.stack([
                    batch_action[i, idx] for i, idx in enumerate(denoising_indices)
                ])  # (B, horizon, Da)

                chain_next = torch.stack([
                    batch_action[i, idx + 1] for i, idx in enumerate(denoising_indices)
                ])  # (B, horizon, Da)

                # Compute log probability using FTPolicy
                logprob = self.policy.get_logprob(
                    obs_dict=batch_obs,
                    chain_before=chain_before,
                    chain_next=chain_next,
                    denoising_idx=denoising_indices,
                    isNorm_action=True,  # Actions in buffer are normalized
                    isNorm_obs=False  # Obs in buffer are unnormalized
                )  # (B, horizon, Da)

                # Sum over horizon and action dims to get scalar logprob per step
                logprob_sum = logprob.sum(dim=[1, 2])  # (B,)

                # Compute values
                values = self.critic(batch_obs).squeeze(-1)  # (B,)

                old_logprobs.append(logprob_sum)
                old_values.append(values)

        old_logprobs = torch.cat(old_logprobs, dim=0)
        old_values = torch.cat(old_values, dim=0)

        return old_logprobs, old_values

    def run(self):
        """Main training loop."""
        # Initialize wandb
        if self.cfg.logging.get('mode', 'disabled') != 'disabled':
            wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                **self.cfg.logging
            )

        for epoch in range(self.cfg.training.num_epochs):
            self.epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.cfg.training.num_epochs}")
            print(f"{'='*60}")

            # Phase 1: Collect trajectories
            print("\n[Phase 1] Collecting trajectories...")
            self._collect_trajectories()

            # Phase 2: Compute advantages
            print("\n[Phase 2] Computing GAE advantages...")
            advantages, old_values = self._compute_gae_advantages(
                gamma=self.cfg.training.gamma,
                gae_lambda=self.cfg.training.gae_lambda
            )

            if advantages is None:
                print("No data in buffer, skipping training")
                continue

            # Phase 3: Compute old_logprob (before policy update)
            print("\n[Phase 3] Computing old logprobs...")
            old_logprobs, _ = self._compute_old_logprob_and_value()

            # Cache for PPO updates
            self.advantages_cache = advantages
            self.old_logprob_cache = old_logprobs
            self.old_value_cache = old_values

            # Phase 4: PPO training
            print("\n[Phase 4] PPO training...")
            train_metrics = self._train_ppo_epoch()

            # Log metrics
            if self.cfg.logging.get('mode', 'disabled') != 'disabled':
                wandb.log(train_metrics, step=self.global_step)

            # Phase 5: Checkpoint
            if (epoch + 1) % self.cfg.training.checkpoint_every == 0:
                self.save_checkpoint(tag=f'epoch_{epoch}')

        # Final save
        self.save_checkpoint(tag='final')

        if self.cfg.logging.get('mode', 'disabled') != 'disabled':
            wandb.finish()

    def _collect_trajectories(self):
        """Collect trajectories using TrajsCollector."""
        self.policy.eval()

        # Collect trajectories
        collected_trajs = self.trajs_collector.collect_trajs()

        # Add to replay buffer
        self.replay_buffer.add_trajectories(collected_trajs)

        self.rollout_step += 1

        # Log collection stats
        n_episodes = len(collected_trajs['episode_ends'])
        total_reward = collected_trajs['reward'].sum().item()
        avg_reward = total_reward / n_episodes if n_episodes > 0 else 0

        print(f"  Collected {n_episodes} episodes, avg_reward={avg_reward:.2f}")

    def _train_ppo_epoch(self):
        """Train policy and critic with PPO for one epoch."""
        self.policy.train()
        self.critic.train()

        epoch_losses = {'policy_loss': [], 'value_loss': []}

        num_batches = self.cfg.training.num_ppo_updates_per_epoch
        batch_size = self.cfg.training.batch_size

        for batch_idx in range(num_batches):
            # Sample batch
            batch, indices = self.replay_buffer.sample_batch(batch_size)
            if batch is None:
                break

            # Get cached advantages and old values for this batch
            batch_advantages = self.advantages_cache[indices]
            batch_old_values = self.old_value_cache[indices]
            batch_old_logprobs = self.old_logprob_cache[indices]

            # Sample random denoising indices
            B = batch_size
            max_idx = min(self.cfg.ppo.max_FTdiff_idx, batch['action'].shape[1] - 2)
            denoising_indices = torch.randint(0, max_idx + 1, (B,), device=self.device)

            # Extract chain_before and chain_next
            chain_before = torch.stack([
                batch['action'][i, idx] for i, idx in enumerate(denoising_indices)
            ])

            chain_next = torch.stack([
                batch['action'][i, idx + 1] for i, idx in enumerate(denoising_indices)
            ])

            # Compute PPO loss
            ploss, vloss = self.ppo_trainer.loss(
                chain_before=chain_before,
                chain_next=chain_next,
                advantage=batch_advantages,
                obs=batch['obs'],
                denoising_idx=denoising_indices,
                old_value=batch_old_values,
                old_logprob=batch_old_logprobs
            )

            # Policy update
            policy_loss = ploss.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.training.get('max_grad_norm', 1.0))
            self.policy_optimizer.step()

            # Critic update
            value_loss = vloss.mean()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.Qnet.parameters(), self.cfg.training.get('max_grad_norm', 1.0))
            self.critic_optimizer.step()

            epoch_losses['policy_loss'].append(policy_loss.item())
            epoch_losses['value_loss'].append(value_loss.item())

            self.global_step += 1

        # Compute mean losses
        metrics = {
            'train/policy_loss': np.mean(epoch_losses['policy_loss']) if epoch_losses['policy_loss'] else 0,
            'train/value_loss': np.mean(epoch_losses['value_loss']) if epoch_losses['value_loss'] else 0,
            'train/buffer_size': len(self.replay_buffer),
            'train/num_episodes': self.replay_buffer.num_episodes
        }

        print(f"  PPO: policy_loss={metrics['train/policy_loss']:.4f}, "
              f"value_loss={metrics['train/value_loss']:.4f}")

        return metrics


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="finetune_diffusion_rl"
)
def main(cfg: OmegaConf):
    workspace = FineTuneDiffusionRLWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()