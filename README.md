# ILDP - Imitation Learning via Diffusion Policy

A modular framework for **diffusion-based imitation learning** in robotic manipulation, with support for **PPO-based reinforcement learning fine-tuning**. Built on top of [Stanford Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) and the [Genesis Simulator](https://genesis-world.readthedocs.io/).

## Overview

ILDP implements action prediction through denoising diffusion probabilistic models (DDPM/DDIM). The system learns to predict multi-step action sequences conditioned on visual observations, enabling smooth and temporally consistent robot control.

**Key Features:**
- Diffusion-based action policy with DDPM and DDIM sampling
- Vision backbones: ResNet18 (with optional SpatialSoftmax) and Vision Transformer (ViT)
- Conditioned U-Net for noise prediction with FiLM conditioning
- EMA (Exponential Moving Average) model for stable training
- PPO-based RL fine-tuning pipeline for policy improvement
- Critic networks supporting both state-only and image+state inputs
- PushT manipulation environment via Genesis simulator
- Zarr-based replay buffer and dataset management
- Hydra configuration system with WandB logging

## Architecture

```
Observation (image + agent_pos)
        |
   [Vision Encoder]  ──>  image features
   [Pos Encoder]     ──>  position features
   [Time Encoder]    ──>  diffusion step embedding
        |
   [Feature Merging] ──>  global conditioning vector
        |
   [Conditioned U-Net] ──>  noise prediction
        |
   [DDPM/DDIM Denoising Loop] ──>  predicted action sequence (horizon steps)
        |
   [Execute first n_action_steps]
```

## Project Structure

```
genesis_ILDP/
├── task/                          # Entry point scripts
│   ├── pretrain_dp.py             # Pre-training launcher
│   ├── run_rl_finetune_dp.py      # RL fine-tuning launcher
│   └── eval.py                    # Evaluation script
│
├── policy/                        # Policy implementations
│   ├── base_image_policy.py       # Abstract base policy
│   ├── pretrain/
│   │   └── action_diffusion_image_policy.py   # Core diffusion policy
│   ├── finetune/
│   │   └── finetune_action_diffusion_image_policy.py  # RL-compatible variant
│   └── benchmark/
│       └── action_diffusion_image_policy_reference.py
│
├── workspace/                     # Training/eval loops
│   ├── base_workspace.py          # Checkpoint save/load management
│   ├── train_action_diffusion_image_workspace.py  # Main training workspace
│   ├── finetune_diffusion_rl_workspace.py
│   └── visual_manual_eval_action_diffusion_image_workspace.py
│
├── model/                         # Neural network components
│   ├── encoding.py                # Time/position encodings
│   ├── diffusion/
│   │   ├── conditioned_unet.py    # U-Net with FiLM conditioning
│   │   └── diffuser.py
│   ├── vision/
│   │   ├── cnn_encoding.py        # ResNet18 + SpatialSoftmax
│   │   ├── vit_encoding.py        # Vision Transformer encoder
│   │   ├── vision_transformer.py  # ViT building blocks
│   │   ├── SpatialSoftmax.py
│   │   └── crop_randomizer.py     # Data augmentation
│   ├── common/
│   │   ├── noise_scheduler.py     # Cosine/linear beta schedules
│   │   ├── ema_model.py           # Exponential moving average
│   │   ├── modules.py             # Shift augmentation, pos shifter
│   │   └── mlp.py
│   ├── rl/
│   │   ├── critic.py              # Value function networks
│   │   └── rl_utils.py            # GAE calculation
│   └── memory/
│       └── memoryNET.py           # Memory-augmented networks
│
├── agent/                         # RL agent components
│   ├── rl_agent/
│   │   ├── base_agent.py          # Base agent with optimizer setup
│   │   └── ppo_agent.py           # PPO implementation
│   ├── trainer/
│   │   ├── base_trainer.py        # Base training loop
│   │   └── finetune_ppo_trainer.py  # PPO training loop
│   ├── collector/
│   │   └── traj_collector.py      # Trajectory collection
│   └── evaluator/
│       └── pusht_evaluator.py
│
├── env/                           # Environments
│   ├── pushT_env.py               # Genesis-based PushT environment
│   ├── pushT_env_collect.py
│   └── pushT_env_server.py
│
├── env_runner/                    # Evaluation runners
│   ├── base_image_runner.py
│   ├── pusht_image_runner.py
│   └── pusht_image_runner_AutoReset.py
│
├── dataset/                       # Data loading
│   ├── base_dataset.py
│   ├── pusht_image_dataset.py     # Zarr-based PushT dataset
│   └── sampler.py
│
├── config/                        # Hydra configurations
│   ├── train/                     # Training configs (CNN, ViT, MPS variants)
│   ├── finetune/                  # Fine-tuning configs
│   └── eval/                      # Evaluation configs (CUDA, MPS)
│
├── utils/                         # Utilities
│   ├── pytorch_util.py
│   ├── seed_util.py               # Reproducibility utilities
│   ├── wandb_logger.py
│   ├── replay_buffer.py
│   └── module_utils.py
│
├── script/                        # Auxiliary scripts
│   ├── collect_data_v1.py         # Data collection
│   ├── merge_datasets.py          # Dataset merging
│   └── clean_data_by_overlap.py   # Data cleaning
│
└── assets/                        # Robot URDF models and meshes
```

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch (with CUDA support recommended)
- Genesis simulator

### Setup

```bash
# Clone the repository
git clone <repo-url> && cd ILDP

# Install the package
pip install -e .

# Install additional dependencies
pip install torch torchvision  # Choose version matching your CUDA
pip install hydra-core wandb opencv-python shapely

# Install Genesis simulator (see Genesis docs)
pip install genesis-world
```

### Dependencies

Core dependencies (from `setup.py`):
- `numpy`, `torch`, `torchvision`
- `hydra-core` - Configuration management
- `wandb` - Experiment tracking
- `einops` - Tensor operations
- `diffusers` - Diffusion model utilities
- `zarr`, `numcodecs` - Dataset storage
- `pandas`, `shapely` - Data processing

## Usage

### 1. Data Collection

Collect demonstration data using the PushT environment:

```bash
python genesis_ILDP/script/collect_data_v1.py
```

Data is stored in Zarr format at the path specified in your config (default: `data/train_data/pusht/`).

### 2. Pre-training

Train a diffusion policy from demonstration data:

```bash
# Default configuration (ResNet18 backbone, CUDA)
python genesis_ILDP/task/pretrain_dp.py

# With ViT backbone
python genesis_ILDP/task/pretrain_dp.py --config-name train_action_diffusion_pusht_vit

# Override parameters
python genesis_ILDP/task/pretrain_dp.py training.num_epochs=200 training.batch_size=128

# For Apple Silicon (MPS)
python genesis_ILDP/task/pretrain_dp.py --config-name train_action_diffusion_pusht_image_mps
```

Available training configs:
| Config | Description |
|--------|-------------|
| `train_action_diffusion_pusht_image` | Default CNN (ResNet18) on CUDA |
| `train_action_diffusion_pusht_vit` | ViT backbone |
| `train_action_diffusion_pusht_vit_custom` | Custom ViT variant |
| `train_action_diffusion_pusht_image_mps` | Apple Silicon support |
| `train_action_diffusion_pusht_image_diffusers` | HuggingFace diffusers variant |

### 3. Evaluation

Evaluate a trained checkpoint:

```bash
# CUDA evaluation
python genesis_ILDP/task/eval.py \
    checkpoint_path=/path/to/checkpoint.ckpt \
    --config-name eval_checkpoint_cuda

# Override evaluation parameters
python genesis_ILDP/task/eval.py \
    checkpoint_path=/path/to/checkpoint.ckpt \
    eval.env_runner.n_test=100 \
    eval.env_runner.max_steps=500
```

### 4. RL Fine-tuning (WIP)

Fine-tune a pretrained policy using PPO:

```bash
python genesis_ILDP/task/run_rl_finetune_dp.py \
    --checkpoint /path/to/pretrained.ckpt \
    --output_dir outputs/finetune \
    --reset_optimizer \
    --reset_epoch
```

## Configuration

All configurations use [Hydra](https://hydra.cc/). Key parameters:

```yaml
# Diffusion parameters
horizon: 16              # Action prediction horizon
n_action_steps: 8        # Actions executed per step
n_obs_steps: 2           # Observation history length
diff_steps: 100          # Number of diffusion steps

# Training
training:
  num_epochs: 500
  batch_size: 64
  lr_scheduler: cosine
  use_ema: true

# Policy
policy:
  use_ddpm: true          # DDPM (true) or DDIM (false)
  # ddim_steps: 15        # DDIM inference steps
```

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Diffusion Policy | Working | DDPM/DDIM, CNN/ViT backbones |
| Pre-training Pipeline | Working | Full training loop with checkpointing |
| Evaluation | Working | Environment rollouts with metrics |
| Dataset (Zarr) | Working | PushT image dataset |
| PPO Fine-tuning | In Progress | Core structure exists, needs completion of advantage calculation and training loop integration |
| Normalizer | Stub | `common/normalizer.py` placeholder |

### Known TODOs

- Complete GAE advantage calculation integration in `finetune_ppo_trainer.py`
- Implement `common/normalizer.py` (currently uses normalizer from dataset)
- Add entropy bonus to PPO loss for exploration
- Complete DDIM recording support in diffusion policy
- Add more environment support beyond PushT

## References

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/) - Chi et al., RSS 2023
- [Genesis: A Universal and Generative Physics Engine](https://genesis-world.readthedocs.io/)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., NeurIPS 2020
- [DPPO: Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588) - Ren et al., 2024

## License

See LICENSE file for details.
