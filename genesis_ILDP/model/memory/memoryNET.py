"""
MemoryNet: A latent state tracker for robotic manipulation.

Core idea:
- Maintains a temporal representation of robot state in latent space
- Updates state based on visual observations and action history
- Co-trained with visual encoder in real-time during robot operation
- Provides conditioning signal for diffusion-based action prediction

Training strategy:
- Random sampling of when to rely on memory vs. current observations
- Can train jointly with visual encoder or as separate memory-only training
- Requires sequential training from episode start (not single-timestep)

Open questions:
- Benchmarking effectiveness on tasks with visual distractions/occlusions
- Metrics: long-horizon task completion rate, robustness to lighting changes

TODO List:
1. [FEATURE] Feature-level adaptive gating mechanism:
   - Goal: Implement dimension-specific update rates for different semantic features
   - Motivation: Global information (e.g., final goal, object positions) should update slower than local/dynamic features (e.g., gripper state, contact forces)
   - Implementation ideas:
     a) Add learnable per-dimension decay rates: gate_weight = base_gate * dim_specific_decay
     b) Add manual feature grouping with different update policies (e.g., slow_features_mask, fast_features_mask)
     c) Use attention mechanism to automatically learn which dimensions require slower/faster updates
   - Need to discuss implementation approach and validate effectiveness

2. [FEATURE] Attention-based memory management:
   - Goal: Use attention mechanism for more robust memory state management over sequences
   - Motivation: Attention is naturally suited for sequential data and can better handle long-term dependencies
   - Implementation ideas:
     a) Multi-head attention over historical states to adaptively weight past memories
     b) Cross-attention between current observation and memory bank to selectively retrieve relevant information
     c) Temporal attention to model the importance decay of historical states
   - Expected benefits: Better generalization, more interpretable memory selection, robust to sequence length variations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MLP_backbone(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation, output_activation=None, normalisation=None, dropout=0.0, device=None):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        if normalisation is not None:
            layers.append(normalisation)
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx != len(dims) - 2:
                layers.append(activation)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        if output_activation is not None:
            layers.append(output_activation)
        self.MLP_model = nn.Sequential(*layers)
        if device is not None:
            self.MLP_model.to(device)

    def forward(self, inputs):
        return self.MLP_model(inputs)

class CNN1D_backbone(nn.Module):

    def __init__(self, input_dim, input_channel, hidden_channels:list, output_dim, strides:list, activation=nn.ReLU(),
                  cnn_final_activation=None, MLP_activation=nn.ReLU(), normalisation=None, dropout=0.0, kernel_size=2, device='cuda:0'):
        super().__init__()

        cnn_layers = []
        if normalisation is not None:
            cnn_layers.append(normalisation)

        channels = [input_channel] + hidden_channels
        for channel_idx in range(len(channels) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=channels[channel_idx], out_channels=channels[channel_idx + 1],
                                        kernel_size=kernel_size, stride=strides[channel_idx]))
            if channel_idx != len(channels) - 2:
                cnn_layers.append(activation)
                if dropout > 0.0:
                    cnn_layers.append(nn.Dropout(dropout))

        if cnn_final_activation is not None:
            cnn_layers.append(cnn_final_activation)
        cnn_layers.append(nn.Flatten())

        self.cnn_model = nn.Sequential(*cnn_layers)

        fc_input_dims = input_dim // int(torch.prod(torch.tensor(strides)).item())
        self.fc_model = MLP_backbone(input_dim=fc_input_dims * hidden_channels[-1], hidden_dims=[fc_input_dims * 2, fc_input_dims * 2],
                                     output_dim=output_dim, activation=MLP_activation, output_activation=MLP_activation, dropout=dropout, device=device)

        if device is not None:
            self.cnn_model.to(device)
            self.fc_model.to(device)

    def forward(self, inputs):
        cnn_outputs = self.cnn_model(inputs)
        return self.fc_model(cnn_outputs)

        
class state_world_model(nn.Module):
    """Predicts next state from current state and action sequence using CNN+MLP architecture."""
    def __init__(self, state_dim, action_horizon, action_dim, world_model_hidden_dims:list, action_chunk_model_channels:list,
                 action_chunk_model_strides:list, action_hidden_dims, device):
        super().__init__()

        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.activation = nn.ReLU()
        self.action_normalisation = nn.LayerNorm(normalized_shape=(action_horizon, action_dim))

        self.world_model = MLP_backbone(input_dim=(state_dim + action_hidden_dims), hidden_dims=world_model_hidden_dims,
                                        output_dim=state_dim, activation=self.activation, device=device)
        # CNN input: (B, action_dim, action_horizon) - action_dim as channels, action_horizon as seq_len
        self.action_chunk_model = CNN1D_backbone(input_dim=action_horizon, input_channel=action_dim,
                                                 hidden_channels=action_chunk_model_channels, output_dim=action_hidden_dims,
                                                 strides=action_chunk_model_strides, device=device)

    def forward(self, prev_state, prev_action):
        # prev_action: (B, action_horizon, action_dim)
        normalized_action = self.action_normalisation(prev_action)
        # Transpose to (B, action_dim, action_horizon) for Conv1d
        normalized_action = normalized_action.transpose(1, 2)
        action_hidden_vec = self.action_chunk_model(normalized_action)
        return self.world_model(torch.cat([prev_state, action_hidden_vec], dim=-1))


class MemoryGate(nn.Module):
    """Computes gating weights for fusing predicted state and observed image features."""
    def __init__(self, img_dim, prev_s_dim, hidden_dims:list, output_dim:int, backbone='mlp', device=None):
        super().__init__()
        self.img_dim = img_dim
        self.prev_s_dim = prev_s_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.model = None
        self.activation = nn.ReLU()
        self.backbone = backbone

        self.img_norm = nn.LayerNorm(img_dim)
        self.state_norm = nn.LayerNorm(prev_s_dim)

        if self.backbone == 'mlp':
            self.model = MLP_backbone(input_dim=(img_dim + prev_s_dim), hidden_dims=hidden_dims, output_dim=output_dim, activation=self.activation,
                                           output_activation=nn.Sigmoid(), device=device)

        if device is not None:
            self.img_norm.to(device)
            self.state_norm.to(device)

    def forward(self, imgs_vec, states):
        assert imgs_vec.shape[-1] == self.img_dim
        assert states.shape[-1] == self.prev_s_dim

        imgs_normalized = self.img_norm(imgs_vec)
        states_normalized = self.state_norm(states)

        return self.model(torch.cat([imgs_normalized, states_normalized], dim=-1))
        
        
class MemoryNet(nn.Module):
    """
    MemoryNet combines physics-based state evolution (world model) with observation-based updates (gating mechanism).

    It maintains a latent state that:
    - Evolves smoothly following physical laws via the world model
    - Adapts to dynamic task-relevant changes via gated image features
    - Ignores visual distractions (lighting, occlusions) through selective gating

    The resulting state vector can be used to condition diffusion models for action prediction.

    Note: state_dim should equal img_feature_dim for compatibility with downstream models.
    """
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.config = config
        self.device = device

        self.state_dim = config.get('state_dim', 256)
        self.img_feature_dim = config.get('img_feature_dim', 256)
        self.action_dim = config.get('action_dim', 7)
        self.action_horizon = config.get('action_horizon', 8)

        self.world_model = state_world_model(
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            world_model_hidden_dims=config.get('world_model_hidden_dims', [512, 512, 256]),
            action_chunk_model_channels=config.get('action_chunk_model_channels', [16, 32, 64]),
            action_chunk_model_strides=config.get('action_chunk_model_strides', [2, 2, 2]),
            action_hidden_dims=config.get('action_hidden_dims', 128),
            device=device
        )

        self.memory_gate = MemoryGate(
            img_dim=self.img_feature_dim,
            prev_s_dim=self.state_dim,
            hidden_dims=config.get('gate_hidden_dims', [256, 128]),
            output_dim=self.state_dim,
            backbone='mlp',
            device=device
        )

    def forward(self, img_vec, prev_state=None, prev_action=None):
        """
        Args:
            img_vec: Current image features (batch_size, img_feature_dim)
            prev_state: Previous state vector (batch_size, state_dim), None for initial timestep
            prev_action: Previous action sequence (batch_size, action_horizon, action_dim), None to skip world model

        Returns:
            state_vec: Updated state (batch_size, state_dim)

        Forward process:
            state_hat = world_model(prev_state, prev_action) if available, else prev_state
            gate = sigmoid(MLP([img_vec, state_hat]))
            state_vec = state_hat * (1 - gate) + img_vec * gate
        """
        if prev_state is None:
            state_hat = img_vec.clone()
        else:
            if prev_action is not None:
                state_hat = self.world_model(prev_state, prev_action)
            else:
                state_hat = prev_state

        input_gate = self.memory_gate(img_vec, state_hat)

        state_vec = state_hat * (1 - input_gate) + img_vec * input_gate

        return state_vec 

