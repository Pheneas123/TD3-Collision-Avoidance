from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN_DIM: int = 256

def _init_mlp(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Tanh(),
        )

        max_action_t = torch.as_tensor(max_action, dtype=torch.float32).view(1, -1)
        self.register_buffer("max_action", max_action_t)

        _init_mlp(self.fc)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.fc(state)
        return out * self.max_action

class td3_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

        _init_mlp(self.q1_net)
        _init_mlp(self.q2_net)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        q1_val = self.q1_net(sa)
        q2_val = self.q2_net(sa)
        return q1_val, q2_val

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa)
