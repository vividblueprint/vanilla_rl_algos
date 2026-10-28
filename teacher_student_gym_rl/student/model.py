"""Compact policy network used to imitate the PPO teacher."""

import torch
from torch import nn


class StudentPolicy(nn.Module):
    """A small policy-only network that returns action logits."""

    def __init__(self, state_size: int, action_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
