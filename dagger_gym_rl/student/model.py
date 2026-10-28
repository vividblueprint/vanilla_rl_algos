"""Compact student policy for CartPole imitation learning."""

import torch
from torch import nn


class StudentPolicy(nn.Module):
    def __init__(self, state_size: int, action_count: int, hidden_size: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
