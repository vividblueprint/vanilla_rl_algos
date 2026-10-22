"""Neural network used to approximate the action-value function Q(s, a)."""

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map a state vector to one estimated long-term value per action."""

    def __init__(self, state_size: int, action_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)