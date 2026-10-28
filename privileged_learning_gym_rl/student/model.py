"""Policy network that consumes only the student's partial observation."""

import torch
from torch import nn


class StudentPolicy(nn.Module):
    """Map cart position and pole angle to action logits."""

    def __init__(self, observation_size: int = 2, action_count: int = 2, hidden_size: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)
