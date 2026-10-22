"""Neural policy that maps states to action logits."""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Return one unnormalized action score (logit) for every discrete action."""

    def __init__(self, state_size: int, action_count: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
