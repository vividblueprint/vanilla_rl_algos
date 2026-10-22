"""Shared actor-critic network with policy and value outputs."""

import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    """Map a state to action logits and a scalar estimate of V(state)."""

    def __init__(self, state_size: int, action_count: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
