"""Shared actor-critic network: action logits plus a state-value estimate."""

import torch
from torch import nn


class ActorCritic(nn.Module):
    """Estimate policy logits pi(a|s) and state value V(s) from one state."""

    def __init__(self, state_size: int, action_count: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
