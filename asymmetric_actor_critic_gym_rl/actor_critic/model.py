"""Separate policy and value networks with different observation inputs."""

import torch
from torch import nn


class AsymmetricActorCritic(nn.Module):
    """Policy maps partial actor observations; value function sees full state."""

    def __init__(self, actor_obs_size: int, state_size: int, action_count: int, hidden_size: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(actor_obs_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, actor_observations: torch.Tensor, states: torch.Tensor):
        return self.actor(actor_observations), self.critic(states).squeeze(-1)
