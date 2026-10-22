"""SAC policy and Q-function networks for continuous actions."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """Gaussian policy squashed by tanh and scaled to the environment bounds."""

    def __init__(
        self,
        state_size: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        action_size = int(action_low.numel())
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def distribution_parameters(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-20.0, 2.0)
        return mean, log_std.exp()

    def sample(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample bounded actions and calculate their corrected log probability."""
        mean, std = self.distribution_parameters(state)
        distribution = Normal(mean, std)
        pre_tanh = mean if deterministic else distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        if deterministic:
            return action, None

        # Change-of-variables correction for tanh, plus the scale from [-1, 1]
        # into the environment's action bounds.
        log_probability = distribution.log_prob(pre_tanh)
        log_probability -= torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_probability -= torch.log(self.action_scale)
        return action, log_probability.sum(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    """Estimate Q(state, action) for a continuous state-action pair."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([state, action], dim=-1))
