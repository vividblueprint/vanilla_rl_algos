"""REINFORCE agent: sample from a policy, then update it from episode returns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from policy_gradient.model import PolicyNetwork


class PolicyGradientAgent:
    """A stochastic policy trained with the Monte Carlo REINFORCE estimate."""

    def __init__(
        self,
        state_size: int,
        action_count: int,
        *,
        learning_rate: float = 1e-2,
        gamma: float = 0.99,
        hidden_size: int = 128,
        device: torch.device | None = None,
    ) -> None:
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_size, action_count, hidden_size).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Sample from pi(a|s), or choose the most likely action for evaluation."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            logits = self.policy(state_tensor)
            if deterministic:
                return int(logits.argmax().item())
            return int(Categorical(logits=logits).sample().item())

    def update_episode(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
    ) -> float:
        """Apply one policy-gradient step using discounted reward-to-go."""
        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states, actions, and rewards must have equal lengths")
        if not rewards:
            raise ValueError("an episode must contain at least one transition")

        # G_t = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
        returns: list[float] = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = float(reward) + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        states_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Center and scale this episode's returns to reduce gradient variance.
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std(unbiased=False) + 1e-8
        )

        distribution = Categorical(logits=self.policy(states_tensor))
        selected_log_probs = distribution.log_prob(actions_tensor)

        # Gradient ascent on expected return is gradient descent on its negative.
        loss = -(selected_log_probs * returns_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(weights)
