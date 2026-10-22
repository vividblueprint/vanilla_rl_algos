"""One-step actor-critic action selection and learning update."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from actor_critic.config import ActorCriticConfig
from actor_critic.model import ActorCriticNetwork


class ActorCriticAgent:
    """On-policy actor-critic using a one-step TD error as its advantage."""

    def __init__(
        self,
        state_size: int,
        action_count: int,
        config: ActorCriticConfig,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(state_size, action_count, config.hidden_size).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=config.learning_rate)

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Sample from the actor policy, or choose its most likely action."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            logits, _ = self.network(state_tensor)
            if deterministic:
                return int(logits.argmax().item())
            return int(Categorical(logits=logits).sample().item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> dict[str, float]:
        """Update actor and critic from one transition using a TD advantage."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action, dtype=torch.int64, device=self.device)

        logits, value = self.network(state_tensor)
        distribution = Categorical(logits=logits)
        log_probability = distribution.log_prob(action_tensor)
        entropy = distribution.entropy()

        # The critic bootstraps from the next state, except at true terminals.
        with torch.no_grad():
            if terminated:
                next_value = torch.zeros((), dtype=torch.float32, device=self.device)
            else:
                _, next_value = self.network(next_state_tensor)
            td_target = float(reward) + self.config.gamma * next_value

        td_error = td_target - value
        # Detach the advantage so actor loss trains the policy, while critic
        # loss separately trains V(state) toward the TD target.
        actor_loss = -log_probability * td_error.detach()
        critic_loss = nn.functional.smooth_l1_loss(value, td_target)
        total_loss = (
            actor_loss
            + self.config.value_coefficient * critic_loss
            - self.config.entropy_coefficient * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "td_error": float(td_error.detach().item()),
            "entropy": float(entropy.detach().item()),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(weights)
