"""PPO action selection, generalized advantage estimation, and updates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from ppo.config import PPOConfig
from ppo.model import ActorCritic


class PPOAgent:
    """Actor-critic learner that updates a policy with PPO's clipped objective."""

    def __init__(
        self,
        state_size: int,
        action_count: int,
        config: PPOConfig,
        *,
        hidden_size: int = 128,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_size, action_count, hidden_size).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=config.learning_rate)

    def act(self, state: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        """Return action, its log probability, and V(state) for rollout storage."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            logits, value = self.network(state_tensor)
            distribution = Categorical(logits=logits)
            action = logits.argmax() if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)
        return int(action.item()), float(log_probability.item()), float(value.item())

    def value(self, state: np.ndarray) -> float:
        """Estimate V(state), used to bootstrap the end of a rollout."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            _, estimate = self.network(state_tensor)
            return float(estimate.item())

    def compute_advantages(
        self,
        rewards: list[float],
        values: list[float],
        next_values: list[float],
        episode_ends: list[bool],
        terminated: list[bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate GAE advantages and value targets for one rollout."""
        cfg = self.config
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0

        for index in reversed(range(len(rewards))):
            # A time-limit truncation still bootstraps from V(next_state); a true
            # terminal state has no future value. Both end recursion across episodes.
            bootstrap_mask = 1.0 - float(terminated[index])
            continuation_mask = 1.0 - float(episode_ends[index])
            delta = rewards[index] + cfg.gamma * bootstrap_mask * next_values[index] - values[index]
            gae = delta + cfg.gamma * cfg.gae_lambda * continuation_mask * gae
            advantages[index] = gae

        value_targets = advantages + np.asarray(values, dtype=np.float32)
        return (
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            torch.as_tensor(value_targets, dtype=torch.float32, device=self.device),
        )

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probabilities: np.ndarray,
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> dict[str, float]:
        """Reuse one rollout for several minibatch gradient steps."""
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(
            old_log_probabilities, dtype=torch.float32, device=self.device
        )
        advantages = advantages.detach()
        value_targets = value_targets.detach()

        # Normalize advantages across the full rollout for steadier policy updates.
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        sample_count = len(states_tensor)
        metrics: dict[str, float] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        update_count = 0

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                logits, predicted_values = self.network(states_tensor[indices])
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions_tensor[indices])
                entropy = distribution.entropy().mean()

                probability_ratio = (new_log_probs - old_log_probs[indices]).exp()
                batch_advantages = advantages[indices]
                unclipped = probability_ratio * batch_advantages
                clipped = torch.clamp(
                    probability_ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * nn.functional.mse_loss(predicted_values, value_targets[indices])
                total_loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                update_count += 1

        return {key: value / update_count for key, value in metrics.items()}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(weights)
