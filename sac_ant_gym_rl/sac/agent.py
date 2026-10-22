"""Soft Actor-Critic networks, entropy tuning, and gradient updates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from sac.config import SACConfig
from sac.model import GaussianActor, QNetwork
from sac.replay_buffer import ReplayBuffer, Transition


class SACAgent:
    """SAC with twin critics, target critics, and learned entropy temperature."""

    def __init__(
        self,
        state_size: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: SACConfig,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        action_size = int(action_low_tensor.numel())

        self.actor = GaussianActor(
            state_size, action_low_tensor, action_high_tensor, config.hidden_size
        ).to(self.device)
        self.q1 = QNetwork(state_size, action_size, config.hidden_size).to(self.device)
        self.q2 = QNetwork(state_size, action_size, config.hidden_size).to(self.device)
        self.target_q1 = QNetwork(state_size, action_size, config.hidden_size).to(self.device)
        self.target_q2 = QNetwork(state_size, action_size, config.hidden_size).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.learning_rate
        )

        target_entropy = config.target_entropy if config.target_entropy is not None else -float(action_size)
        self.target_entropy = torch.tensor(target_entropy, dtype=torch.float32, device=self.device)
        self.log_alpha = torch.tensor(
            np.log(config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return a bounded policy action as a NumPy array."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)
        return action.cpu().numpy()

    def remember(self, transition: Transition) -> None:
        self.replay.add(transition)

    def update(self, batch_size: int) -> dict[str, float]:
        """Update critics, actor, entropy temperature, and target critics."""
        batch = self.replay.sample(batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([t.action for t in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            [t.terminated for t in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Bellman target uses the smaller target critic and includes the entropy
        # bonus, which values both reward and future policy randomness.
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q = torch.minimum(
                self.target_q1(next_states, next_actions), self.target_q2(next_states, next_actions)
            )
            target_q = rewards + self.config.gamma * (1.0 - terminated) * (
                next_q - self.alpha.detach() * next_log_probs
            )

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        critic_loss = 0.5 * (
            nn.functional.mse_loss(current_q1, target_q)
            + nn.functional.mse_loss(current_q2, target_q)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic weights while preserving gradients through Q(s, a) to a.
        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        sampled_actions, log_probs = self.actor.sample(states)
        actor_q = torch.minimum(
            self.q1(states, sampled_actions), self.q2(states, sampled_actions)
        )
        actor_loss = (self.alpha.detach() * log_probs - actor_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.q1.requires_grad_(True)
        self.q2.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (log_probs.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.lerp_(param, self.config.tau)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.lerp_(param, self.config.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(weights)
