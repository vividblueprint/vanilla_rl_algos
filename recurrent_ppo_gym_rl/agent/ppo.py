"""PPO updates for an LSTM policy trained on sequential rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from recurrent_ppo_gym_rl.agent.model import RecurrentActorCritic
from recurrent_ppo_gym_rl.config import RecurrentPPOConfig


class RecurrentPPOAgent:
    def __init__(self, observation_size: int, action_count: int, config: RecurrentPPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = RecurrentActorCritic(observation_size, action_count, config.hidden_size).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=config.learning_rate)

    def initial_hidden(self):
        return self.network.initial_hidden(1, self.device)

    def act(self, observation: np.ndarray, hidden, *, episode_start: bool = False, deterministic: bool = False):
        if episode_start:
            hidden = self.initial_hidden()
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value, next_hidden = self.network.step(tensor, hidden)
            distribution = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item()), tuple(x.detach() for x in next_hidden)

    def value(self, observation: np.ndarray, hidden) -> float:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value, _ = self.network.step(tensor, hidden)
        return float(value.item())

    def compute_advantages(self, rewards, values, next_values, episode_ends, terminated):
        cfg = self.config
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            bootstrap = 1.0 - float(terminated[index])
            continuation = 1.0 - float(episode_ends[index])
            delta = rewards[index] + cfg.gamma * bootstrap * next_values[index] - values[index]
            gae = delta + cfg.gamma * cfg.gae_lambda * continuation * gae
            advantages[index] = gae
        targets = advantages + np.asarray(values, dtype=np.float32)
        return (torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
                torch.as_tensor(targets, dtype=torch.float32, device=self.device))

    def update(self, observations, actions, old_log_probs, episode_starts, advantages, value_targets, initial_hidden):
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        starts = torch.as_tensor(episode_starts, dtype=torch.bool, device=self.device)
        advantages = advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        value_targets = value_targets.detach()
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.config.update_epochs):
            logits, values = self.network.sequence(observations, starts, initial_hidden)
            distribution = Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)
            ratio = (log_probs - old_log_probs).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * nn.functional.mse_loss(values, value_targets)
            entropy = distribution.entropy().mean()
            loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
        return {key: value / self.config.update_epochs for key, value in metrics.items()}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
