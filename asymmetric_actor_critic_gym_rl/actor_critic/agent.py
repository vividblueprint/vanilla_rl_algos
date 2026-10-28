"""PPO agent with an actor on partial observations and critic on full states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from asymmetric_actor_critic_gym_rl.actor_critic.model import AsymmetricActorCritic
from asymmetric_actor_critic_gym_rl.config import AsymmetricPPOConfig


class AsymmetricPPOAgent:
    def __init__(self, actor_obs_size: int, state_size: int, action_count: int,
                 config: AsymmetricPPOConfig, hidden_size: int = 128):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = AsymmetricActorCritic(actor_obs_size, state_size, action_count, hidden_size).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=config.learning_rate)

    def act(self, actor_observation: np.ndarray, state: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            actor_tensor = torch.as_tensor(actor_observation, dtype=torch.float32, device=self.device)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            logits, value = self.network(actor_tensor, state_tensor)
            distribution = Categorical(logits=logits)
            action = logits.argmax() if deterministic else distribution.sample()
            return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return float(self.network.critic(state_tensor).item())

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

    def update(self, actor_observations, states, actions, old_log_probs, advantages, value_targets):
        actor_obs = torch.as_tensor(actor_observations, dtype=torch.float32, device=self.device)
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages = (advantages.detach() - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        value_targets = value_targets.detach()
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(len(actor_obs), device=self.device)
            for start in range(0, len(actor_obs), self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size]
                logits, values = self.network(actor_obs[indices], states_tensor[indices])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions_tensor[indices])
                ratio = (log_probs - old_log_probs[indices]).exp()
                batch_adv = advantages[indices]
                clipped = ratio.clamp(1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_adv
                policy_loss = -torch.minimum(ratio * batch_adv, clipped).mean()
                value_loss = 0.5 * nn.functional.mse_loss(values, value_targets[indices])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                updates += 1
        return {key: value / updates for key, value in metrics.items()}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
