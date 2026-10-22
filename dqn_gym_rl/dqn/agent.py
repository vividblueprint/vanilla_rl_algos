"""DQN's function approximator, replay memory, and learning update."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from dqn.model import QNetwork
from dqn.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    """Owns the online/target networks and applies the DQN learning rule."""

    def __init__(
        self,
        state_size: int,
        action_count: int,
        *,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        replay_capacity: int = 50_000,
        device: torch.device | None = None,
    ) -> None:
        self.action_count = action_count
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(state_size, action_count).to(self.device)
        self.target_net = QNetwork(state_size, action_count).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(replay_capacity)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose randomly with probability epsilon; otherwise choose max Q."""
        if random.random() < epsilon:
            return random.randrange(self.action_count)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return int(self.online_net(state_tensor).argmax().item())

    def remember(self, transition: Transition) -> None:
        self.replay.add(transition)

    def learn(self, batch_size: int) -> float:
        """Take one gradient step on a random mini-batch of transitions."""
        batch = self.replay.sample(batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        # Gymnasium distinguishes a true terminal state from a time-limit truncation.
        terminated = torch.as_tensor([t.terminated for t in batch], dtype=torch.float32, device=self.device)

        # The network emits every action value; gather selects the action taken.
        predicted_q = self.online_net(states).gather(1, actions).squeeze(1)

        # Bellman target: r + gamma * max_a' Q_target(s', a'); no bootstrap if terminal.
        with torch.no_grad():
            best_next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * (1.0 - terminated) * best_next_q

        loss = nn.functional.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target_network(self) -> None:
        """Copy current online weights into the slowly updated target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(weights)
        self.sync_target_network()
