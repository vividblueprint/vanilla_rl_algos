"""Student action selection and supervised training on aggregated labels."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from dagger_gym_rl.student.model import StudentPolicy


class StudentAgent:
    def __init__(self, state_size: int, action_count: int, hidden_size: int = 64, learning_rate: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = StudentPolicy(state_size, action_count, hidden_size).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

    def act(self, state: np.ndarray) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return int(self.policy(tensor).argmax().item())

    def fit(self, states: np.ndarray, actions: np.ndarray, *, epochs: int, batch_size: int) -> float:
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        losses = []
        for _ in range(epochs):
            order = torch.randperm(len(state_tensor), device=self.device)
            for start in range(0, len(state_tensor), batch_size):
                indices = order[start:start + batch_size]
                loss = F.cross_entropy(self.policy(state_tensor[indices]), action_tensor[indices])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
