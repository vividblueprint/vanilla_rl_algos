"""Student policy optimization and checkpoint operations."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from teacher_student_gym_rl.student.model import StudentPolicy


class StudentAgent:
    """Fit a compact policy to teacher action distributions."""

    def __init__(
        self,
        state_size: int,
        action_count: int,
        *,
        hidden_size: int = 32,
        learning_rate: float = 1e-3,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = StudentPolicy(state_size, action_count, hidden_size).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

    def update(
        self,
        states: torch.Tensor,
        teacher_logits: torch.Tensor,
        *,
        temperature: float,
    ) -> float:
        """Apply one KL-divergence update against a batch of teacher logits."""
        student_log_probs = F.log_softmax(self.policy(states) / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def act(self, state: np.ndarray) -> int:
        """Choose the highest-probability student action."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return int(self.policy(state_tensor).argmax().item())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(weights)
