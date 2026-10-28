"""Student distillation updates and checkpoint operations."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from privileged_learning_gym_rl.student.model import StudentPolicy


class StudentAgent:
    """Learn teacher action preferences from student-visible observations."""

    def __init__(self, hidden_size: int = 32, learning_rate: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = StudentPolicy(hidden_size=hidden_size).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

    def update(self, observations: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> float:
        student_log_probs = F.log_softmax(self.policy(observations) / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def act(self, observation: np.ndarray) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            return int(self.policy(tensor).argmax().item())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), destination)

    def load(self, path: str | Path) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
