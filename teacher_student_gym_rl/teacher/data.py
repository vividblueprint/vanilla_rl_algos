"""Collect rollout states and action logits from a trained teacher."""

import numpy as np
import torch

from teacher_student_gym_rl.env import make_env
from teacher_student_gym_rl.teacher.agent import PPOAgent


def collect_teacher_data(
    teacher: PPOAgent, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collect teacher-visited states and its action logits for each state."""
    env = make_env()
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    try:
        state, _ = env.reset(seed=seed)
        while len(states) < sample_count:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=teacher.device)
            with torch.no_grad():
                teacher_logits, _ = teacher.network(state_tensor)
            probabilities = torch.softmax(teacher_logits, dim=-1).cpu().numpy()
            states.append(np.asarray(state, dtype=np.float32).copy())
            logits.append(teacher_logits.cpu().numpy().copy())
            action = int(rng.choice(len(probabilities), p=probabilities))
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
    finally:
        env.close()
    return np.stack(states), np.stack(logits)
