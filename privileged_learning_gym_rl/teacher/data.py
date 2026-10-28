"""Collect privileged teacher logits paired with student-visible states."""

import numpy as np
import torch

from ppo_gym_rl.ppo.agent import PPOAgent
from privileged_learning_gym_rl.env import make_env, student_observation


def collect_demonstrations(
    teacher: PPOAgent, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Record partial observations and full-state teacher action logits."""
    env = make_env()
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    try:
        state, _ = env.reset(seed=seed)
        while len(observations) < sample_count:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=teacher.device)
            with torch.no_grad():
                teacher_logits, _ = teacher.network(state_tensor)
            probabilities = torch.softmax(teacher_logits, dim=-1).cpu().numpy()
            observations.append(np.asarray(student_observation(state), dtype=np.float32).copy())
            logits.append(teacher_logits.cpu().numpy().copy())
            action = int(rng.choice(len(probabilities), p=probabilities))
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
    finally:
        env.close()
    return np.stack(observations), np.stack(logits)
