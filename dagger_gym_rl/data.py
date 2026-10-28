"""Collect expert demonstrations and student-induced states for DAgger."""

import numpy as np

from dagger_gym_rl.env import make_env
from dagger_gym_rl.student.agent import StudentAgent
from ppo_gym_rl.ppo.agent import PPOAgent


def collect_expert_demonstrations(expert: PPOAgent, episodes: int, seed: int):
    """Collect a warm-start dataset from expert-controlled episodes."""
    env = make_env()
    states, actions = [], []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            terminated = truncated = False
            while not (terminated or truncated):
                action = expert.act(state, deterministic=True)[0]
                states.append(np.asarray(state, dtype=np.float32).copy())
                actions.append(action)
                state, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()
    return np.stack(states), np.asarray(actions, dtype=np.int64)


def collect_student_states(expert: PPOAgent, student: StudentAgent, episodes: int, seed: int):
    """Run student trajectories and ask the expert to label every visited state."""
    env = make_env()
    states, labels = [], []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            terminated = truncated = False
            while not (terminated or truncated):
                states.append(np.asarray(state, dtype=np.float32).copy())
                labels.append(expert.act(state, deterministic=True)[0])
                action = student.act(state)
                state, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()
    return np.stack(states), np.asarray(labels, dtype=np.int64)
