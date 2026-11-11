"""CartPole environment and partial observation used by the GRU policy."""

import gymnasium as gym
import numpy as np


def make_env(render_mode: str | None = None) -> gym.Env:
    return gym.make("CartPole-v1", render_mode=render_mode)


def policy_observation(state: np.ndarray) -> np.ndarray:
    """Expose cart position and pole angle; leave velocities hidden."""
    return np.asarray(state[[0, 2]], dtype=np.float32)
