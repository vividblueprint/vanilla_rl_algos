"""CartPole environment exposing a fixed-length window of partial observations."""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ObservationHistory(gym.Wrapper):
    """Stack recent cart-position and pole-angle pairs into one observation."""

    def __init__(self, env: gym.Env, history_length: int = 4):
        super().__init__(env)
        if history_length < 1:
            raise ValueError("history_length must be at least 1")
        self.history_length = history_length
        self.history: deque[np.ndarray] = deque(maxlen=history_length)
        low = np.tile(np.asarray([-4.8, -0.418], dtype=np.float32), history_length)
        high = np.tile(np.asarray([4.8, 0.418], dtype=np.float32), history_length)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def _visible(observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation[[0, 2]], dtype=np.float32)

    def _stack(self) -> np.ndarray:
        return np.concatenate(tuple(self.history), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        visible = self._visible(observation)
        self.history.clear()
        for _ in range(self.history_length):
            self.history.append(visible.copy())
        return self._stack(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(self._visible(observation))
        return self._stack(), reward, terminated, truncated, info


def make_env(render_mode: str | None = None, history_length: int = 4) -> gym.Env:
    """Create a frame-stacked partial-observation CartPole environment."""
    return ObservationHistory(gym.make("CartPole-v1", render_mode=render_mode), history_length=history_length)
