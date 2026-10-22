"""Gymnasium environment construction for the PPO example."""

import gymnasium as gym


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create CartPole-v1, optionally rendering frames to a human window."""
    return gym.make("CartPole-v1", render_mode=render_mode)
