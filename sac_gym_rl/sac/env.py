"""Gymnasium environment construction for the SAC example."""

import gymnasium as gym


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create Pendulum-v1, a continuous-action control task."""
    return gym.make("Pendulum-v1", render_mode=render_mode)
