"""Gymnasium environment construction for the SAC example."""

import gymnasium as gym


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create Ant-v5, a continuous-action quadruped locomotion task."""
    return gym.make("Ant-v5", render_mode=render_mode)
