"""CartPole environment construction for DAgger."""

import gymnasium as gym


def make_env(render_mode: str | None = None) -> gym.Env:
    return gym.make("CartPole-v1", render_mode=render_mode)
