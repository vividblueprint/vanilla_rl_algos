"""CartPole environment construction."""

import gymnasium as gym


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create CartPole-v1, optionally rendering frames to a human window."""
    return gym.make("CartPole-v1", render_mode=render_mode)


def student_observation(state):
    """Expose cart position and pole angle; hide both velocities from the student."""
    return state[[0, 2]]
