"""CartPole observations for separate actor and critic inputs."""

import gymnasium as gym


def make_env(render_mode: str | None = None) -> gym.Env:
    return gym.make("CartPole-v1", render_mode=render_mode)


def actor_observation(state):
    """Actor observes cart position and pole angle; critic gets the full state."""
    return state[[0, 2]]
