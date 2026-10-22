"""Default settings for the REINFORCE CartPole lesson."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyGradientConfig:
    episodes: int = 600
    seed: int = 7
    learning_rate: float = 1e-2
    gamma: float = 0.99
    hidden_size: int = 128


DEFAULT_CONFIG = PolicyGradientConfig()
