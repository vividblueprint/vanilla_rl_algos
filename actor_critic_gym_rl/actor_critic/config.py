"""Default hyperparameters for one-step actor-critic on CartPole."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActorCriticConfig:
    episodes: int = 1_000
    seed: int = 7
    learning_rate: float = 1e-3
    gamma: float = 0.99
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 128


DEFAULT_CONFIG = ActorCriticConfig()
