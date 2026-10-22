"""Default DQN and training hyperparameters for the CartPole lesson."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DQNConfig:
    episodes: int = 1400
    seed: int = 7
    learning_rate: float = 1e-3
    gamma: float = 0.99
    replay_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 1_000
    target_update_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000


DEFAULT_CONFIG = DQNConfig()
