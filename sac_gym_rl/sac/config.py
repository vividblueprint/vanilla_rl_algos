"""Default SAC hyperparameters for the Pendulum lesson."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SACConfig:
    total_timesteps: int = 100_000
    seed: int = 7
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.2
    target_entropy: float | None = None
    replay_capacity: int = 100_000
    batch_size: int = 256
    learning_starts: int = 5_000
    hidden_size: int = 256


DEFAULT_CONFIG = SACConfig()
