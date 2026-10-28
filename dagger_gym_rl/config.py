"""Expert, rollout, and student-training settings for DAgger."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DAggerConfig:
    expert_timesteps: int = 100_000
    seed: int = 7
    initial_expert_episodes: int = 10
    iterations: int = 10
    student_episodes_per_iteration: int = 5
    student_epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_size: int = 64


DEFAULT_CONFIG = DAggerConfig()
