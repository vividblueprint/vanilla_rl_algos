"""Hyperparameters for privileged teacher and student training."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivilegedLearningConfig:
    teacher_timesteps: int = 100_000
    sample_count: int = 20_000
    student_steps: int = 2_000
    seed: int = 7
    learning_rate: float = 1e-3
    temperature: float = 2.0
    batch_size: int = 256
    student_hidden_size: int = 32


DEFAULT_CONFIG = PrivilegedLearningConfig()
