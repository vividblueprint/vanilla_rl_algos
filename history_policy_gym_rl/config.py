"""Training settings for the history-based CartPole policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryPolicyConfig:
    total_timesteps: int = 100_000
    seed: int = 7
    history_length: int = 8


DEFAULT_CONFIG = HistoryPolicyConfig()
