"""Training settings for recurrent PPO."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrentPPOConfig:
    total_timesteps: int = 100_000
    seed: int = 7
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 512
    update_epochs: int = 5
    hidden_size: int = 128


DEFAULT_CONFIG = RecurrentPPOConfig()
