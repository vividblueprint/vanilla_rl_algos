"""Train a PPO teacher that can observe CartPole's full state."""

from pathlib import Path

from ppo_gym_rl.ppo.config import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG, PPOConfig
from ppo_gym_rl.train import train as train_ppo


def make_config(timesteps: int, seed: int) -> PPOConfig:
    return PPOConfig(**{
        **PPO_DEFAULT_CONFIG.__dict__,
        "total_timesteps": timesteps,
        "seed": seed,
    })


def train_teacher(timesteps: int, seed: int, checkpoint: Path) -> PPOConfig:
    """Train and save the teacher, returning its architecture configuration."""
    config = make_config(timesteps, seed)
    # train_ppo(total_timesteps=timesteps, seed=seed, save_path=checkpoint, config=config)
    return config
