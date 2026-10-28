"""Train a PPO teacher for the distillation lesson."""

from pathlib import Path

from ppo_gym_rl.ppo.config import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from ppo_gym_rl.ppo.config import PPOConfig
from ppo_gym_rl.train import train as train_ppo


def train_teacher(timesteps: int, seed: int, save_path: Path) -> None:
    """Train and save a PPO teacher using the shared PPO lesson."""
    config = PPOConfig(**{
        **PPO_DEFAULT_CONFIG.__dict__,
        "total_timesteps": timesteps,
        "seed": seed,
    })
    train_ppo(total_timesteps=timesteps, seed=seed, save_path=save_path, config=config)


def make_teacher_config(timesteps: int, seed: int) -> PPOConfig:
    """Return the PPO config corresponding to the requested teacher run."""
    return PPOConfig(**{
        **PPO_DEFAULT_CONFIG.__dict__,
        "total_timesteps": timesteps,
        "seed": seed,
    })
