"""Train and load a PPO expert policy."""

from pathlib import Path

from ppo_gym_rl.ppo.agent import PPOAgent
from ppo_gym_rl.ppo.config import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG, PPOConfig
from ppo_gym_rl.train import train as train_ppo


def train_expert(timesteps: int, seed: int, checkpoint: Path) -> PPOAgent:
    config = PPOConfig(**{
        **PPO_DEFAULT_CONFIG.__dict__,
        "total_timesteps": timesteps,
        "seed": seed,
    })
    # train_ppo(total_timesteps=timesteps, seed=seed, save_path=checkpoint, config=config)
    expert = PPOAgent(state_size=4, action_count=2, config=config)
    expert.load(checkpoint)
    return expert
