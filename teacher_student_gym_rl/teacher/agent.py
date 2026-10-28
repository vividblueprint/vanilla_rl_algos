"""Teacher policy construction using the repository's PPO implementation."""

from ppo_gym_rl.ppo.agent import PPOAgent
from ppo_gym_rl.ppo.config import PPOConfig


def make_teacher(state_size: int, action_count: int, config: PPOConfig) -> PPOAgent:
    """Build the PPO teacher used for demonstrations and distillation."""
    return PPOAgent(state_size=state_size, action_count=action_count, config=config)
