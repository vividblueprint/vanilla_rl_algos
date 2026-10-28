"""Train PPO on CartPole with episode-by-episode physics randomization."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ppo_gym_rl.ppo.config import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from ppo_gym_rl.train import train as train_ppo

from domain_randomization_gym_rl.config import DEFAULT_CONFIG, DomainRandomizationConfig
from domain_randomization_gym_rl.env import make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "cartpole_domain_randomized_ppo.pt"


def train(
    config: DomainRandomizationConfig = DEFAULT_CONFIG,
    save_path: Path = DEFAULT_CHECKPOINT,
) -> None:
    """Train a PPO policy while drawing new physics parameters each episode."""
    ppo_config = replace(
        PPO_DEFAULT_CONFIG,
        total_timesteps=config.total_timesteps,
        seed=config.seed,
    )
    train_ppo(
        total_timesteps=config.total_timesteps,
        seed=config.seed,
        save_path=save_path,
        config=ppo_config,
        env_factory=lambda: make_env(config=config, seed=config.seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_CONFIG.total_timesteps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--save", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    train(replace(DEFAULT_CONFIG, total_timesteps=args.timesteps, seed=args.seed), args.save)


if __name__ == "__main__":
    main()
