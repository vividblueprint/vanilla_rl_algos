"""Train a PPO actor-critic on CartPole-v1."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ppo.agent import PPOAgent
from ppo.config import DEFAULT_CONFIG, PPOConfig
from ppo.env import make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "ppo" / "checkpoints" / "cartpole_ppo.pt"


def train(
    total_timesteps: int = DEFAULT_CONFIG.total_timesteps,
    seed: int = DEFAULT_CONFIG.seed,
    save_path: Path = DEFAULT_CHECKPOINT,
    config: PPOConfig = DEFAULT_CONFIG,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    agent = PPOAgent(
        state_size=int(env.observation_space.shape[0]),
        action_count=int(env.action_space.n),
        config=config,
    )

    state, _ = env.reset(seed=seed)
    episode_return = 0.0
    recent_returns: deque[float] = deque(maxlen=20)
    total_steps = 0

    try:
        while total_steps < total_timesteps:
            batch_size = min(config.rollout_steps, total_timesteps - total_steps)
            states: list[np.ndarray] = []
            actions: list[int] = []
            old_log_probs: list[float] = []
            values: list[float] = []
            next_values: list[float] = []
            rewards: list[float] = []
            episode_ends: list[bool] = []
            terminated_flags: list[bool] = []

            for _ in range(batch_size):
                action, log_probability, value = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_end = terminated or truncated

                states.append(np.asarray(state, dtype=np.float32).copy())
                actions.append(action)
                old_log_probs.append(log_probability)
                values.append(value)
                rewards.append(float(reward))
                episode_ends.append(episode_end)
                terminated_flags.append(terminated)
                # Bootstrap through time limits, but not through true terminals.
                next_values.append(0.0 if terminated else agent.value(next_state))

                episode_return += reward
                total_steps += 1
                if episode_end:
                    recent_returns.append(episode_return)
                    episode_return = 0.0
                    state, _ = env.reset()
                else:
                    state = next_state

            advantages, value_targets = agent.compute_advantages(
                rewards, values, next_values, episode_ends, terminated_flags
            )
            metrics = agent.update(
                states=np.stack(states),
                actions=np.asarray(actions, dtype=np.int64),
                old_log_probabilities=np.asarray(old_log_probs, dtype=np.float32),
                advantages=advantages,
                value_targets=value_targets,
            )

            if total_steps % (10 * config.rollout_steps) < batch_size:
                mean_return = np.mean(recent_returns) if recent_returns else float("nan")
                print(
                    f"steps {total_steps:6d} | mean return (last 20) {mean_return:6.1f} | "
                    f"policy loss {metrics['policy_loss']:7.3f} | "
                    f"value loss {metrics['value_loss']:7.3f} | entropy {metrics['entropy']:.3f}"
                )
    finally:
        env.close()

    agent.save(save_path)
    print(f"Saved policy to {save_path} (device: {agent.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_CONFIG.total_timesteps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--save", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, seed=args.seed, save_path=args.save)


if __name__ == "__main__":
    main()
