"""Train Soft Actor-Critic on Gymnasium's quadruped Ant-v5 task."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sac.agent import SACAgent
from sac.config import DEFAULT_CONFIG, SACConfig
from sac.env import make_env
from sac.replay_buffer import Transition


DEFAULT_CHECKPOINT = Path(__file__).parent / "sac" / "checkpoints" / "ant_sac.pt"


def train(
    total_timesteps: int = DEFAULT_CONFIG.total_timesteps,
    seed: int = DEFAULT_CONFIG.seed,
    save_path: Path = DEFAULT_CHECKPOINT,
    config: SACConfig = DEFAULT_CONFIG,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    env.action_space.seed(seed)
    agent = SACAgent(
        state_size=int(env.observation_space.shape[0]),
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        config=config,
    )
    state, _ = env.reset(seed=seed)
    recent_returns: deque[float] = deque(maxlen=10)
    episode_return = 0.0
    episode = 0

    try:
        for step in range(1, total_timesteps + 1):
            if step < config.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(
                Transition(
                    state=np.asarray(state, dtype=np.float32).copy(),
                    action=np.asarray(action, dtype=np.float32).copy(),
                    reward=float(reward),
                    next_state=np.asarray(next_state, dtype=np.float32).copy(),
                    terminated=terminated,
                )
            )
            state = next_state
            episode_return += reward

            if step >= config.learning_starts and len(agent.replay) >= config.batch_size:
                metrics = agent.update(config.batch_size)
            else:
                metrics = None

            if terminated or truncated:
                episode += 1
                recent_returns.append(episode_return)
                episode_return = 0.0
                state, _ = env.reset()
                if episode % 10 == 0:
                    mean_return = np.mean(recent_returns)
                    if metrics is None:
                        print(f"step {step:6d} | episode {episode:4d} | mean return {mean_return:8.1f}")
                    else:
                        print(
                            f"step {step:6d} | episode {episode:4d} | mean return {mean_return:8.1f} | "
                            f"critic loss {metrics['critic_loss']:7.3f} | alpha {metrics['alpha']:.3f}"
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
