"""Train a DQN policy on CartPole-v1 and save its neural-network weights."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.config import DEFAULT_CONFIG, DQNConfig
from dqn.env import make_env
from dqn.replay_buffer import Transition


DEFAULT_CHECKPOINT = Path(__file__).parent / "dqn" / "checkpoints" / "cartpole_dqn.pt"


def train(
    episodes: int = DEFAULT_CONFIG.episodes,
    seed: int = DEFAULT_CONFIG.seed,
    save_path: Path = DEFAULT_CHECKPOINT,
    config: DQNConfig = DEFAULT_CONFIG,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    agent = DQNAgent(
        state_size=int(env.observation_space.shape[0]),
        action_count=int(env.action_space.n),
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )

    total_steps = 0
    recent_returns: deque[float] = deque(maxlen=20)

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed if episode == 1 else None)
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                # Explore at first, then gradually rely more on learned Q-values.
                decay_fraction = min(1.0, total_steps / config.epsilon_decay_steps)
                epsilon = config.epsilon_start + decay_fraction * (
                    config.epsilon_end - config.epsilon_start
                )
                action = agent.act(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.remember(Transition(state, action, reward, next_state, terminated))
                state = next_state
                episode_return += reward
                total_steps += 1

                if total_steps >= config.warmup_steps and len(agent.replay) >= config.batch_size:
                    agent.learn(config.batch_size)

                if total_steps % config.target_update_every == 0:
                    agent.sync_target_network()

            recent_returns.append(episode_return)
            if episode % 10 == 0:
                print(
                    f"episode {episode:3d} | steps {total_steps:6d} | "
                    f"mean return (last 20) {np.mean(recent_returns):6.1f} | epsilon {epsilon:.2f}"
                )
    finally:
        env.close()

    agent.save(save_path)
    print(f"Saved weights to {save_path} (device: {agent.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG.episodes)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--save", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    train(episodes=args.episodes, seed=args.seed, save_path=args.save)


if __name__ == "__main__":
    main()
