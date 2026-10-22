"""Train a one-step actor-critic on CartPole-v1."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from actor_critic.agent import ActorCriticAgent
from actor_critic.config import DEFAULT_CONFIG, ActorCriticConfig
from actor_critic.env import make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "actor_critic" / "checkpoints" / "cartpole_actor_critic.pt"


def train(
    episodes: int = DEFAULT_CONFIG.episodes,
    seed: int = DEFAULT_CONFIG.seed,
    save_path: Path = DEFAULT_CHECKPOINT,
    config: ActorCriticConfig = DEFAULT_CONFIG,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    agent = ActorCriticAgent(
        state_size=int(env.observation_space.shape[0]),
        action_count=int(env.action_space.n),
        config=config,
    )
    recent_returns: deque[float] = deque(maxlen=20)

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed if episode == 1 else None)
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                episode_return += reward

            recent_returns.append(episode_return)
            if episode % 10 == 0:
                print(f"episode {episode:4d} | mean return (last 20) {np.mean(recent_returns):6.1f}")
    finally:
        env.close()

    agent.save(save_path)
    print(f"Saved actor-critic to {save_path} (device: {agent.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG.episodes)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--save", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    train(episodes=args.episodes, seed=args.seed, save_path=args.save)


if __name__ == "__main__":
    main()
