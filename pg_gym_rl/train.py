"""Train a REINFORCE policy on CartPole-v1."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from policy_gradient.agent import PolicyGradientAgent
from policy_gradient.config import DEFAULT_CONFIG, PolicyGradientConfig
from policy_gradient.env import make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "policy_gradient" / "checkpoints" / "cartpole_reinforce.pt"


def train(
    episodes: int = DEFAULT_CONFIG.episodes,
    seed: int = DEFAULT_CONFIG.seed,
    save_path: Path = DEFAULT_CHECKPOINT,
    config: PolicyGradientConfig = DEFAULT_CONFIG,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    agent = PolicyGradientAgent(
        state_size=int(env.observation_space.shape[0]),
        action_count=int(env.action_space.n),
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        hidden_size=config.hidden_size,
    )
    recent_returns: deque[float] = deque(maxlen=20)

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed if episode == 1 else None)
            episode_states: list[np.ndarray] = []
            episode_actions: list[int] = []
            episode_rewards: list[float] = []
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                episode_states.append(np.asarray(state, dtype=np.float32).copy())
                episode_actions.append(action)
                episode_rewards.append(float(reward))
                state = next_state

            agent.update_episode(episode_states, episode_actions, episode_rewards)
            episode_return = sum(episode_rewards)
            recent_returns.append(episode_return)

            if episode % 10 == 0:
                print(
                    f"episode {episode:3d} | mean return (last 20) "
                    f"{np.mean(recent_returns):6.1f} | episode length {len(episode_rewards):3d}"
                )
    finally:
        env.close()

    agent.save(save_path)
    print(f"Saved policy to {save_path} (device: {agent.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG.episodes)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--save", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    train(episodes=args.episodes, seed=args.seed, save_path=args.save)


if __name__ == "__main__":
    main()
