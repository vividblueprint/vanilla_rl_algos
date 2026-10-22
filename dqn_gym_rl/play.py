"""Run a saved DQN policy in a visible CartPole window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dqn.agent import DQNAgent
from dqn.env import make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "dqn" / "checkpoints" / "cartpole_dqn.pt"


def play(checkpoint: Path = DEFAULT_CHECKPOINT, episodes: int = 5, delay: float = 0.02) -> None:
    env = make_env(render_mode="human")
    agent = DQNAgent(
        state_size=int(env.observation_space.shape[0]),
        action_count=int(env.action_space.n),
    )
    agent.load(checkpoint)

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(state)  # epsilon=0: greedy evaluation
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                if delay > 0:
                    time.sleep(delay)
            print(f"episode {episode}: return {episode_return:.0f}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.02)
    args = parser.parse_args()
    play(checkpoint=args.checkpoint, episodes=args.episodes, delay=args.delay)


if __name__ == "__main__":
    main()
