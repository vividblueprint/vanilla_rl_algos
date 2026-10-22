"""Run a saved SAC policy in a visible Ant-v5 window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sac.agent import SACAgent
from sac.config import DEFAULT_CONFIG
from sac.env import make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "sac" / "checkpoints" / "ant_sac.pt"


def play(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    episodes: int = 5,
    delay: float = 0.02,
    deterministic: bool = True,
) -> None:
    env = make_env(render_mode="human")
    agent = SACAgent(
        state_size=int(env.observation_space.shape[0]),
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        config=DEFAULT_CONFIG,
    )
    agent.load(checkpoint)

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(state, deterministic=deterministic)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                if delay > 0:
                    time.sleep(delay)
            print(f"episode {episode}: return {episode_return:.1f}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    play(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        delay=args.delay,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
