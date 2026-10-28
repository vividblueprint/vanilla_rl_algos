"""Run a saved PPO policy in a visible CartPole window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:  # Package import when reused by another lesson.
    from .ppo.agent import PPOAgent
    from .ppo.config import DEFAULT_CONFIG
    from .ppo.env import make_env
except ImportError:  # Direct script execution from the repository root.
    from ppo.agent import PPOAgent
    from ppo.config import DEFAULT_CONFIG
    from ppo.env import make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "ppo" / "checkpoints" / "cartpole_ppo.pt"


def play(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    episodes: int = 5,
    delay: float = 0.02,
    deterministic: bool = True,
) -> None:
    env = make_env(render_mode="human")
    agent = PPOAgent(
        state_size=int(env.observation_space.shape[0]),
        action_count=int(env.action_space.n),
        config=DEFAULT_CONFIG,
    )
    agent.load(checkpoint)

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _, _ = agent.act(state, deterministic=deterministic)
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
    parser.add_argument("--stochastic", action="store_true", help="sample actions from the policy")
    args = parser.parse_args()
    play(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        delay=args.delay,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
