"""Run the actor using only its partial CartPole observation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from asymmetric_actor_critic_gym_rl.actor_critic.agent import AsymmetricPPOAgent
from asymmetric_actor_critic_gym_rl.config import DEFAULT_CONFIG
from asymmetric_actor_critic_gym_rl.env import actor_observation, make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "actor_critic" / "checkpoints" / "cartpole_asymmetric_ppo.pt"


def play(checkpoint: Path = DEFAULT_CHECKPOINT, episodes: int = 5, delay: float = 0.02) -> None:
    env = make_env(render_mode="human")
    agent = AsymmetricPPOAgent(2, int(env.observation_space.shape[0]), int(env.action_space.n), DEFAULT_CONFIG)
    agent.load(checkpoint)
    agent.network.eval()
    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _, _ = agent.act(actor_observation(state), state, deterministic=True)
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
    play(args.checkpoint, args.episodes, args.delay)


if __name__ == "__main__":
    main()
