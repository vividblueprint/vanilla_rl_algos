"""Run the GRU policy while carrying hidden state across CartPole steps."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from gru_ppo_gym_rl.agent.ppo import GRUPPOAgent
from gru_ppo_gym_rl.config import DEFAULT_CONFIG
from gru_ppo_gym_rl.env import make_env, policy_observation


DEFAULT_CHECKPOINT = Path(__file__).parent / "agent" / "checkpoints" / "cartpole_gru_ppo.pt"


def play(checkpoint: Path = DEFAULT_CHECKPOINT, episodes: int = 5, delay: float = 0.02) -> None:
    env = make_env(render_mode="human")
    agent = GRUPPOAgent(2, int(env.action_space.n), DEFAULT_CONFIG)
    agent.load(checkpoint)
    agent.network.eval()
    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            hidden = agent.initial_hidden()
            episode_return = 0.0
            terminated = truncated = False
            first_step = True
            while not (terminated or truncated):
                action, _, _, hidden = agent.act(
                    policy_observation(state), hidden, episode_start=first_step, deterministic=True
                )
                first_step = False
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
