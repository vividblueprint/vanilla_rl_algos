"""Run the student using only cart position and pole angle."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from privileged_learning_gym_rl.env import make_env, student_observation
from privileged_learning_gym_rl.student.agent import StudentAgent


DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "cartpole_partial_state_student.pt"


def play(checkpoint: Path = DEFAULT_CHECKPOINT, episodes: int = 5, delay: float = 0.02) -> None:
    env = make_env(render_mode="human")
    student = StudentAgent()
    student.load(checkpoint)
    student.policy.eval()
    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = student.act(student_observation(state))
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
