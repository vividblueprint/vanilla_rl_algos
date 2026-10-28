"""Train a student policy with Dataset Aggregation (DAgger)."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dagger_gym_rl.config import DEFAULT_CONFIG, DAggerConfig
from dagger_gym_rl.data import collect_expert_demonstrations, collect_student_states
from dagger_gym_rl.student.agent import StudentAgent
from dagger_gym_rl.expert import train_expert


ROOT = Path(__file__).parent
EXPERT_CHECKPOINT = ROOT / "checkpoints" / "cartpole_expert.pt"
STUDENT_CHECKPOINT = ROOT / "checkpoints" / "cartpole_dagger_student.pt"


def train(
    config: DAggerConfig = DEFAULT_CONFIG,
    expert_checkpoint: Path = EXPERT_CHECKPOINT,
    student_checkpoint: Path = STUDENT_CHECKPOINT,
) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    expert = train_expert(config.expert_timesteps, config.seed, expert_checkpoint)
    student = StudentAgent(
        state_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
    )

    states, labels = collect_expert_demonstrations(expert, config.initial_expert_episodes, config.seed + 1)
    loss = student.fit(states, labels, epochs=config.student_epochs, batch_size=config.batch_size)
    print(f"warm start | examples {len(states):6d} | cross-entropy {loss:.4f}")

    for iteration in range(1, config.iterations + 1):
        new_states, new_labels = collect_student_states(
            expert,
            student,
            config.student_episodes_per_iteration,
            config.seed + iteration + 1,
        )
        states = np.concatenate((states, new_states), axis=0)
        labels = np.concatenate((labels, new_labels), axis=0)
        loss = student.fit(states, labels, epochs=config.student_epochs, batch_size=config.batch_size)
        print(
            f"DAgger round {iteration:2d}/{config.iterations} | "
            f"new states {len(new_states):5d} | aggregate {len(states):6d} | "
            f"cross-entropy {loss:.4f}"
        )

    student.save(student_checkpoint)
    print(f"Saved DAgger student to {student_checkpoint} (device: {student.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-timesteps", type=int, default=DEFAULT_CONFIG.expert_timesteps)
    parser.add_argument("--iterations", type=int, default=DEFAULT_CONFIG.iterations)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--expert-save", type=Path, default=EXPERT_CHECKPOINT)
    parser.add_argument("--student-save", type=Path, default=STUDENT_CHECKPOINT)
    args = parser.parse_args()
    train(
        replace(
            DEFAULT_CONFIG,
            expert_timesteps=args.expert_timesteps,
            iterations=args.iterations,
            seed=args.seed,
        ),
        args.expert_save,
        args.student_save,
    )


if __name__ == "__main__":
    main()
