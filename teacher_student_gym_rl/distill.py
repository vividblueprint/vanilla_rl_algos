"""Orchestrate teacher training, demonstration collection, and distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from teacher_student_gym_rl.config import DEFAULT_CONFIG, DistillationConfig
from teacher_student_gym_rl.env import make_env
from teacher_student_gym_rl.student.agent import StudentAgent
from teacher_student_gym_rl.teacher.agent import make_teacher
from teacher_student_gym_rl.teacher.data import collect_teacher_data
from teacher_student_gym_rl.teacher.train import make_teacher_config, train_teacher


ROOT = Path(__file__).parent
TEACHER_CHECKPOINT = ROOT / "checkpoints" / "cartpole_teacher.pt"
STUDENT_CHECKPOINT = ROOT / "checkpoints" / "cartpole_student.pt"


def distill(
    teacher_checkpoint: Path = TEACHER_CHECKPOINT,
    student_checkpoint: Path = STUDENT_CHECKPOINT,
    config: DistillationConfig = DEFAULT_CONFIG,
) -> None:
    """Train a PPO teacher and fit a smaller policy to its action distribution."""
    torch.manual_seed(config.seed)
    # train_teacher(config.teacher_timesteps, config.seed, teacher_checkpoint)

    env = make_env()
    try:
        state_size = int(env.observation_space.shape[0])
        action_count = int(env.action_space.n)
    finally:
        env.close()

    teacher_config = make_teacher_config(config.teacher_timesteps, config.seed)
    teacher = make_teacher(state_size, action_count, teacher_config)
    teacher.load(teacher_checkpoint)
    states, teacher_logits = collect_teacher_data(teacher, config.sample_count, config.seed + 1)

    student = StudentAgent(
        state_size,
        action_count,
        hidden_size=config.student_hidden_size,
        learning_rate=config.learning_rate,
        device=teacher.device,
    )
    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=student.device)
    teacher_tensor = torch.as_tensor(teacher_logits, dtype=torch.float32, device=student.device)

    for step in range(config.student_steps):
        indices = torch.randint(len(state_tensor), (config.batch_size,), device=student.device)
        loss = student.update(
            state_tensor[indices],
            teacher_tensor[indices],
            temperature=config.temperature,
        )
        if (step + 1) % 500 == 0 or step + 1 == config.student_steps:
            print(f"distillation step {step + 1:5d} | KL loss {loss:.5f}")

    student.save(student_checkpoint)
    print(f"Saved student policy to {student_checkpoint} (device: {student.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-timesteps", type=int, default=DEFAULT_CONFIG.teacher_timesteps)
    parser.add_argument("--samples", type=int, default=DEFAULT_CONFIG.sample_count)
    parser.add_argument("--student-steps", type=int, default=DEFAULT_CONFIG.student_steps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--teacher-save", type=Path, default=TEACHER_CHECKPOINT)
    parser.add_argument("--student-save", type=Path, default=STUDENT_CHECKPOINT)
    args = parser.parse_args()
    distill(
        teacher_checkpoint=args.teacher_save,
        student_checkpoint=args.student_save,
        config=DistillationConfig(
            teacher_timesteps=args.teacher_timesteps,
            sample_count=args.samples,
            student_steps=args.student_steps,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
