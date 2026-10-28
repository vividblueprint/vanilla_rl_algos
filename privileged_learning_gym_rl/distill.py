"""Train a full-state PPO teacher and distill it into a partial-state student."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ppo_gym_rl.ppo.agent import PPOAgent
from privileged_learning_gym_rl.config import DEFAULT_CONFIG, PrivilegedLearningConfig
from privileged_learning_gym_rl.env import make_env
from privileged_learning_gym_rl.student.agent import StudentAgent
from privileged_learning_gym_rl.teacher.data import collect_demonstrations
from privileged_learning_gym_rl.teacher.train import train_teacher


ROOT = Path(__file__).parent
TEACHER_CHECKPOINT = ROOT / "checkpoints" / "cartpole_full_state_teacher.pt"
STUDENT_CHECKPOINT = ROOT / "checkpoints" / "cartpole_partial_state_student.pt"


def distill(
    teacher_checkpoint: Path = TEACHER_CHECKPOINT,
    student_checkpoint: Path = STUDENT_CHECKPOINT,
    config: PrivilegedLearningConfig = DEFAULT_CONFIG,
) -> None:
    """Train teacher, collect demonstrations, and fit the student policy."""
    torch.manual_seed(config.seed)
    teacher_config = train_teacher(config.teacher_timesteps, config.seed, teacher_checkpoint)
    env = make_env()
    try:
        state_size = int(env.observation_space.shape[0])
        action_count = int(env.action_space.n)
    finally:
        env.close()

    teacher = PPOAgent(state_size, action_count, teacher_config)
    teacher.load(teacher_checkpoint)
    observations, logits = collect_demonstrations(teacher, config.sample_count, config.seed + 1)
    student = StudentAgent(config.student_hidden_size, config.learning_rate)
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=student.device)
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32, device=student.device)

    for step in range(config.student_steps):
        indices = torch.randint(len(obs_tensor), (config.batch_size,), device=student.device)
        loss = student.update(obs_tensor[indices], logits_tensor[indices], config.temperature)
        if (step + 1) % 500 == 0 or step + 1 == config.student_steps:
            print(f"distillation step {step + 1:5d} | KL loss {loss:.5f}")
    student.save(student_checkpoint)
    print(f"Saved partial-observation student to {student_checkpoint} (device: {student.device}).")


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
        args.teacher_save,
        args.student_save,
        PrivilegedLearningConfig(
            teacher_timesteps=args.teacher_timesteps,
            sample_count=args.samples,
            student_steps=args.student_steps,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
