# Privileged learning on CartPole

This lesson trains a PPO teacher with access to all four CartPole state values:
cart position, cart velocity, pole angle, and pole angular velocity. The student
sees only cart position and pole angle at training and play time. The teacher's
full-state action distribution provides the learning signal for the student.

## Files

- `config.py` holds teacher and distillation settings.
- `env.py` creates CartPole and defines the student's partial observation.
- `teacher/train.py` trains the full-state PPO policy.
- `teacher/data.py` collects partial observations paired with teacher logits.
- `student/model.py` and `student/agent.py` define and train the student.
- `distill.py` coordinates teacher training and student distillation.
- `play.py` runs the student using only its partial observations.

## Run the lesson

From the repository root:

```bash
uv run python -m privileged_learning_gym_rl.distill
uv run python -m privileged_learning_gym_rl.play
```

Reduce the runtime for a quick experiment with:

```bash
uv run python -m privileged_learning_gym_rl.distill --teacher-timesteps 20000 --samples 5000 --student-steps 500
```

The default checkpoints are `privileged_learning_gym_rl/checkpoints/cartpole_full_state_teacher.pt`
and `privileged_learning_gym_rl/checkpoints/cartpole_partial_state_student.pt`.

## What is privileged

The teacher receives velocities that the student cannot observe. During training,
the teacher supplies action preferences for each state while demonstrations
record only what the student can observe. The student learns from those
full-state decisions using only position and angle. It cannot recover velocity
uniquely from one observation, so it cannot match every teacher decision. This
gap makes the information advantage explicit.
