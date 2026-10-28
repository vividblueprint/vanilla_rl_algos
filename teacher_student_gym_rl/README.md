# Teacher–student policy distillation on CartPole

This lesson trains a PPO teacher, records its action preferences on states from
teacher rollouts, and trains a smaller policy-only student to match them. Both
use the shared UV environment from the repository root. The student has one
32-unit hidden layer; the PPO teacher has a 128-unit actor-critic network.

## Files

- `config.py` holds the distillation and student hyperparameters.
- `env.py` creates CartPole-v1.
- `teacher/train.py` trains the PPO teacher using the existing PPO lesson.
- `teacher/data.py` collects teacher-visited states and action logits.
- `student/model.py` defines the compact student policy.
- `student/agent.py` handles distillation updates and checkpoints.
- `distill.py` coordinates teacher training, data collection, and student fitting.
- `play.py` runs the saved student policy.

## Run the lesson

From the repository root:

```bash
uv run python -m teacher_student_gym_rl.distill
uv run python -m teacher_student_gym_rl.play
```

Adjust the training budget and data size with:

```bash
uv run python -m teacher_student_gym_rl.distill --teacher-timesteps 150000 --samples 30000 --student-steps 3000 --seed 7
```

The default checkpoints are `teacher_student_gym_rl/checkpoints/cartpole_teacher.pt`
and `teacher_student_gym_rl/checkpoints/cartpole_student.pt`.

## What the student learns

The teacher produces a distribution over actions for each state. The student
uses a temperature-scaled KL-divergence loss to match that distribution:

```text
L = temperature^2 * KL(teacher_distribution || student_distribution)
```

Matching probabilities gives the student more information than copying only
the teacher's most likely action. The sampled states come from teacher
rollouts, so the student learns on states the teacher encounters. Distillation
does not guarantee identical returns: the smaller model has less capacity and
sees only a finite set of states.
