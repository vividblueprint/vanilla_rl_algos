# DAgger on CartPole

DAgger (Dataset Aggregation) trains an imitation policy on the states it
actually visits. This lesson first trains a PPO expert and warm-starts a
student from expert demonstrations. In each DAgger round, the student runs its
own episodes, the expert labels every visited state with its preferred action,
and those examples are added to the dataset. The student is retrained on the
full aggregated dataset after each round.

## Files

- `config.py` holds expert, rollout, and student-training settings.
- `env.py` creates CartPole-v1.
- `expert.py` trains and loads the PPO expert.
- `data.py` collects expert demonstrations and labels student-visited states.
- `student/model.py` and `student/agent.py` define and train the student.
- `train.py` runs the warm start and DAgger aggregation rounds.
- `play.py` runs the saved student policy.

## Run

From the repository root:

```bash
uv run python -m dagger_gym_rl.train
uv run python -m dagger_gym_rl.play
```

Reduce the training workload for a quick run:

```bash
uv run python -m dagger_gym_rl.train --expert-timesteps 20000 --iterations 3
```

The default checkpoints are `dagger_gym_rl/checkpoints/cartpole_expert.pt` and
`dagger_gym_rl/checkpoints/cartpole_dagger_student.pt`.
