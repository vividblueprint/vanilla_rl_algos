# History-based policies on CartPole

This lesson uses explicit observation history (frame stacking). The policy sees
the last few cart positions and pole angles rather than one observation or the
environment's velocity values. Changes across frames give it clues about
motion, which can help when velocity is hidden. The stacked vector is fed to a
standard feed-forward PPO policy; this lesson does not use a recurrent network.

## Files

- `config.py` sets the history length and training budget.
- `env.py` hides velocity and stacks recent position/angle observations.
- `train.py` trains PPO on the stacked input, reusing the shared PPO loop.
- `play.py` maintains the same observation history while running the policy.

## Run

From the repository root:

```bash
uv run python -m history_policy_gym_rl.train
uv run python -m history_policy_gym_rl.play
```

Change the observation window with `--history-length`, using the same value for
training and play:

```bash
uv run python -m history_policy_gym_rl.train --history-length 6
uv run python -m history_policy_gym_rl.play --history-length 6
```

The default checkpoint is `history_policy_gym_rl/checkpoints/cartpole_history_ppo.pt`.
