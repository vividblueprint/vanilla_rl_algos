# Soft Actor-Critic on Gymnasium Pendulum

This project sits beside the other RL lessons and uses the shared UV
environment from the repository root. It implements continuous-action SAC
without a prebuilt RL trainer.

## Files

- `sac/model.py` defines the squashed Gaussian actor and Q-function network.
- `sac/replay_buffer.py` stores off-policy state/action transitions.
- `sac/agent.py` updates the twin critics, actor, entropy temperature, and
  target critics.
- `sac/env.py` creates Pendulum-v1.
- `sac/config.py` contains the hyperparameters.
- `train.py` collects experience and trains SAC.
- `play.py` loads the actor and runs it in a visible window.

From the repository root:

```bash
uv run sac_gym_rl/train.py
uv run sac_gym_rl/play.py
```

Training options:

```bash
uv run sac_gym_rl/train.py --timesteps 150000 --seed 7
uv run sac_gym_rl/play.py --stochastic
```

The default checkpoint is `sac_gym_rl/sac/checkpoints/pendulum_sac.pt`.

## The SAC idea

SAC learns a stochastic continuous-action policy and two critics. The actor
outputs a Gaussian mean and standard deviation, samples with the
reparameterization trick, then applies `tanh` and rescales the action to the
environment's bounds. Its log probability includes the corrections for both
the `tanh` transform and action scaling.

The replay buffer makes SAC off-policy: transitions from older versions of the
policy can be reused. For each sampled transition, the critics fit a target
based on the smaller of two slowly updated target critics:

```text
target = reward + gamma * (min(target_Q1, target_Q2) - alpha * log_pi(next_action|next_state))
```

The minimum helps reduce overestimation. The entropy term rewards policies
that retain some randomness. The actor minimizes `alpha * log_pi - min(Q1,Q2)`,
so it prefers rewarding actions while balancing exploration. SAC tunes `alpha`
automatically toward a target entropy. Target critics use Polyak averaging,
which moves their weights a small fraction toward the learned critics after
each update.

CartPole used a small discrete action set; Pendulum has a continuous torque
action. This is a closer algorithmic step toward robot control, though it is
still a simple simulator and not a deployment-ready robot policy.
