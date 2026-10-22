# SAC on Gymnasium Ant

This project moves the SAC implementation from the Pendulum exercise to a
simulated quadruped. It uses the shared UV environment from the repository
root. Gymnasium's `Ant-v5` has an eight-dimensional continuous action for joint
torques and a 105-value observation. It is a much harder task than Pendulum.

## Files

- `sac/model.py` defines the squashed Gaussian actor and twin Q networks.
- `sac/replay_buffer.py` stores continuous-action experience.
- `sac/agent.py` contains SAC's actor, twin critics, target critics, and
  automatic entropy tuning.
- `sac/env.py` creates `Ant-v5`.
- `sac/config.py` sets the initial Ant training budget and hyperparameters.
- `train.py` collects experience and trains the agent.
- `play.py` loads and renders the trained policy.

From the repository root:

```bash
uv run sac_ant_gym_rl/train.py
uv run sac_ant_gym_rl/play.py
```

The initial training budget is 500,000 environment steps and may take a while.
Try a short run first to verify the setup:

```bash
uv run sac_ant_gym_rl/train.py --timesteps 20000
```

The default checkpoint is
`sac_ant_gym_rl/sac/checkpoints/ant_sac.pt`.

## What changed from Pendulum

The SAC update is unchanged. The environment supplies the observation size,
action size, and action bounds, so the actor and critics adapt to Ant's larger
spaces. The actor's `tanh` output is scaled to the environment's torque bounds.
The replay buffer stores only the true `terminated` flag, so time-limit
truncations still bootstrap from their final observation.

Ant's reward includes forward movement and costs for control effort. Early
training can produce low or negative returns; look for a sustained improvement
in the average episode return, and inspect rendered behavior rather than
judging from one episode. This is still a benchmark robot in simulation. A
custom robot model, randomized physics, sensor noise, and safety-aware hardware
work are later steps toward deployment.
