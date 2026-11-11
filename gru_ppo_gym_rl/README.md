# Recurrent PPO with a GRU on CartPole

This lesson uses the same partially observed CartPole setup as
`recurrent_ppo_gym_rl`: the policy receives position and pole angle, while
velocities are hidden. A GRU carries one hidden state tensor from step to step,
so the policy can use information from earlier observations. The state resets
at episode boundaries. PPO updates unroll complete rollout sequences and apply
episode-boundary masks.

## Files

- `config.py` contains PPO and GRU settings.
- `env.py` creates CartPole and selects the policy's partial observation.
- `agent/model.py` defines the recurrent actor-critic network.
- `agent/ppo.py` implements memory-aware action selection, GAE, and PPO updates.
- `train.py` gathers sequential rollouts and updates the policy.
- `play.py` carries GRU memory while running episodes.

## Run

From the repository root:

```bash
uv run python -m gru_ppo_gym_rl.train
uv run python -m gru_ppo_gym_rl.play
```

Short run:

```bash
uv run python -m gru_ppo_gym_rl.train --timesteps 20000
```

The default checkpoint is `gru_ppo_gym_rl/agent/checkpoints/cartpole_gru_ppo.pt`.
