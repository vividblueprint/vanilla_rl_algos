# Recurrent PPO with an LSTM on CartPole

This lesson trains an LSTM actor-critic with PPO. The policy sees cart position
and pole angle while velocities remain hidden. It carries its LSTM hidden and
cell states across steps, allowing its action to depend on prior observations.
Both memory tensors reset when an episode ends. PPO updates unroll the rollout
sequence and apply episode-boundary masks so memory does not leak across episodes.

## Files

- `config.py` contains PPO and LSTM settings.
- `env.py` creates CartPole and selects the policy's partial observation.
- `agent/model.py` defines the recurrent actor-critic network.
- `agent/ppo.py` implements memory-aware action selection, GAE, and PPO updates.
- `train.py` gathers sequential rollouts and updates the policy.
- `play.py` carries LSTM memory while running episodes.

## Run

From the repository root:

```bash
uv run python -m recurrent_ppo_gym_rl.train
uv run python -m recurrent_ppo_gym_rl.play
```

Short run:

```bash
uv run python -m recurrent_ppo_gym_rl.train --timesteps 20000
```

The default checkpoint is `recurrent_ppo_gym_rl/agent/checkpoints/cartpole_recurrent_ppo.pt`.
