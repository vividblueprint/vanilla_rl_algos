# Domain randomization on CartPole

This lesson trains PPO while changing CartPole's physics at every episode
reset. Gravity, cart mass, pole mass, pole half-length, and applied force are
sampled from configurable ranges. At evaluation, the saved policy runs in the
standard CartPole-v1 environment. Training across varied dynamics can help a
policy handle changes in the system, though it does not guarantee robustness to
every unseen environment.

## Files

- `config.py` defines the training budget, seed, and physics ranges.
- `env.py` implements randomized resets and the nominal evaluation environment.
- `train.py` configures the ranges and reuses the PPO training loop.
- `play.py` runs the saved policy on standard CartPole.

## Run

From the repository root:

```bash
uv run python -m domain_randomization_gym_rl.train
uv run python -m domain_randomization_gym_rl.play
```

Choose a shorter run with:

```bash
uv run python -m domain_randomization_gym_rl.train --timesteps 30000 --seed 7
```

The default checkpoint is `domain_randomization_gym_rl/checkpoints/cartpole_domain_randomized_ppo.pt`.
