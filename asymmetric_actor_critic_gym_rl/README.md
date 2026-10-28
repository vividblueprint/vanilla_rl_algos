# Asymmetric actor-critic on CartPole

This PPO lesson uses different information for the actor and critic. The actor
chooses actions from cart position and pole angle. The critic receives the full
four-value state, including cart and pole velocities, to estimate value targets
and advantages during training. The critic is not used for action selection, so
the policy can run using only the actor's partial observation.

## Files

- `config.py` contains PPO hyperparameters.
- `env.py` creates CartPole and selects the actor's partial observation.
- `actor_critic/model.py` defines separate actor and critic networks.
- `actor_critic/agent.py` implements PPO updates with separate inputs.
- `train.py` gathers rollouts and trains both networks.
- `play.py` runs the actor using only position and angle.

## Run

From the repository root:

```bash
uv run python -m asymmetric_actor_critic_gym_rl.train
uv run python -m asymmetric_actor_critic_gym_rl.play
```

Short run example:

```bash
uv run python -m asymmetric_actor_critic_gym_rl.train --timesteps 30000 --seed 7
```

The default checkpoint is `asymmetric_actor_critic_gym_rl/actor_critic/checkpoints/cartpole_asymmetric_ppo.pt`.
