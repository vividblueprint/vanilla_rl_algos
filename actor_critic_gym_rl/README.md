# One-Step Actor-Critic on Gymnasium CartPole

This project sits beside `dqn_gym_rl`, `pg_gym_rl`, and `ppo_gym_rl`, and uses
the shared UV environment from the repository root. It implements a basic
on-policy, one-step actor-critic algorithm.

## Files

- `actor_critic/model.py` has a shared neural network with actor and critic heads.
- `actor_critic/agent.py` samples actions and updates both heads from one step.
- `actor_critic/env.py` creates CartPole-v1.
- `actor_critic/config.py` holds the hyperparameters.
- `train.py` collects episodes and updates after every environment step.
- `play.py` loads the saved network and runs it in a visible window.

From the repository root:

```bash
uv run actor_critic_gym_rl/train.py
uv run actor_critic_gym_rl/play.py
```

## The actor-critic idea

The actor is a policy `pi(a|s)`: it assigns probabilities to actions. The
critic estimates `V(s)`: expected discounted future reward from a state. At
each step, the critic compares the observed reward plus next-state value with
the current value estimate:

```text
TD target = reward + gamma * V(next_state)
TD error  = TD target - V(state)
```

The TD error is a one-step estimate of the action's advantage. If it is
positive, the action turned out better than the critic expected, so the actor
increases its probability. If it is negative, the actor decreases it. The
critic learns to predict the TD target. The code detaches the TD error in the
actor loss so the actor update does not change the critic target through that
path.

This is an on-policy algorithm: it samples from its current policy and updates
immediately from each transition. It has no replay buffer, and unlike PPO it
does not collect a rollout, estimate GAE, or clip policy updates. One-step
updates are simple, but their advantage estimates can be noisy; n-step returns,
GAE, and PPO are natural improvements to study next.
