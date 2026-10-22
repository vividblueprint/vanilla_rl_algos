# PPO on Gymnasium CartPole

This project sits beside `dqn_gym_rl` and uses the shared UV environment from
the repository root. It implements the clipped form of **Proximal Policy
Optimization (PPO)** with a shared actor-critic network.

## Files

- `ppo/model.py` estimates policy logits and state value `V(s)`.
- `ppo/agent.py` selects actions, computes generalized advantages, and applies
  PPO minibatch updates.
- `ppo/env.py` creates CartPole-v1.
- `ppo/config.py` holds rollout, optimizer, and PPO hyperparameters.
- `train.py` gathers rollouts and updates the actor-critic.
- `play.py` loads the saved policy and runs it in a visible window.

From the repository root:

```bash
uv run ppo_gym_rl/train.py
uv run ppo_gym_rl/play.py
```

Training options:

```bash
uv run ppo_gym_rl/train.py --timesteps 150000 --seed 7
uv run ppo_gym_rl/play.py --stochastic
```

The default checkpoint is `ppo_gym_rl/ppo/checkpoints/cartpole_ppo.pt`.

## The PPO idea

PPO learns both a policy and a value function. The **actor** represents a
probability distribution over actions. The **critic** estimates `V(s)`, the
expected discounted reward from a state. The critic helps judge whether an
action did better or worse than expected.

The training loop first collects a rollout using the current policy. For each
step it records the state, action, old action log-probability, reward, value,
and next-state value. It then computes generalized advantage estimates (GAE):

```text
delta_t = reward_t + gamma * V(next_state) - V(state)
advantage_t = delta_t + gamma * lambda * advantage_(t+1)
```

Episode boundaries stop the advantage recursion. True terminals do not
bootstrap from a next-state value; time-limit truncations do. The critic target
is `advantage + V(state)`.

PPO compares each action's new probability with the probability used to collect
the rollout. The ratio is `new_probability / old_probability`. Its clipped
objective limits how much that ratio can move away from 1 during an update,
which helps prevent a single batch from changing the policy too sharply. PPO
reuses each rollout for several shuffled minibatch epochs, then collects fresh
data from the updated policy.

The combined loss includes policy loss, value-prediction loss, and an entropy
bonus. Entropy rewards keeping some action diversity while learning. PPO is
on-policy: it does not keep a replay buffer to train on arbitrarily old data.
