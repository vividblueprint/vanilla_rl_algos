# Policy Gradients on Gymnasium CartPole

This project sits beside `dqn_gym_rl` and uses the shared UV environment from
the repository root. It implements **REINFORCE**, a simple Monte Carlo policy
gradient algorithm, without using a prebuilt RL trainer.

## Files

- `policy_gradient/model.py` maps a state to action logits.
- `policy_gradient/agent.py` samples actions and updates the policy from a
  completed episode's discounted reward-to-go.
- `policy_gradient/env.py` creates CartPole-v1.
- `policy_gradient/config.py` holds the default hyperparameters.
- `train.py` collects episodes and performs one update per episode.
- `play.py` loads the saved policy and runs it with a visible window.

From the repository root:

```bash
uv run pg_gym_rl/train.py
uv run pg_gym_rl/play.py
```

Training options:

```bash
uv run pg_gym_rl/train.py --episodes 800 --seed 7
uv run pg_gym_rl/play.py --deterministic
```

The default checkpoint is `pg_gym_rl/policy_gradient/checkpoints/cartpole_reinforce.pt`.

## The policy-gradient idea

DQN learns action values and chooses actions from those values. Policy gradient
directly learns a policy, written $\pi_\theta(action | state)$: a probability
distribution over actions for each state. The network outputs logits, and a
Categorical distribution turns them into action probabilities. Training samples
actions from that distribution, which keeps exploration part of the policy.

After an episode, calculate the discounted reward-to-go for every time step:

$G_t = r_t + \gamma*r_{t+1} + \gamma^2*r_{t+2} + ...$

The REINFORCE gradient estimate is:

$\sum_t G_t * grad(\log \pi_\theta(action_t | state_t))$

In words: increase the probability of actions followed by good returns, and
decrease it for actions followed by poor returns. PyTorch optimizers minimize
losses, so the code uses the negative of that expression as its loss. Returns
are normalized within the episode to reduce update scale and variance; this is
a simple variance-reduction technique, not a learned value-function baseline.

REINFORCE is on-policy and episodic: it waits until an episode ends, then uses
that episode's data for one update. Unlike DQN, this implementation has no
replay buffer and no target network. Its updates can have high variance, which
is one reason actor-critic methods add a learned value baseline later.
