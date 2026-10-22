# DQN on Gymnasium CartPole

This folder is the DQN project. It uses the repository-root UV environment and
starts with a discrete, small problem so we can understand deep Q-learning
before moving to continuous robot actions.

## Files

- `env.py` creates the Gymnasium environment. It knows about CartPole, not DQN.
- `model.py` defines the Q-network that approximates the action-value function.
- `replay_buffer.py` stores transitions and samples training batches.
- `agent.py` combines the networks and replay memory with action selection and
  the Bellman learning update.
- `config.py` keeps the default training and DQN hyperparameters together.
- `train.py` collects experience, trains the agent, reports episode returns,
  and saves the learned network weights.
- `play.py` loads those weights and runs the policy with a visible window.

## Run

From the repository root:

```bash
uv run dqn_gym_rl/train.py
uv run dqn_gym_rl/play.py
```

Training options:

```bash
uv run dqn_gym_rl/train.py --episodes 600 --seed 7 --save dqn_gym_rl/dqn/checkpoints/cartpole_dqn.pt
```

The default training run writes `dqn_gym_rl/dqn/checkpoints/cartpole_dqn.pt`. The play
script expects that checkpoint, so train once before playing.

## The learning idea

Tabular Q-learning stores a separate value for each state/action pair. DQN uses
a neural network as a function approximator: `QNetwork(state)` returns one
estimated value for each available action. The agent usually chooses the
highest-valued action, but epsilon-greedy exploration sometimes chooses a
random action instead.

Each step produces a transition `(state, action, reward, next_state,
terminated)`. DQN samples old transitions from replay memory and trains the
online network's `Q(state, action)` prediction toward the Bellman target:

$reward + \gamma * \max_a Q_{target}(next_state, a)$

For a true terminal state, the future-value term is zero. Gymnasium also
reports `truncated` when an episode ends because of a time limit. The collector
ends that episode, while the learner still bootstraps from the next state.

Two networks are used: the online network is optimized each learning step;
the target network is copied from it every 500 environment steps. This slows
the movement of the targets. Replay memory randomizes which past examples are
seen together and reduces the correlation in consecutive experience.

The training log's return is the sum of rewards in an episode. In CartPole,
larger returns mean the pole stayed balanced longer. A high score is evidence
of a useful policy on this task, not evidence that the method is ready for a
robot. This DQN handles a small discrete action space; legged robot torques are
continuous, so we will later study actor-critic methods such as SAC and PPO.
