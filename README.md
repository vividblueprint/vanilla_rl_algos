# Reinforcement Learning Gym Lab

An educational collection of reinforcement learning implementations built with
PyTorch and Gymnasium. The projects start with small discrete-action tasks and
progress toward continuous control and quadruped simulation. Each algorithm is
implemented as a separate, runnable lesson so its models, replay buffers,
updates, and training loops can be studied directly.

## Projects

| Project | Algorithm | Environment / focus |
| --- | --- | --- |
| [`dqn_gym_rl`](dqn_gym_rl/README.md) | Deep Q-Network (DQN) | CartPole; discrete actions, replay buffer, target network |
| [`pg_gym_rl`](pg_gym_rl/README.md) | REINFORCE | CartPole; Monte Carlo policy gradients |
| [`actor_critic_gym_rl`](actor_critic_gym_rl/README.md) | One-step actor-critic | CartPole; online policy and value updates |
| [`ppo_gym_rl`](ppo_gym_rl/README.md) | Proximal Policy Optimization (PPO) | CartPole; clipped updates and generalized advantage estimation |
| [`teacher_student_gym_rl`](teacher_student_gym_rl/README.md) | Policy distillation | CartPole; distills a PPO teacher into a smaller student |
| [`privileged_learning_gym_rl`](privileged_learning_gym_rl/README.md) | Privileged learning | CartPole; teacher sees full state, student acts from partial observations |
| [`domain_randomization_gym_rl`](domain_randomization_gym_rl/README.md) | Domain randomization | CartPole; trains PPO across randomized physics and evaluates on nominal physics |
| [`asymmetric_actor_critic_gym_rl`](asymmetric_actor_critic_gym_rl/README.md) | Asymmetric actor-critic | CartPole; actor uses partial observations, critic uses full state |
| [`dagger_gym_rl`](dagger_gym_rl/README.md) | DAgger | CartPole; iteratively labels student-visited states with an expert |
| [`history_policy_gym_rl`](history_policy_gym_rl/README.md) | History-based policy | CartPole; PPO acts on a window of recent partial observations |
| [`sac_gym_rl`](sac_gym_rl/README.md) | Soft Actor-Critic (SAC) | Pendulum; off-policy continuous control |
| [`sac_ant_gym_rl`](sac_ant_gym_rl/README.md) | Soft Actor-Critic (SAC) | Ant; quadruped simulation with continuous joint actions |

## Setup

The repository uses one shared Python environment at its root. Install
dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Python 3.12 or newer is required. The project dependencies include PyTorch,
Gymnasium, and the MuJoCo support needed by the robotics environments.

## Train and play

Run commands from the repository root. For example, to train and then render a
CartPole DQN policy:

```bash
uv run dqn_gym_rl/train.py
uv run dqn_gym_rl/play.py
```

To train and render the Ant SAC policy:

```bash
uv run sac_ant_gym_rl/train.py
uv run sac_ant_gym_rl/play.py
```

Each project README documents its own command-line options and default
checkpoint location. Train a project before running `play.py`, unless its
checkpoint already exists. Rendered environments require a graphical display.

## Learning path

The projects are arranged to make it easier to compare key ideas:

1. **DQN** learns action values for discrete actions, using replay and a target
   network.
2. **REINFORCE** directly optimizes a stochastic policy from complete episodes.
3. **Actor-critic** adds a learned value estimate to guide policy updates.
4. **PPO** stabilizes on-policy updates with clipped objectives and GAE.
5. **SAC** uses replay, twin critics, and entropy regularization for continuous
   actions.
6. **SAC on Ant** applies continuous-action control to a simulated quadruped.

These are learning projects and simulation experiments. A successful benchmark
policy is not, by itself, ready for physical robot deployment; that requires
additional work on robot modeling, system identification, randomized
simulation, sensing, and safety.
