"""Train PPO with an LSTM policy on partially observed CartPole."""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from recurrent_ppo_gym_rl.agent.ppo import RecurrentPPOAgent
from recurrent_ppo_gym_rl.config import DEFAULT_CONFIG, RecurrentPPOConfig
from recurrent_ppo_gym_rl.env import make_env, policy_observation


DEFAULT_CHECKPOINT = Path(__file__).parent / "agent" / "checkpoints" / "cartpole_recurrent_ppo.pt"


def train(config: RecurrentPPOConfig = DEFAULT_CONFIG, save_path: Path = DEFAULT_CHECKPOINT) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = make_env()
    agent = RecurrentPPOAgent(2, int(env.action_space.n), config)
    state, _ = env.reset(seed=config.seed)
    episode_start = True
    hidden = agent.initial_hidden()
    recent_returns: deque[float] = deque(maxlen=20)
    episode_return = 0.0
    total_steps = 0
    try:
        while total_steps < config.total_timesteps:
            batch_size = min(config.rollout_steps, config.total_timesteps - total_steps)
            initial_hidden = tuple(value.detach().clone() for value in hidden)
            observations, actions, old_log_probs, values = [], [], [], []
            next_values, rewards, episode_ends, terminated_flags, episode_starts = [], [], [], [], []
            for _ in range(batch_size):
                observation = policy_observation(state)
                episode_starts.append(episode_start)
                action, log_prob, value, hidden_after = agent.act(
                    observation, hidden, episode_start=episode_start
                )
                next_state, reward, terminated, truncated, _ = env.step(action)
                end = terminated or truncated
                observations.append(observation.copy())
                actions.append(action)
                old_log_probs.append(log_prob)
                values.append(value)
                rewards.append(float(reward))
                episode_ends.append(end)
                terminated_flags.append(terminated)
                next_values.append(
                    0.0 if terminated else agent.value(policy_observation(next_state), hidden_after)
                )
                episode_return += reward
                total_steps += 1
                if end:
                    recent_returns.append(episode_return)
                    episode_return = 0.0
                    state, _ = env.reset()
                    episode_start = True
                    hidden = agent.initial_hidden()
                else:
                    state = next_state
                    episode_start = False
                    hidden = hidden_after

            advantages, targets = agent.compute_advantages(
                rewards, values, next_values, episode_ends, terminated_flags
            )
            metrics = agent.update(
                np.stack(observations), np.asarray(actions), np.asarray(old_log_probs, dtype=np.float32),
                np.asarray(episode_starts), advantages, targets, initial_hidden,
            )
            if total_steps % (10 * config.rollout_steps) < batch_size:
                mean_return = np.mean(recent_returns) if recent_returns else float("nan")
                print(
                    f"steps {total_steps:6d} | mean return (last 20) {mean_return:6.1f} | "
                    f"policy loss {metrics['policy_loss']:7.3f} | value loss {metrics['value_loss']:7.3f}"
                )
    finally:
        env.close()
    agent.save(save_path)
    print(f"Saved recurrent policy to {save_path} (device: {agent.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_CONFIG.total_timesteps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--save", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    train(replace(DEFAULT_CONFIG, total_timesteps=args.timesteps, seed=args.seed), args.save)


if __name__ == "__main__":
    main()
