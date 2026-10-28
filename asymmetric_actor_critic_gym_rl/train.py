"""Train asymmetric PPO on CartPole-v1."""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from asymmetric_actor_critic_gym_rl.actor_critic.agent import AsymmetricPPOAgent
from asymmetric_actor_critic_gym_rl.config import DEFAULT_CONFIG, AsymmetricPPOConfig
from asymmetric_actor_critic_gym_rl.env import actor_observation, make_env


DEFAULT_CHECKPOINT = Path(__file__).parent / "actor_critic" / "checkpoints" / "cartpole_asymmetric_ppo.pt"


def train(config: AsymmetricPPOConfig = DEFAULT_CONFIG, save_path: Path = DEFAULT_CHECKPOINT) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = make_env()
    agent = AsymmetricPPOAgent(
        actor_obs_size=2,
        state_size=int(env.observation_space.shape[0]),
        action_count=int(env.action_space.n),
        config=config,
    )
    state, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    recent_returns: deque[float] = deque(maxlen=20)
    total_steps = 0
    try:
        while total_steps < config.total_timesteps:
            batch_size = min(config.rollout_steps, config.total_timesteps - total_steps)
            actor_observations, states, actions = [], [], []
            old_log_probs, values, next_values = [], [], []
            rewards, episode_ends, terminated_flags = [], [], []
            for _ in range(batch_size):
                visible = actor_observation(state)
                action, log_prob, value = agent.act(visible, state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                end = terminated or truncated
                actor_observations.append(np.asarray(visible, dtype=np.float32).copy())
                states.append(np.asarray(state, dtype=np.float32).copy())
                actions.append(action)
                old_log_probs.append(log_prob)
                values.append(value)
                rewards.append(float(reward))
                episode_ends.append(end)
                terminated_flags.append(terminated)
                next_values.append(0.0 if terminated else agent.value(next_state))
                episode_return += reward
                total_steps += 1
                if end:
                    recent_returns.append(episode_return)
                    episode_return = 0.0
                    state, _ = env.reset()
                else:
                    state = next_state

            advantages, targets = agent.compute_advantages(
                rewards, values, next_values, episode_ends, terminated_flags
            )
            metrics = agent.update(
                np.stack(actor_observations), np.stack(states), np.asarray(actions),
                np.asarray(old_log_probs, dtype=np.float32), advantages, targets,
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
    print(f"Saved policy to {save_path} (device: {agent.device}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_CONFIG.total_timesteps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--save", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    train(replace(DEFAULT_CONFIG, total_timesteps=args.timesteps, seed=args.seed), args.save)


if __name__ == "__main__":
    main()
