"""CartPole with physical parameters randomized at every episode reset."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from domain_randomization_gym_rl.config import DEFAULT_CONFIG, DomainRandomizationConfig


class RandomizedCartPole(gym.Wrapper):
    """Sample CartPole dynamics per episode while preserving its observation space."""

    def __init__(
        self,
        env: gym.Env,
        config: DomainRandomizationConfig = DEFAULT_CONFIG,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cartpole = self.unwrapped
        cfg = self.config
        cartpole.gravity = float(self.rng.uniform(*cfg.gravity))
        cartpole.masscart = float(self.rng.uniform(*cfg.cart_mass))
        cartpole.masspole = float(self.rng.uniform(*cfg.pole_mass))
        cartpole.length = float(self.rng.uniform(*cfg.pole_half_length))
        cartpole.force_mag = float(self.rng.uniform(*cfg.force_magnitude))
        cartpole.total_mass = cartpole.masscart + cartpole.masspole
        cartpole.polemass_length = cartpole.masspole * cartpole.length
        return self.env.reset(seed=seed, options=options)


def make_env(
    config: DomainRandomizationConfig = DEFAULT_CONFIG,
    seed: int | None = None,
) -> gym.Env:
    """Create a CartPole environment that redraws its physics on reset."""
    return RandomizedCartPole(gym.make("CartPole-v1"), config=config, seed=seed)


def make_nominal_env(render_mode: str | None = None) -> gym.Env:
    """Create the standard, unrandomized CartPole environment for evaluation."""
    return gym.make("CartPole-v1", render_mode=render_mode)
