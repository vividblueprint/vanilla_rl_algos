"""Training and physics-randomization settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainRandomizationConfig:
    total_timesteps: int = 150_000
    seed: int = 7
    gravity: tuple[float, float] = (8.0, 12.0)
    cart_mass: tuple[float, float] = (0.5, 2.0)
    pole_mass: tuple[float, float] = (0.05, 0.2)
    pole_half_length: tuple[float, float] = (0.25, 0.75)
    force_magnitude: tuple[float, float] = (8.0, 12.0)


DEFAULT_CONFIG = DomainRandomizationConfig()
