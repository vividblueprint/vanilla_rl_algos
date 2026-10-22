"""Bounded replay memory for off-policy SAC training."""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.items: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.items, batch_size)

    def __len__(self) -> int:
        return len(self.items)
