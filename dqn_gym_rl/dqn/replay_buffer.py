"""Experience data structure and replay memory for DQN."""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """One observed transition: state, action, reward, next state, terminal."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayBuffer:
    """Bounded memory that lets us learn from shuffled past experiences."""

    def __init__(self, capacity: int) -> None:
        self.items: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.items, batch_size)

    def __len__(self) -> int:
        return len(self.items)
