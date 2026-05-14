from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Bot(ABC):
    def __init__(self, name: str = "unknown"):
        self.name = name

    @abstractmethod
    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        """Return the next action for the current board state."""

    def select_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        # Backward-compatible alias for existing environment/tests.
        return self.next_action(board, player, rng)

