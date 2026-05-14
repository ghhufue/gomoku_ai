from __future__ import annotations

import numpy as np

from bots.base import Bot


class RandomBot(Bot):
    def __init__(self, name: str = "random"):
        super().__init__(name=name)

    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        del player
        empties = np.flatnonzero(board.reshape(-1) == 0)
        return int(rng.choice(empties))

