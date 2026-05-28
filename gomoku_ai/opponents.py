from __future__ import annotations

import numpy as np

from bots import create_bot
from gomoku_ai.checkpoint_opponent import CheckpointOpponent


class EpisodeOpponentSampler:
    """Sample one configured bot per episode and keep it fixed for that game."""

    def __init__(self, bot_pool: tuple[str, ...], weights: tuple[float, ...]):
        if len(bot_pool) == 0:
            raise ValueError("bot_pool must not be empty")
        if len(bot_pool) != len(weights):
            raise ValueError("bot_pool and weights must have the same length")
        self.bot_pool = tuple(bot_pool)
        raw_weights = np.asarray(weights, dtype=np.float64)
        if np.any(raw_weights < 0) or raw_weights.sum() <= 0:
            raise ValueError("weights must be non-negative and sum to a positive value")
        self.weights = raw_weights / raw_weights.sum()
        self.bots = {name: create_bot(name=name) for name in self.bot_pool}
        self.current_name: str | None = None

    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        if self.current_name is None or np.count_nonzero(board) <= 1:
            self.current_name = str(rng.choice(self.bot_pool, p=self.weights))
        return int(self.bots[self.current_name].next_action(board, player, rng))


class MixedOpponentSampler:
    """Sample named bots and historical checkpoint opponents per episode."""

    def __init__(
        self,
        bot_pool: tuple[str, ...],
        bot_weights: tuple[float, ...],
        checkpoint_paths: tuple[str, ...] = (),
        checkpoint_weight: float = 0.0,
        device: str = "cpu",
    ):
        names: list[str] = []
        weights: list[float] = []
        self.opponents = {}

        for name, weight in zip(bot_pool, bot_weights):
            names.append(name)
            weights.append(float(weight))
            self.opponents[name] = create_bot(name=name)

        for index, path in enumerate(checkpoint_paths):
            name = f"checkpoint:{index}"
            names.append(name)
            weights.append(float(checkpoint_weight))
            self.opponents[name] = CheckpointOpponent(path, device=device)

        if not names:
            raise ValueError("at least one opponent is required")
        raw_weights = np.asarray(weights, dtype=np.float64)
        if np.any(raw_weights < 0) or raw_weights.sum() <= 0:
            raise ValueError("opponent weights must be non-negative and sum to a positive value")

        self.names = tuple(names)
        self.weights = raw_weights / raw_weights.sum()
        self.current_name: str | None = None

    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        if self.current_name is None or np.count_nonzero(board) <= 1:
            self.current_name = str(rng.choice(self.names, p=self.weights))
        return int(self.opponents[self.current_name].next_action(board, player, rng))
