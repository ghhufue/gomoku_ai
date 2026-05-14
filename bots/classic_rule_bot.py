from __future__ import annotations

import numpy as np

from bots.base import Bot
from bots.reward_driven_bot import neighboring_actions_with_radius
from gomoku_ai.cpp_backend import score_classic_candidates
from gomoku_ai.env import EMPTY


class ClassicRuleBot(Bot):
    def __init__(self, candidate_radius: int = 2, thread_batch_size: int = 10, name: str = "classic_rule"):
        super().__init__(name=name)
        self.candidate_radius = max(1, int(candidate_radius))
        self.thread_batch_size = max(1, int(thread_batch_size))

    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        candidates = neighboring_actions_with_radius(board, radius=self.candidate_radius)
        if not candidates:
            empties = np.flatnonzero(board.reshape(-1) == EMPTY)
            return int(rng.choice(empties))

        scored_candidates = score_classic_candidates(
            board,
            candidates,
            player,
            thread_batch_size=self.thread_batch_size,
        )

        best_action = int(candidates[0])
        best_score = -1_000_000.0
        block_action: int | None = None

        for item in scored_candidates:
            action = int(item["action"])
            if bool(item["self_win"]):
                return action
            if block_action is None and bool(item["opp_win"]):
                block_action = action
            score = float(item["score"]) + float(rng.random()) * 0.01

            if score > best_score:
                best_score = score
                best_action = action

        if block_action is not None:
            return block_action

        return best_action
