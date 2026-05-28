from __future__ import annotations

import numpy as np

from bots.base import Bot
from gomoku_ai.cpp_backend import score_reward_candidates
from gomoku_ai.env import (
    BOARD_SIZE,
    EMPTY,
    coord_to_action,
)


def neighboring_actions(board: np.ndarray) -> list[int]:
    return neighboring_actions_with_radius(board, radius=2)


def neighboring_actions_with_radius(board: np.ndarray, radius: int) -> list[int]:
    stones = np.argwhere(board != EMPTY)
    if len(stones) == 0:
        center = BOARD_SIZE // 2
        return [coord_to_action(center, center)]

    candidates: set[int] = set()
    for row, col in stones:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = int(row) + dr
                nc = int(col) + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == EMPTY:
                    candidates.add(coord_to_action(nr, nc))

    return list(candidates)


class RewardDrivenBot(Bot):
    def __init__(
        self,
        candidate_radius: int = 2,
        top_k: int = 1,
        thread_batch_size: int = 10,
    ):
        self.candidate_radius = max(1, int(candidate_radius))
        self.top_k = max(1, int(top_k))
        self.thread_batch_size = max(1, int(thread_batch_size))

    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        candidates = neighboring_actions_with_radius(board, radius=self.candidate_radius)
        if not candidates:
            empties = np.flatnonzero(board.reshape(-1) == EMPTY)
            return int(rng.choice(empties))

        scored_actions = [
            (float(item["reward"]), int(item["action"]))
            for item in score_reward_candidates(
                board,
                candidates,
                player,
                thread_batch_size=self.thread_batch_size,
            )
        ]

        scored_actions.sort(key=lambda item: item[0], reverse=True)
        top_actions = scored_actions[: min(self.top_k, len(scored_actions))]
        if len(top_actions) == 1:
            return top_actions[0][1]

        rewards = np.asarray([item[0] for item in top_actions], dtype=np.float64)
        rewards = rewards - rewards.max()
        weights = np.exp(rewards)
        probabilities = weights / weights.sum()
        selected = int(rng.choice(len(top_actions), p=probabilities))
        return top_actions[selected][1]
