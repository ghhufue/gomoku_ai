from __future__ import annotations

import math

import numpy as np

from gomoku_ai.env import (
    BOARD_SIZE,
    EMPTY,
    FIVE_IDX,
    LIVE_FOUR_IDX,
    LIVE_THREE_IDX,
    LIVE_TWO_IDX,
    RUSH_FOUR_IDX,
    SLEEP_THREE_IDX,
    action_to_coord,
    classify_move_counts,
    coord_to_action,
)


def neighboring_actions(board: np.ndarray) -> list[int]:
    stones = np.argwhere(board != EMPTY)
    if len(stones) == 0:
        center = BOARD_SIZE // 2
        return [coord_to_action(center, center)]

    candidates: set[int] = set()
    for row, col in stones:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr = int(row) + dr
                nc = int(col) + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == EMPTY:
                    candidates.add(coord_to_action(nr, nc))

    return list(candidates)


class RuleBasedBot:
    def select_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        candidates = neighboring_actions(board)
        if not candidates:
            empties = np.flatnonzero(board.reshape(-1) == EMPTY)
            return int(rng.choice(empties))

        best_action = candidates[0]
        best_score = -1_000_000.0
        block_action: int | None = None
        center_row = BOARD_SIZE // 2
        center_col = BOARD_SIZE // 2

        for action in candidates:
            row, col = action_to_coord(action)
            board[row, col] = player
            pattern = classify_move_counts(board, row, col, player)
            board[row, col] = EMPTY
            if pattern[FIVE_IDX]:
                return action

            board[row, col] = -player
            opp_pattern = classify_move_counts(board, row, col, -player)
            board[row, col] = EMPTY
            if block_action is None and opp_pattern[FIVE_IDX]:
                block_action = action

            score = 0.0
            score += pattern[LIVE_FOUR_IDX] * 600
            score += pattern[RUSH_FOUR_IDX] * 260
            score += pattern[LIVE_THREE_IDX] * 90
            score += pattern[SLEEP_THREE_IDX] * 30
            score += pattern[LIVE_TWO_IDX] * 12
            score -= math.hypot(row - center_row, col - center_col)
            score += float(rng.random()) * 0.01

            if score > best_score:
                best_score = score
                best_action = action

        if block_action is not None:
            return block_action

        return best_action
