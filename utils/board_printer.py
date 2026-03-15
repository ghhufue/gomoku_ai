from __future__ import annotations

from typing import Iterable

import numpy as np

from gomoku_ai.env import BLACK, BOARD_SIZE, EMPTY, WHITE


RESET = "\033[0m"
FG_BLACK = "\033[1;34m"
FG_WHITE = "\033[1;33m"
FG_LAST_MOVE = "\033[1;31m"
FG_GRID = "\033[90m"

BLACK_STONE = "●"
WHITE_STONE = "○"
EMPTY_CELL = "·"


def _cell_text(value: int, is_last_move: bool) -> str:
    if value == BLACK:
        stone = BLACK_STONE
        color = FG_LAST_MOVE if is_last_move else FG_BLACK
    elif value == WHITE:
        stone = WHITE_STONE
        color = FG_LAST_MOVE if is_last_move else FG_WHITE
    else:
        stone = EMPTY_CELL
        color = FG_LAST_MOVE if is_last_move else FG_GRID
    return f"{color}{stone}{RESET}"


def format_board(
    board: np.ndarray,
    *,
    last_move: tuple[int, int] | None = None,
    highlights: Iterable[tuple[int, int]] | None = None,
) -> str:
    highlight_set = set(highlights or [])
    if last_move is not None:
        highlight_set.add(last_move)

    header = "   " + " ".join(f"{col:02d}" for col in range(BOARD_SIZE))
    rows = [header]
    for row in range(BOARD_SIZE):
        cells: list[str] = []
        for col in range(BOARD_SIZE):
            is_last_move = (row, col) in highlight_set
            cells.append(_cell_text(int(board[row, col]), is_last_move))
        rows.append(f"{row:02d} " + " ".join(cells))
    return "\n".join(rows)


def print_board(
    board: np.ndarray,
    *,
    last_move: tuple[int, int] | None = None,
    highlights: Iterable[tuple[int, int]] | None = None,
) -> None:
    print(format_board(board, last_move=last_move, highlights=highlights))
