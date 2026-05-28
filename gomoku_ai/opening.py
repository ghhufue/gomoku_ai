from __future__ import annotations

import numpy as np

from gomoku_ai.cpp_backend import reset_env_state
from gomoku_ai.env import BLACK, BOARD_SIZE, EMPTY, WHITE, action_to_coord, coord_to_action, neighboring_action_mask


def has_five(board: np.ndarray, row: int, col: int, player: int) -> bool:
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            rr = row + dr * sign
            cc = col + dc * sign
            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == player:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= 5:
            return True
    return False


def random_legal_action(board: np.ndarray, rng: np.random.Generator, radius: int = 2) -> int | None:
    mask = neighboring_action_mask(board, radius=radius, opening_radius=1)
    actions = np.flatnonzero(mask)
    if actions.size == 0:
        actions = np.flatnonzero(board.reshape(-1) == EMPTY)
    if actions.size == 0:
        return None
    return int(rng.choice(actions))


def apply_random_move(
    board: np.ndarray,
    player: int,
    rng: np.random.Generator,
    *,
    radius: int = 2,
    max_attempts: int = 64,
) -> int | None:
    for _ in range(max_attempts):
        action = random_legal_action(board, rng, radius=radius)
        if action is None:
            return None
        row, col = action_to_coord(action)
        if board[row, col] != EMPTY:
            continue
        board[row, col] = player
        if has_five(board, row, col, player):
            board[row, col] = EMPTY
            continue
        return action
    return None


def apply_random_opening_pairs_to_env(
    env,
    pairs: int,
    rng: np.random.Generator,
    *,
    radius: int = 2,
) -> list[int]:
    """Add random historical move pairs while preserving the agent-to-move turn.

    `GomokuEnv` is single-agent: every `step()` starts with the model side to move,
    then the built-in opponent replies. For black models, a historical pair is
    black then white; for white models, it is white then black because reset()
    already lets black make the initial opening move.
    """
    if pairs <= 0:
        return []

    moves: list[int] = []
    if env.agent_player == BLACK:
        sequence = (BLACK, WHITE)
    else:
        sequence = (WHITE, BLACK)

    for _ in range(pairs):
        for player in sequence:
            action = apply_random_move(env.board, player, rng, radius=radius)
            if action is None:
                reset_env_state(env.env_id, env.board)
                return moves
            env.last_move = action
            moves.append(action)
    reset_env_state(env.env_id, env.board)
    return moves


def apply_random_opening_pairs_to_board(
    board: np.ndarray,
    pairs: int,
    rng: np.random.Generator,
    *,
    first_player: int = BLACK,
    radius: int = 2,
) -> tuple[int | None, int]:
    """Apply random pairs to a raw board and return `(last_move, next_player)`."""
    last_move: int | None = None
    current = first_player
    for _ in range(max(0, pairs) * 2):
        action = apply_random_move(board, current, rng, radius=radius)
        if action is None:
            break
        last_move = action
        current = -current
    return last_move, current


def move_coord_payload(action: int, player: int, source: str) -> dict[str, object]:
    row, col = action_to_coord(action)
    player_label = "black" if player == BLACK else "white"
    return {
        "player": player_label,
        "color": player_label,
        "source": source,
        "action": int(action),
        "row": int(row),
        "col": int(col),
    }
