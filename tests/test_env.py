from __future__ import annotations

import numpy as np

from bots import create_bot
from gomoku_ai.env import BLACK, BOARD_SIZE, EMPTY, GomokuEnv, classify_move, classify_move_counts, coord_to_action, evaluate_reward


class StaticOpponent:
    def __init__(self, action: int):
        self.action = action

    def select_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        if board.reshape(-1)[self.action] == EMPTY:
            return self.action
        return int(np.flatnonzero(board.reshape(-1) == EMPTY)[0])


def empty_board() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def place_many(board: np.ndarray, stones: list[tuple[int, int, int]]) -> np.ndarray:
    for row, col, player in stones:
        board[row, col] = player
    return board


def test_classify_move_detects_five_in_row() -> None:
    board = empty_board()
    place_many(board, [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])
    board[7, 7] = BLACK

    pattern = classify_move(board, 7, 7, BLACK)

    assert pattern["five"] is True


def test_classify_move_counts_detect_live_three() -> None:
    board = empty_board()
    place_many(board, [(7, 5, BLACK), (7, 6, BLACK)])
    board[7, 7] = BLACK

    pattern = classify_move_counts(board, 7, 7, BLACK)

    assert pattern[3] >= 1


def test_evaluate_reward_uses_cpp_backend() -> None:
    board = empty_board()

    reward, info = evaluate_reward(board, 7, 7, BLACK)

    assert isinstance(reward, float)
    assert info["reward"] == reward
    assert "offense_score" in info
    assert "defense_score" in info
    assert "special_rewards" in info
    assert "events" in info


def test_env_illegal_move_ends_episode() -> None:
    env = GomokuEnv(opponent=StaticOpponent(coord_to_action(0, 1)), seed=1)
    env.board[0, 0] = BLACK
    env.done = False

    result = env.step(coord_to_action(0, 0))

    assert result.done is True
    assert result.info["illegal_move"] is True
    assert result.info["agent_result"] == "loss"


def test_env_reports_win_before_opponent_turn() -> None:
    env = GomokuEnv(opponent=StaticOpponent(coord_to_action(0, 0)), seed=1)
    env.agent_player = BLACK
    env.done = False
    env.board.fill(EMPTY)
    place_many(env.board, [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])

    result = env.step(coord_to_action(7, 7))

    assert result.done is True
    assert result.info["agent_result"] == "win"
    assert result.reward > 0.0


def test_reward_driven_bot_can_be_constructed() -> None:
    bot = create_bot(name="reward_driven_hard")

    assert bot is not None
