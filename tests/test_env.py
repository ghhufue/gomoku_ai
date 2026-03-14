from __future__ import annotations

import numpy as np

from gomoku_ai.env import (
    BLACK,
    BOARD_SIZE,
    CRITICAL_REWARD,
    EMPTY,
    PROBE_REWARD,
    SHAPE_REWARD,
    STEP_PENALTY,
    TERMINAL_REWARD,
    GomokuEnv,
    classify_move,
    coord_to_action,
    evaluate_shape_reward,
    immediate_winning_actions,
    threat_summary,
)


class StaticOpponent:
    def __init__(self, action: int):
        self.action = action

    def select_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        if board.reshape(-1)[self.action] == EMPTY:
            return self.action
        empties = np.flatnonzero(board.reshape(-1) == EMPTY)
        return int(empties[0])


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


def test_classify_move_detects_live_four() -> None:
    board = empty_board()
    place_many(board, [(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])
    board[7, 7] = BLACK

    pattern = classify_move(board, 7, 7, BLACK)

    assert pattern["live_four"] >= 1
    assert pattern["five"] is False


def test_classify_move_detects_live_three() -> None:
    board = empty_board()
    place_many(board, [(7, 5, BLACK), (7, 6, BLACK)])
    board[7, 7] = BLACK

    pattern = classify_move(board, 7, 7, BLACK)

    assert pattern["live_three"] >= 1


def test_classify_move_detects_rush_four() -> None:
    board = empty_board()
    place_many(board, [(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 3, -BLACK)])
    board[7, 7] = BLACK

    pattern = classify_move(board, 7, 7, BLACK)

    assert pattern["rush_four"] >= 1
    assert pattern["live_four"] == 0


def test_classify_move_detects_sleep_three() -> None:
    board = empty_board()
    place_many(board, [(7, 4, -BLACK), (7, 5, BLACK), (7, 6, BLACK)])
    board[7, 7] = BLACK

    pattern = classify_move(board, 7, 7, BLACK)

    assert pattern["sleep_three"] >= 1


def test_classify_move_detects_double_live_three() -> None:
    board = empty_board()
    place_many(board, [(7, 6, BLACK), (7, 8, BLACK), (6, 7, BLACK), (8, 7, BLACK)])
    board[7, 7] = BLACK

    pattern = classify_move(board, 7, 7, BLACK)

    assert pattern["live_three"] >= 2


def test_immediate_winning_actions_finds_both_ends() -> None:
    board = empty_board()
    place_many(board, [(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])

    wins = set(immediate_winning_actions(board, BLACK))

    assert wins == {coord_to_action(7, 3), coord_to_action(7, 8)}


def test_threat_summary_counts_winning_actions() -> None:
    board = empty_board()
    place_many(board, [(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])

    summary = threat_summary(board, BLACK)

    assert summary["winning_actions"] == 2


def test_evaluate_shape_reward_rewards_blocking_opponent_win() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 3, BLACK), (7, 4, -BLACK), (7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK)])
    board_after = board_before.copy()
    board_after[7, 8] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 7, 8, BLACK)

    assert reward >= CRITICAL_REWARD
    assert info["opp_threats_before"]["winning_actions"] == 1
    assert info["opp_threats_after"]["winning_actions"] == 0


def test_evaluate_shape_reward_rewards_live_three_creation() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 5, BLACK), (7, 6, BLACK)])
    board_after = board_before.copy()
    board_after[7, 7] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 7, 7, BLACK)

    assert reward > STEP_BASELINE()
    assert info["self_pattern"]["live_three"] >= 1


def test_evaluate_shape_reward_rewards_double_live_three_more_than_single_live_three() -> None:
    single_before = empty_board()
    place_many(single_before, [(7, 5, BLACK), (7, 6, BLACK)])
    single_after = single_before.copy()
    single_after[7, 7] = BLACK
    single_reward, _ = evaluate_shape_reward(single_before, single_after, 7, 7, BLACK)

    double_before = empty_board()
    place_many(double_before, [(7, 6, BLACK), (7, 8, BLACK), (6, 7, BLACK), (8, 7, BLACK)])
    double_after = double_before.copy()
    double_after[7, 7] = BLACK
    double_reward, info = evaluate_shape_reward(double_before, double_after, 7, 7, BLACK)

    assert info["self_pattern"]["live_three"] >= 2
    assert double_reward > single_reward


def test_evaluate_shape_reward_partial_block_does_not_get_critical_bonus() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 4, -BLACK), (7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK)])
    board_after = board_before.copy()
    board_after[7, 8] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 7, 8, BLACK)

    assert info["opp_threats_before"]["winning_actions"] == 2
    assert info["opp_threats_after"]["winning_actions"] == 1
    assert reward < CRITICAL_REWARD


def test_evaluate_shape_reward_live_two_stays_small() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 6, BLACK)])
    board_after = board_before.copy()
    board_after[7, 7] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 7, 7, BLACK)

    assert info["self_pattern"]["live_two"] >= 1
    assert STEP_PENALTY < reward < SHAPE_REWARD


def test_evaluate_shape_reward_reward_is_reasonably_bounded_for_non_terminal_move() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 6, BLACK), (7, 8, BLACK), (6, 7, BLACK), (8, 7, BLACK)])
    board_after = board_before.copy()
    board_after[7, 7] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 7, 7, BLACK)

    assert info["self_pattern"]["five"] is False
    assert reward < TERMINAL_REWARD
    assert reward <= CRITICAL_REWARD + SHAPE_REWARD + 60.0


def STEP_BASELINE() -> float:
    return -1.0


def test_evaluate_shape_reward_returns_terminal_reward_for_win() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])
    board_after = board_before.copy()
    board_after[7, 7] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 7, 7, BLACK)

    assert reward == TERMINAL_REWARD
    assert info["self_pattern"]["five"] is True


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
    assert result.reward == TERMINAL_REWARD
