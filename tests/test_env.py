from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import (
    BLACK,
    BOARD_SIZE,
    CRITICAL_REWARD,
    EMPTY,
    CPP_BACKEND_AVAILABLE,
    DEFAULT_REWARD_CONFIG,
    PROBE_REWARD,
    RewardConfig,
    SHAPE_REWARD,
    STEP_PENALTY,
    TERMINAL_REWARD,
    UNRESOLVED_FOUR_THREAT_PENALTY,
    UNRESOLVED_WINNING_THREAT_PENALTY,
    GomokuEnv,
    _py_affected_actions,
    _py_classify_move_counts,
    _py_classify_move_shape_details,
    _py_immediate_winning_actions,
    _py_threat_summary,
    affected_actions,
    classify_move,
    classify_move_counts,
    classify_move_shape_details,
    coord_to_action,
    evaluate_shape_reward,
    immediate_winning_actions,
    threat_summary,
)
from utils.board_printer import print_board


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


def reward_for_move(stones_before: list[tuple[int, int, int]], move: tuple[int, int], player: int = BLACK) -> tuple[float, dict[str, object]]:
    board_before = empty_board()
    place_many(board_before, stones_before)
    board_after = board_before.copy()
    row, col = move
    board_after[row, col] = player
    return evaluate_shape_reward(board_before, board_after, row, col, player)


@dataclass(frozen=True)
class RewardDebugCase:
    case_id: int
    name: str
    stones_before: tuple[tuple[int, int, int], ...]
    move: tuple[int, int]
    player: int
    description: str


def make_reward_debug_cases() -> dict[str, RewardDebugCase]:
    return {
        "block_opponent_win": RewardDebugCase(
            case_id=1,
            name="block_opponent_win",
            stones_before=((7, 3, BLACK), (7, 4, -BLACK), (7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK)),
            move=(7, 8),
            player=BLACK,
            description="阻挡对手下一手直接成五。",
        ),
        "create_live_three": RewardDebugCase(
            case_id=2,
            name="create_live_three",
            stones_before=((7, 5, BLACK), (7, 6, BLACK)),
            move=(7, 7),
            player=BLACK,
            description="形成自己的活三。",
        ),
        "ignore_opponent_live_four": RewardDebugCase(
            case_id=3,
            name="ignore_opponent_live_four",
            stones_before=((7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK), (0, 0, BLACK)),
            move=(10, 10),
            player=BLACK,
            description="对手已有四威胁，但当前落子没有处理。",
        ),
        "ignore_opponent_live_three": RewardDebugCase(
            case_id=4,
            name="ignore_opponent_live_three",
            stones_before=((7, 6, -BLACK), (7, 7, -BLACK), (7, 8, -BLACK)),
            move=(10, 10),
            player=BLACK,
            description="对手已有活三前驱威胁，但当前落子没有处理。",
        ),
        "block_opponent_live_three": RewardDebugCase(
            case_id=5,
            name="block_opponent_live_three",
            stones_before=((7, 6, -BLACK), (7, 7, -BLACK), (7, 8, -BLACK)),
            move=(7, 5),
            player=BLACK,
            description="去挡对手活三的一端。",
        ),
        "terminal_win": RewardDebugCase(
            case_id=6,
            name="terminal_win",
            stones_before=((7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)),
            move=(7, 7),
            player=BLACK,
            description="当前落子直接成五获胜。",
        ),
    }


REWARD_DEBUG_CASES = make_reward_debug_cases()
REWARD_DEBUG_CASES_BY_ID = {case.case_id: case for case in REWARD_DEBUG_CASES.values()}


def build_reward_debug_case(case: RewardDebugCase) -> tuple[np.ndarray, np.ndarray]:
    board_before = empty_board()
    place_many(board_before, list(case.stones_before))
    board_after = board_before.copy()
    row, col = case.move
    board_after[row, col] = case.player
    return board_before, board_after


def print_reward_debug_case(case_name: str | None = None, case_id: int | None = None) -> int:
    case = None
    if case_id is not None:
        case = REWARD_DEBUG_CASES_BY_ID.get(case_id)
    if case is None and case_name is not None:
        case = REWARD_DEBUG_CASES.get(case_name)
    if case is None:
        target = f"id={case_id}" if case_id is not None else f"name={case_name}"
        print(f"unknown case: {target}")
        print("available cases:")
        for item in list_reward_debug_cases():
            print(f"  - [{item.case_id}] {item.name}: {item.description}")
        return 1

    board_before, board_after = build_reward_debug_case(case)
    row, col = case.move
    reward, info = evaluate_shape_reward(board_before, board_after, row, col, case.player)

    print(f"[case] [{case.case_id}] {case.name}")
    print(f"[desc] {case.description}")
    print(f"[move] player={'black' if case.player == BLACK else 'white'} row={row} col={col}")
    print("[board_before]")
    print_board(board_before)
    print("[board_after]")
    print_board(board_after, last_move=(row, col))
    print(f"[reward] total={reward:.2f}")
    print("[reward_components]")
    for key, value in info["reward_components"].items():
        print(f"  - {key}: {float(value):.2f}")
    print("[self_pattern]")
    print(f"  - {info['self_pattern']}")
    print("[opp_threats_before]")
    print(f"  - {info['opp_threats_before']}")
    print("[opp_threats_after]")
    print(f"  - {info['opp_threats_after']}")
    return 0


def list_reward_debug_cases() -> list[RewardDebugCase]:
    return sorted(REWARD_DEBUG_CASES.values(), key=lambda item: item.case_id)


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
    assert pattern["rush_four"] == 0
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


def test_threat_summary_open_three_counts_live_four_actions_without_rush_four_overlap() -> None:
    board = empty_board()
    place_many(board, [(7, 6, BLACK), (7, 7, BLACK), (7, 8, BLACK)])

    summary = threat_summary(board, BLACK)

    assert summary["winning_actions"] == 0
    assert summary["live_four"] >= 2
    assert summary["rush_four"] == 0


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


def test_weighted_live_two_rewards_follow_scale_order() -> None:
    contiguous_reward, contiguous_info = reward_for_move([(7, 6, BLACK)], (7, 7))
    gap1_reward, gap1_info = reward_for_move([(7, 5, BLACK)], (7, 7))
    gap2_reward, gap2_info = reward_for_move([(7, 4, BLACK)], (7, 7))

    assert contiguous_info["move_shape_details"]["family_totals"]["live_two"] == 1.0
    assert gap1_info["move_shape_details"]["family_totals"]["live_two"] == 0.5
    assert gap2_info["move_shape_details"]["family_totals"]["live_two"] == 0.15
    assert contiguous_reward > gap1_reward > gap2_reward


def test_weighted_live_three_rewards_follow_scale_order() -> None:
    contiguous_reward, contiguous_info = reward_for_move([(7, 5, BLACK), (7, 6, BLACK)], (7, 7))
    gap1_reward, gap1_info = reward_for_move([(7, 5, BLACK), (7, 7, BLACK)], (7, 8))
    gap2_reward, gap2_info = reward_for_move([(7, 5, BLACK), (7, 6, BLACK)], (7, 9))

    assert contiguous_info["move_shape_details"]["family_totals"]["live_three"] == 1.0
    assert gap1_info["move_shape_details"]["family_totals"]["live_three"] == 0.7
    assert gap2_info["move_shape_details"]["family_totals"]["live_three"] == 0.3
    assert contiguous_reward > gap1_reward > gap2_reward


def test_weighted_four_rewards_follow_scale_order() -> None:
    live_four_contiguous_reward, live_four_contiguous_info = reward_for_move(
        [(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)],
        (7, 7),
    )
    live_four_gap_reward, live_four_gap_info = reward_for_move(
        [(7, 4, BLACK), (7, 5, BLACK), (7, 7, BLACK)],
        (7, 8),
    )
    rush_four_contiguous_reward, rush_four_contiguous_info = reward_for_move(
        [(7, 3, -BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)],
        (7, 7),
    )
    rush_four_gap_reward, rush_four_gap_info = reward_for_move(
        [(7, 3, -BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 7, BLACK)],
        (7, 8),
    )

    assert live_four_contiguous_info["move_shape_details"]["family_totals"]["live_four"] == 1.0
    assert live_four_gap_info["move_shape_details"]["family_totals"]["live_four"] == 0.5
    assert rush_four_contiguous_info["move_shape_details"]["family_totals"]["rush_four"] == 1.0
    assert rush_four_gap_info["move_shape_details"]["family_totals"]["rush_four"] == 0.5
    assert live_four_contiguous_reward > live_four_gap_reward
    assert rush_four_contiguous_reward > rush_four_gap_reward
    assert live_four_contiguous_reward > rush_four_contiguous_reward


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
    assert info["unresolved_winning_threat"] is True


def test_evaluate_shape_reward_penalizes_ignoring_opponent_winning_threat() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 4, -BLACK), (7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK)])
    board_after = board_before.copy()
    board_after[10, 10] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 10, 10, BLACK)

    assert info["opp_threats_before"]["winning_actions"] == 2
    assert info["opp_threats_after"]["winning_actions"] == 2
    assert info["unresolved_winning_threat"] is True
    assert reward < -100.0


def test_evaluate_shape_reward_penalizes_ignoring_opponent_live_four() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK), (0, 0, BLACK)])
    board_after = board_before.copy()
    board_after[10, 10] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 10, 10, BLACK)

    assert info["opp_threats_before"]["winning_actions"] == 0
    assert info["opp_threats_before"]["live_four"] + info["opp_threats_before"]["rush_four"] > 0
    assert info["opp_threats_after"]["live_four"] + info["opp_threats_after"]["rush_four"] >= (
        info["opp_threats_before"]["live_four"] + info["opp_threats_before"]["rush_four"]
    )
    assert info["unresolved_four_threat"] is True
    assert reward < 0.0


def test_threat_summary_does_not_count_dead_four_as_rush_four() -> None:
    board = empty_board()
    place_many(board, [(7, 4, BLACK), (7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK), (7, 8, -BLACK), (7, 9, BLACK)])

    summary = threat_summary(board, -BLACK)

    assert summary["winning_actions"] == 0
    assert summary["live_four"] == 0
    assert summary["rush_four"] == 0


def test_evaluate_shape_reward_does_not_penalize_ignoring_dead_four() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 4, BLACK), (7, 5, -BLACK), (7, 6, -BLACK), (7, 7, -BLACK), (7, 8, -BLACK), (7, 9, BLACK)])
    board_after = board_before.copy()
    board_after[10, 10] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 10, 10, BLACK)

    assert info["opp_threats_before"]["winning_actions"] == 0
    assert info["opp_threats_before"]["live_four"] == 0
    assert info["opp_threats_before"]["rush_four"] == 0
    assert info["opp_threats_after"]["live_four"] == 0
    assert info["opp_threats_after"]["rush_four"] == 0
    assert info["unresolved_four_threat"] is False
    assert info["reward_components"]["unresolved_four_penalty"] == 0.0
    assert reward > -DEFAULT_REWARD_CONFIG.unresolved_four_threat_penalty


def test_evaluate_shape_reward_penalizes_ignoring_opponent_live_three() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 6, -BLACK), (7, 7, -BLACK), (7, 8, -BLACK)])
    board_after = board_before.copy()
    board_after[10, 10] = BLACK

    reward, info = evaluate_shape_reward(board_before, board_after, 10, 10, BLACK)

    assert info["opp_threats_before"]["live_four"] > 0
    assert info["opp_threats_before"]["rush_four"] == 0
    assert info["opp_threats_after"]["live_four"] >= info["opp_threats_before"]["live_four"]
    assert info["unresolved_live_three_threat"] is True
    assert reward <= -DEFAULT_REWARD_CONFIG.unresolved_live_three_threat_penalty


def test_evaluate_shape_reward_blocking_opponent_live_three_avoids_live_three_penalty() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 6, -BLACK), (7, 7, -BLACK), (7, 8, -BLACK)])
    ignored_after = board_before.copy()
    ignored_after[10, 10] = BLACK
    board_after = board_before.copy()
    board_after[7, 5] = BLACK

    ignored_reward, ignored_info = evaluate_shape_reward(board_before, ignored_after, 10, 10, BLACK)
    reward, info = evaluate_shape_reward(board_before, board_after, 7, 5, BLACK)

    assert info["opp_threats_before"]["live_four"] > 0
    assert info["unresolved_live_three_threat"] is False
    assert info["unresolved_four_threat"] is False
    assert ignored_info["unresolved_live_three_threat"] is True
    assert ignored_info["unresolved_four_threat"] is True
    assert reward > 0.0
    assert reward > ignored_reward


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


def test_evaluate_shape_reward_uses_custom_reward_config() -> None:
    board_before = empty_board()
    place_many(board_before, [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])
    board_after = board_before.copy()
    board_after[7, 7] = BLACK

    custom_reward = RewardConfig(terminal_reward=321.0)
    reward, info = evaluate_shape_reward(board_before, board_after, 7, 7, BLACK, reward_config=custom_reward)

    assert reward == 321.0
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


def test_cpp_backend_matches_python_reference_on_hot_path() -> None:
    if not CPP_BACKEND_AVAILABLE:
        return

    board = empty_board()
    place_many(
        board,
        [
            (7, 7, BLACK),
            (7, 6, BLACK),
            (7, 8, BLACK),
            (6, 7, -BLACK),
            (8, 8, BLACK),
            (8, 6, -BLACK),
        ],
    )

    assert classify_move_counts(board, 7, 9, BLACK) == _py_classify_move_counts(board, 7, 9, BLACK)
    assert threat_summary(board, BLACK) == _py_threat_summary(board.copy(), BLACK)
    assert immediate_winning_actions(board, BLACK) == _py_immediate_winning_actions(board.copy(), BLACK)
    assert affected_actions(board, 7, 7) == _py_affected_actions(board.copy(), 7, 7)
    assert classify_move_shape_details(board, 7, 9, BLACK) == _py_classify_move_shape_details(board, 7, 9, BLACK)


@dataclass(frozen=True)
class CaseIntent:
    case_id: int
    test_name: str
    intent: str
    debug_case_id: int | None = None


TEST_INTENTS: tuple[CaseIntent, ...] = (
    CaseIntent(1, "test_classify_move_detects_five_in_row", "验证成五识别正确。", 6),
    CaseIntent(2, "test_classify_move_detects_live_four", "验证活四识别正确。"),
    CaseIntent(3, "test_classify_move_detects_live_three", "验证活三识别正确。", 2),
    CaseIntent(4, "test_classify_move_detects_rush_four", "验证冲四识别正确。"),
    CaseIntent(5, "test_classify_move_detects_sleep_three", "验证眠三识别正确。"),
    CaseIntent(6, "test_classify_move_detects_double_live_three", "验证双活三识别正确。"),
    CaseIntent(7, "test_immediate_winning_actions_finds_both_ends", "验证立即制胜点搜索能找到两端。"),
    CaseIntent(8, "test_threat_summary_counts_winning_actions", "验证 threat_summary 对立即制胜点计数正确。"),
    CaseIntent(9, "test_threat_summary_open_three_counts_live_four_actions_without_rush_four_overlap", "验证活三局面不会把 live_four 与 rush_four 重叠计数。"),
    CaseIntent(10, "test_evaluate_shape_reward_rewards_blocking_opponent_win", "验证成功挡住对手立即取胜点会获得高奖励。", 1),
    CaseIntent(11, "test_evaluate_shape_reward_rewards_live_three_creation", "验证形成自己的活三会获得正奖励。", 2),
    CaseIntent(12, "test_evaluate_shape_reward_rewards_double_live_three_more_than_single_live_three", "验证双活三奖励高于单活三。"),
    CaseIntent(13, "test_evaluate_shape_reward_partial_block_does_not_get_critical_bonus", "验证只挡住一部分必胜点时不会拿到完整关键奖励。"),
    CaseIntent(14, "test_evaluate_shape_reward_penalizes_ignoring_opponent_winning_threat", "验证无视对手立即取胜威胁会被重罚。"),
    CaseIntent(15, "test_evaluate_shape_reward_penalizes_ignoring_opponent_live_four", "验证无视对手四威胁会被惩罚。", 3),
    CaseIntent(16, "test_evaluate_shape_reward_penalizes_ignoring_opponent_live_three", "验证无视对手活三前驱威胁会被惩罚。", 4),
    CaseIntent(17, "test_evaluate_shape_reward_blocking_opponent_live_three_avoids_live_three_penalty", "验证去挡对手活三会优于完全不挡。", 5),
    CaseIntent(18, "test_evaluate_shape_reward_live_two_stays_small", "验证活二奖励保持较小。"),
    CaseIntent(19, "test_evaluate_shape_reward_reward_is_reasonably_bounded_for_non_terminal_move", "验证非终局单步奖励有合理上界。"),
    CaseIntent(20, "test_evaluate_shape_reward_returns_terminal_reward_for_win", "验证成五时直接返回终局奖励。", 6),
    CaseIntent(21, "test_evaluate_shape_reward_uses_custom_reward_config", "验证 reward 配置可被自定义参数覆盖。", 6),
    CaseIntent(22, "test_env_illegal_move_ends_episode", "验证环境中非法落子会直接判负终局。"),
    CaseIntent(23, "test_env_reports_win_before_opponent_turn", "验证我方成五时环境会在对手行动前结束。"),
    CaseIntent(24, "test_cpp_backend_matches_python_reference_on_hot_path", "验证 C++ 后端与 Python 参考实现一致。"),
)

TEST_INTENTS_BY_ID = {item.case_id: item for item in TEST_INTENTS}
TEST_FUNCTIONS: dict[str, Callable[[], None]] = {name: obj for name, obj in globals().items() if name.startswith("test_")}


def run_single_test_case(case_id: int) -> int:
    intent = TEST_INTENTS_BY_ID.get(case_id)
    if intent is None:
        print(f"unknown case id: {case_id}")
        for item in TEST_INTENTS:
            print(f"  - [{item.case_id}] {item.test_name}: {item.intent}")
        return 1

    print(f"[run] [{intent.case_id}] {intent.test_name}")
    print(f"[intent] {intent.intent}")
    test_fn = TEST_FUNCTIONS[intent.test_name]
    test_fn()
    print("[assert] passed")

    if intent.debug_case_id is not None:
        print_reward_debug_case(case_id=intent.debug_case_id)
    else:
        print("[detail] 当前测试没有单独的棋盘 / reward 调试案例。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gomoku env tests or inspect a single reward case.")
    parser.add_argument("--case", action="append", default=[], help="Run a named debug case instead of full pytest.")
    parser.add_argument("--case-id", action="append", type=int, default=[], help="Run a single numbered test case.")
    parser.add_argument("--list-cases", action="store_true", help="List available debug cases.")
    args = parser.parse_args(argv)

    if args.list_cases:
        for item in TEST_INTENTS:
            extra = f" -> debug[{item.debug_case_id}]" if item.debug_case_id is not None else ""
            print(f"[{item.case_id}] {item.test_name}{extra}")
            print(f"    {item.intent}")
        return 0

    if args.case_id:
        status = 0
        for case_id in args.case_id:
            status = max(status, run_single_test_case(case_id))
        return status

    if args.case:
        status = 0
        for case_name in args.case:
            status = max(status, print_reward_debug_case(case_name))
        return status

    import pytest

    return int(pytest.main([__file__]))


if __name__ == "__main__":
    raise SystemExit(main())
