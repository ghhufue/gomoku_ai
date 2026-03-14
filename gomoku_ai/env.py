from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gomoku_ai.cpp_backend import (
    BACKEND_AVAILABLE as CPP_BACKEND_AVAILABLE,
    affected_actions as cpp_affected_actions,
    classify_move_counts as cpp_classify_move_counts,
    immediate_winning_actions as cpp_immediate_winning_actions,
    threat_summary as cpp_threat_summary,
)


BOARD_SIZE = 15
BOARD_AREA = BOARD_SIZE * BOARD_SIZE

EMPTY = 0
BLACK = 1
WHITE = -1

DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))

TERMINAL_REWARD = 1000.0
CRITICAL_REWARD = 100.0
SHAPE_REWARD = 20.0
PROBE_REWARD = 5.0
STEP_PENALTY = -1.0
UNRESOLVED_WINNING_THREAT_PENALTY = 120.0
UNRESOLVED_FOUR_THREAT_PENALTY = 45.0

PATTERN_KEYS = ("winning_actions", "live_four", "rush_four", "live_three", "sleep_three", "live_two")

LIVE_FOUR_PATTERNS = {"_XXXX_", "_XXX_X_", "_XX_XX_", "_X_XXX_"}
LIVE_THREE_PATTERNS = {"_XXX_", "_XX_X_", "_X_XX_"}
SLEEP_THREE_PATTERNS = {
    "OXXX__",
    "__XXXO",
    "O_XXX_",
    "_XXX_O",
    "OX_XX_",
    "_XX_XO",
    "OXX_X_",
    "_X_XXO",
}
LIVE_TWO_PATTERNS = {
    "_XX_",
    "_X_X_",
    "__XX__",
    "__X_X__",
    "_X__X_",
}
ANALYZE_RADIUS = 5
ANALYZE_CENTER = ANALYZE_RADIUS
ANALYZE_WINDOW_SPANS = tuple(
    (start, start + length)
    for length in (4, 5, 6, 7)
    for start in range(max(0, ANALYZE_CENTER - length + 1), min(ANALYZE_CENTER, 2 * ANALYZE_RADIUS + 1 - length) + 1)
)
FIVE_IDX = 0
LIVE_FOUR_IDX = 1
RUSH_FOUR_IDX = 2
LIVE_THREE_IDX = 3
SLEEP_THREE_IDX = 4
LIVE_TWO_IDX = 5


@dataclass
class StepResult:
    observation: np.ndarray
    action_mask: np.ndarray
    reward: float
    done: bool
    info: dict


def empty_pattern_summary() -> dict[str, int]:
    return {key: 0 for key in PATTERN_KEYS}


def action_to_coord(action: int) -> tuple[int, int]:
    return divmod(action, BOARD_SIZE)


def coord_to_action(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def direction_line(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int, radius: int = 5) -> tuple[str, int]:
    chars: list[str] = []
    board_size = BOARD_SIZE
    for offset in range(-radius, radius + 1):
        current_row = row + offset * dr
        current_col = col + offset * dc
        if current_row < 0 or current_row >= board_size or current_col < 0 or current_col >= board_size:
            chars.append("O")
            continue
        value = board[current_row, current_col]
        if value == player:
            chars.append("X")
        elif value == EMPTY:
            chars.append("_")
        else:
            chars.append("O")
    return "".join(chars), radius


def analyze_direction(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> tuple[int, int, int, int, int, int]:
    line, center = direction_line(board, row, col, dr, dc, player, ANALYZE_RADIUS)

    contiguous = 1
    left = center - 1
    while left >= 0 and line[left] == "X":
        contiguous += 1
        left -= 1
    right = center + 1
    line_len = len(line)
    while right < line_len and line[right] == "X":
        contiguous += 1
        right += 1

    if contiguous >= 5:
        return (1, 0, 0, 0, 0, 0)

    live_four = 0
    rush_four = 0
    live_three = 0
    sleep_three = 0
    live_two = 0

    for start, end in ANALYZE_WINDOW_SPANS:
        window = line[start:end]
        stones = window.count("X")
        empties = window.count("_")

        if window in LIVE_FOUR_PATTERNS:
            live_four = 1
            continue

        if stones == 4 and empties == 1:
            rush_four = 1
            continue

        if window in LIVE_THREE_PATTERNS:
            live_three = 1
            continue

        if window in SLEEP_THREE_PATTERNS:
            sleep_three = 1
            continue

        if window in LIVE_TWO_PATTERNS:
            live_two = 1

    return (0, live_four, rush_four, live_three, sleep_three, live_two)


def pattern_tuple_to_dict(pattern: tuple[int, int, int, int, int, int]) -> dict[str, int | bool]:
    return {
        "five": bool(pattern[FIVE_IDX]),
        "live_four": pattern[LIVE_FOUR_IDX],
        "rush_four": pattern[RUSH_FOUR_IDX],
        "live_three": pattern[LIVE_THREE_IDX],
        "sleep_three": pattern[SLEEP_THREE_IDX],
        "live_two": pattern[LIVE_TWO_IDX],
    }


def _py_classify_move_counts(board: np.ndarray, row: int, col: int, player: int) -> tuple[int, int, int, int, int, int]:
    five = 0
    live_four = 0
    rush_four = 0
    live_three = 0
    sleep_three = 0
    live_two = 0

    for dr, dc in DIRS:
        direction_patterns = analyze_direction(board, row, col, dr, dc, player)
        five |= direction_patterns[FIVE_IDX]
        live_four += direction_patterns[LIVE_FOUR_IDX]
        rush_four += direction_patterns[RUSH_FOUR_IDX]
        live_three += direction_patterns[LIVE_THREE_IDX]
        sleep_three += direction_patterns[SLEEP_THREE_IDX]
        live_two += direction_patterns[LIVE_TWO_IDX]

    return (five, live_four, rush_four, live_three, sleep_three, live_two)


def classify_move_counts(board: np.ndarray, row: int, col: int, player: int) -> tuple[int, int, int, int, int, int]:
    if CPP_BACKEND_AVAILABLE:
        return cpp_classify_move_counts(board, row, col, player)
    return _py_classify_move_counts(board, row, col, player)


def classify_move(board: np.ndarray, row: int, col: int, player: int) -> dict[str, int | bool]:
    return pattern_tuple_to_dict(classify_move_counts(board, row, col, player))


def empty_actions(board: np.ndarray) -> np.ndarray:
    return np.flatnonzero(board.reshape(-1) == EMPTY)


def iter_empty_action_patterns(
    board: np.ndarray, player: int, actions: Iterable[int] | None = None
) -> Iterable[tuple[int, tuple[int, int, int, int, int, int]]]:
    action_list = empty_actions(board).tolist() if actions is None else list(actions)
    for action in action_list:
        row, col = action_to_coord(int(action))
        board[row, col] = player
        pattern = classify_move_counts(board, row, col, player)
        board[row, col] = EMPTY
        yield int(action), pattern


def accumulate_pattern(summary: dict[str, int], pattern: tuple[int, int, int, int, int, int]) -> None:
    if pattern[FIVE_IDX]:
        summary["winning_actions"] += 1
    summary["live_four"] += pattern[LIVE_FOUR_IDX]
    summary["rush_four"] += pattern[RUSH_FOUR_IDX]
    summary["live_three"] += pattern[LIVE_THREE_IDX]
    summary["sleep_three"] += pattern[SLEEP_THREE_IDX]
    summary["live_two"] += pattern[LIVE_TWO_IDX]


def _py_immediate_winning_actions(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> list[int]:
    wins: list[int] = []
    for action, pattern in iter_empty_action_patterns(board, player, actions):
        if pattern[FIVE_IDX]:
            wins.append(action)
    return wins


def immediate_winning_actions(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> list[int]:
    if CPP_BACKEND_AVAILABLE:
        return cpp_immediate_winning_actions(board, player, actions)
    return _py_immediate_winning_actions(board, player, actions)


def _py_threat_summary(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> dict[str, int]:
    summary = empty_pattern_summary()
    for _, pattern in iter_empty_action_patterns(board, player, actions):
        accumulate_pattern(summary, pattern)
    return summary


def threat_summary(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> dict[str, int]:
    if CPP_BACKEND_AVAILABLE:
        return cpp_threat_summary(board, player, actions)
    return _py_threat_summary(board, player, actions)


def _py_affected_actions(board: np.ndarray, row: int, col: int, radius: int = 5) -> list[int]:
    actions: list[int] = []
    for action in empty_actions(board).tolist():
        action_row, action_col = action_to_coord(int(action))
        row_delta = action_row - row
        col_delta = action_col - col
        if (
            (action_row == row and abs(col_delta) <= radius)
            or (action_col == col and abs(row_delta) <= radius)
            or (abs(row_delta) == abs(col_delta) and abs(row_delta) <= radius)
        ):
            actions.append(int(action))
    return actions


def affected_actions(board: np.ndarray, row: int, col: int, radius: int = 5) -> list[int]:
    if CPP_BACKEND_AVAILABLE:
        return cpp_affected_actions(board, row, col, radius)
    return _py_affected_actions(board, row, col, radius)


def apply_local_threat_delta(
    board_before: np.ndarray,
    board_after: np.ndarray,
    row: int,
    col: int,
    player: int,
    before_summary: dict[str, int],
) -> dict[str, int]:
    # A single new stone only affects candidate moves that share one of the
    # four analyzed lines with the placed position and lie within classify radius.
    before_actions = affected_actions(board_before, row, col)
    if not before_actions:
        return before_summary.copy()

    after_empty_mask = board_after.reshape(-1) == EMPTY
    after_actions = [action for action in before_actions if after_empty_mask[action]]
    before_local = threat_summary(board_before, player, before_actions)
    after_local = threat_summary(board_after, player, after_actions)

    updated = before_summary.copy()
    for key in PATTERN_KEYS:
        updated[key] += after_local[key] - before_local[key]
    return updated


def summary_score(summary: dict[str, int]) -> float:
    return (
        summary["winning_actions"] * 80.0
        + summary["live_four"] * 30.0
        + summary["rush_four"] * 18.0
        + summary["live_three"] * 8.0
        + summary["sleep_three"] * 3.0
        + summary["live_two"] * 1.0
    )


def clamp_bonus(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def move_pattern_reward(pattern: dict[str, int | bool]) -> float:
    if pattern["five"]:
        return TERMINAL_REWARD
    if pattern["live_four"] or pattern["rush_four"]:
        return CRITICAL_REWARD
    if int(pattern["live_three"]) >= 2:
        return SHAPE_REWARD + 10.0
    if pattern["live_three"]:
        return SHAPE_REWARD
    if pattern["sleep_three"]:
        return 8.0
    if pattern["live_two"]:
        return PROBE_REWARD
    return 0.0


def evaluate_shape_reward(board_before: np.ndarray, board_after: np.ndarray, row: int, col: int, player: int) -> tuple[float, dict[str, object]]:
    mine_counts = classify_move_counts(board_after, row, col, player)
    if mine_counts[FIVE_IDX]:
        return TERMINAL_REWARD, {"self_pattern": pattern_tuple_to_dict(mine_counts)}

    mine = pattern_tuple_to_dict(mine_counts)

    before_self = threat_summary(board_before, player)
    before_opp = threat_summary(board_before, -player)
    after_self = apply_local_threat_delta(board_before, board_after, row, col, player, before_self)
    after_opp = apply_local_threat_delta(board_before, board_after, row, col, -player, before_opp)

    reward = STEP_PENALTY
    reward += move_pattern_reward(mine)

    offense_delta = summary_score(after_self) - summary_score(before_self)
    defense_delta = summary_score(before_opp) - summary_score(after_opp)
    reward += clamp_bonus(offense_delta * 0.12, 40.0)
    reward += clamp_bonus(defense_delta * 0.15, 60.0)

    unresolved_winning_threat = before_opp["winning_actions"] > 0 and after_opp["winning_actions"] > 0
    unresolved_four_threat = (
        before_opp["winning_actions"] == 0
        and after_opp["winning_actions"] == 0
        and before_opp["live_four"] + before_opp["rush_four"] > 0
        and after_opp["live_four"] + after_opp["rush_four"] >= before_opp["live_four"] + before_opp["rush_four"]
    )

    if before_opp["winning_actions"] > 0 and after_opp["winning_actions"] == 0:
        reward += CRITICAL_REWARD
    elif before_opp["live_four"] + before_opp["rush_four"] > after_opp["live_four"] + after_opp["rush_four"]:
        reward += 30.0
    elif before_opp["live_three"] > after_opp["live_three"]:
        reward += SHAPE_REWARD

    if unresolved_winning_threat:
        reward -= UNRESOLVED_WINNING_THREAT_PENALTY
    elif unresolved_four_threat:
        reward -= UNRESOLVED_FOUR_THREAT_PENALTY

    info = {
        "self_pattern": mine,
        "offense_delta": offense_delta,
        "defense_delta": defense_delta,
        "unresolved_winning_threat": unresolved_winning_threat,
        "unresolved_four_threat": unresolved_four_threat,
        "self_threats_after": after_self,
        "opp_threats_before": before_opp,
        "opp_threats_after": after_opp,
    }
    return reward, info


class GomokuEnv:
    """Single-agent environment where the agent plays against a built-in opponent."""

    def __init__(self, opponent, seed: int | None = None):
        self.opponent = opponent
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.agent_player = BLACK
        self.done = False

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.board.fill(EMPTY)
        self.done = False
        self.agent_player = BLACK if self.rng.random() < 0.5 else WHITE

        if self.agent_player == WHITE:
            opening = self.opponent.select_action(self.board.copy(), BLACK, self.rng)
            row, col = action_to_coord(opening)
            self.board[row, col] = BLACK

        return self.observation(), self.action_mask()

    def observation(self) -> np.ndarray:
        current = self.agent_player
        obs = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.board == current).astype(np.float32)
        obs[1] = (self.board == -current).astype(np.float32)
        obs[2] = (self.board == EMPTY).astype(np.float32)
        return obs

    def action_mask(self) -> np.ndarray:
        return self.board.reshape(-1) == EMPTY

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        row, col = action_to_coord(action)
        info: dict[str, object] = {}
        if not inside(row, col) or self.board[row, col] != EMPTY:
            self.done = True
            return StepResult(
                self.observation(),
                self.action_mask(),
                -TERMINAL_REWARD,
                True,
                {"illegal_move": True, "agent_result": "loss"},
            )

        before = self.board.copy()
        self.board[row, col] = self.agent_player
        reward, reward_info = evaluate_shape_reward(before, self.board, row, col, self.agent_player)
        info.update(reward_info)

        mine = reward_info["self_pattern"]
        if mine["five"]:
            self.done = True
            info["winner"] = self.agent_player
            info["agent_result"] = "win"
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        if not np.any(self.board == EMPTY):
            self.done = True
            info["draw"] = True
            info["agent_result"] = "draw"
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        opponent_action = self.opponent.select_action(self.board.copy(), -self.agent_player, self.rng)
        opp_row, opp_col = action_to_coord(opponent_action)
        if self.board[opp_row, opp_col] != EMPTY:
            empties = empty_actions(self.board)
            fallback = int(self.rng.choice(empties))
            opp_row, opp_col = action_to_coord(fallback)
            opponent_action = fallback

        self.board[opp_row, opp_col] = -self.agent_player
        opp_patterns = classify_move(self.board, opp_row, opp_col, -self.agent_player)
        info["opponent_action"] = opponent_action
        info["opponent_pattern"] = opp_patterns
        if opp_patterns["five"]:
            self.done = True
            info["winner"] = -self.agent_player
            info["agent_result"] = "loss"
            return StepResult(self.observation(), self.action_mask(), reward - TERMINAL_REWARD, True, info)

        if not np.any(self.board == EMPTY):
            self.done = True
            info["draw"] = True
            info["agent_result"] = "draw"
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        return StepResult(self.observation(), self.action_mask(), reward, False, info)


class VectorEnv:
    def __init__(self, envs: Iterable[GomokuEnv]):
        self.envs = list(envs)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs, masks = zip(*(env.reset() for env in self.envs))
        return np.stack(obs), np.stack(masks)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        next_obs = []
        next_masks = []
        rewards = []
        dones = []
        infos: list[dict] = []

        for env, action in zip(self.envs, actions.tolist()):
            result = env.step(int(action))
            obs = result.observation
            mask = result.action_mask
            if result.done:
                obs, mask = env.reset()
            next_obs.append(obs)
            next_masks.append(mask)
            rewards.append(result.reward)
            dones.append(result.done)
            infos.append(result.info)

        return (
            np.stack(next_obs),
            np.stack(next_masks),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )
