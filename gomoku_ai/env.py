from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Iterable

import numpy as np

from gomoku_ai.cpp_backend import (
    BACKEND_AVAILABLE as CPP_BACKEND_AVAILABLE,
    apply_env_move as cpp_apply_env_move,
    clear_env_state as cpp_clear_env_state,
    env_done as cpp_env_done,
    env_winner as cpp_env_winner,
    evaluate_env_reward as cpp_evaluate_env_reward,
    evaluate_reward as cpp_evaluate_reward,
    reset_env_state as cpp_reset_env_state,
)

FORCE_FOUR_TRAINING_UNTIL = 0
BOARD_SIZE = 15
BOARD_AREA = BOARD_SIZE * BOARD_SIZE

EMPTY = 0
BLACK = 1
WHITE = -1

DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))

TERMINAL_REWARD = 1000.0
ANALYZE_RADIUS = 5
ANALYZE_CENTER = ANALYZE_RADIUS
ANALYZE_WINDOW_SPANS = tuple(
    (start, start + length)
    for length in (4, 5, 6, 7)
    for start in range(
        max(0, ANALYZE_CENTER - length + 1),
        min(ANALYZE_CENTER, 2 * ANALYZE_RADIUS + 1 - length) + 1,
    )
)

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


def action_to_coord(action: int) -> tuple[int, int]:
    return divmod(action, BOARD_SIZE)


def coord_to_action(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def empty_actions(board: np.ndarray) -> np.ndarray:
    return np.flatnonzero(board.reshape(-1) == EMPTY)


def neighboring_action_mask(
    board: np.ndarray,
    radius: int = 2,
    opening_radius: int = 1,
) -> np.ndarray:
    if radius < 0 or opening_radius < 0:
        raise ValueError("radius and opening_radius must be non-negative")

    empty_mask = board == EMPTY
    occupied = np.argwhere(board != EMPTY)
    candidate = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

    if len(occupied) == 0:
        center = BOARD_SIZE // 2
        candidate[center, center] = True
    else:
        active_radius = opening_radius if len(occupied) == 1 else radius
        for row, col in occupied:
            row_start = max(0, int(row) - active_radius)
            row_end = min(BOARD_SIZE, int(row) + active_radius + 1)
            col_start = max(0, int(col) - active_radius)
            col_end = min(BOARD_SIZE, int(col) + active_radius + 1)
            candidate[row_start:row_end, col_start:col_end] = True

    candidate &= empty_mask
    if not candidate.any():
        candidate = empty_mask.copy()
    return candidate.reshape(-1)


def direction_line(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int, radius: int = ANALYZE_RADIUS) -> tuple[str, int]:
    chars: list[str] = []
    for offset in range(-radius, radius + 1):
        current_row = row + offset * dr
        current_col = col + offset * dc
        if not inside(current_row, current_col):
            chars.append("O")
            continue
        value = int(board[current_row, current_col])
        if value == player:
            chars.append("X")
        elif value == EMPTY:
            chars.append("_")
        else:
            chars.append("O")
    return "".join(chars), radius


def is_rush_four_window(window: str) -> bool:
    if window.count("X") != 4 or window.count("_") != 1:
        return False
    return "XXXXX" in window.replace("_", "X")


def analyze_direction(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> tuple[int, int, int, int, int, int]:
    line, center = direction_line(board, row, col, dr, dc, player)

    contiguous = 1
    left = center - 1
    while left >= 0 and line[left] == "X":
        contiguous += 1
        left -= 1
    right = center + 1
    while right < len(line) and line[right] == "X":
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
        if window in LIVE_FOUR_PATTERNS:
            live_four = 1
            continue
        if is_rush_four_window(window):
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

    if live_four:
        rush_four = 0

    return (0, live_four, rush_four, live_three, sleep_three, live_two)


def _py_classify_move_counts(board: np.ndarray, row: int, col: int, player: int) -> tuple[int, int, int, int, int, int]:
    five = 0
    live_four = 0
    rush_four = 0
    live_three = 0
    sleep_three = 0
    live_two = 0

    for dr, dc in DIRS:
        pattern = analyze_direction(board, row, col, dr, dc, player)
        five |= pattern[FIVE_IDX]
        live_four += pattern[LIVE_FOUR_IDX]
        rush_four += pattern[RUSH_FOUR_IDX]
        live_three += pattern[LIVE_THREE_IDX]
        sleep_three += pattern[SLEEP_THREE_IDX]
        live_two += pattern[LIVE_TWO_IDX]

    return (five, live_four, rush_four, live_three, sleep_three, live_two)


def classify_move_counts(board: np.ndarray, row: int, col: int, player: int) -> tuple[int, int, int, int, int, int]:
    return _py_classify_move_counts(board, row, col, player)


def classify_move(board: np.ndarray, row: int, col: int, player: int) -> dict[str, int | bool]:
    pattern = classify_move_counts(board, row, col, player)
    return {
        "five": bool(pattern[FIVE_IDX]),
        "live_four": pattern[LIVE_FOUR_IDX],
        "rush_four": pattern[RUSH_FOUR_IDX],
        "live_three": pattern[LIVE_THREE_IDX],
        "sleep_three": pattern[SLEEP_THREE_IDX],
        "live_two": pattern[LIVE_TWO_IDX],
    }


def _py_immediate_winning_actions(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> list[int]:
    wins: list[int] = []
    candidates = empty_actions(board).tolist() if actions is None else [int(action) for action in actions]
    for action in candidates:
        row, col = action_to_coord(action)
        board[row, col] = player
        if classify_move_counts(board, row, col, player)[FIVE_IDX]:
            wins.append(action)
        board[row, col] = EMPTY
    return wins


def immediate_winning_actions(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> list[int]:
    return _py_immediate_winning_actions(board, player, actions)


def _py_threat_summary(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> dict[str, int]:
    summary = {
        "winning_actions": 0,
        "live_four": 0,
        "rush_four": 0,
        "live_three": 0,
        "sleep_three": 0,
        "live_two": 0,
    }
    candidates = empty_actions(board).tolist() if actions is None else [int(action) for action in actions]
    for action in candidates:
        row, col = action_to_coord(action)
        if board[row, col] != EMPTY:
            continue
        board[row, col] = player
        pattern = classify_move_counts(board, row, col, player)
        board[row, col] = EMPTY
        if pattern[FIVE_IDX]:
            summary["winning_actions"] += 1
        summary["live_four"] += pattern[LIVE_FOUR_IDX]
        summary["rush_four"] += pattern[RUSH_FOUR_IDX]
        summary["live_three"] += pattern[LIVE_THREE_IDX]
        summary["sleep_three"] += pattern[SLEEP_THREE_IDX]
        summary["live_two"] += pattern[LIVE_TWO_IDX]
    return summary

def _find_move_to_make_four(board: np.ndarray, player: int) -> int | None:
    for action in empty_actions(board):
        row, col = action_to_coord(int(action))
        board[row, col] = player
        pattern = classify_move_counts(board, row, col, player)
        board[row, col] = EMPTY
        if pattern[LIVE_FOUR_IDX] > 0 or pattern[RUSH_FOUR_IDX] > 0:
            return int(action)
    return None


def _has_live_three(board: np.ndarray, player: int) -> bool:
    summary = threat_summary(board, player)
    return summary["live_three"] > 0

def threat_summary(board: np.ndarray, player: int, actions: Iterable[int] | None = None) -> dict[str, int]:
    return _py_threat_summary(board, player, actions)


def evaluate_reward(board_before: np.ndarray, row: int, col: int, player: int) -> tuple[float, dict[str, object]]:
    if not CPP_BACKEND_AVAILABLE:
        raise RuntimeError("C++ backend is required for reward evaluation")
    payload = cpp_evaluate_reward(board_before, row, col, player)
    reward = float(payload["reward"])
    return reward, payload


class GomokuEnv:
    """Single-agent environment where the agent plays against a built-in opponent."""

    _next_env_id = 0

    def __init__(self, opponent, seed: int | None = None):
        self.opponent = opponent
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.agent_player = BLACK
        self.done = False
        self.env_id = GomokuEnv._next_env_id
        GomokuEnv._next_env_id += 1
        self.current_update = 0

    def __del__(self):
        if CPP_BACKEND_AVAILABLE:
            try:
                cpp_clear_env_state(self.env_id)
            except Exception:
                pass

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.board.fill(EMPTY)
        self.done = False
        self.agent_player = BLACK if self.rng.random() < 0.5 else WHITE

        if self.agent_player == WHITE:
            opening = call_bot_action(self.opponent, self.board.copy(), BLACK, self.rng)
            row, col = action_to_coord(opening)
            self.board[row, col] = BLACK

        if CPP_BACKEND_AVAILABLE:
            cpp_reset_env_state(self.env_id, self.board)

        return self.observation(), self.action_mask()

    def observation(self) -> np.ndarray:
        current = self.agent_player
        obs = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.board == current).astype(np.float32)
        obs[1] = (self.board == -current).astype(np.float32)
        obs[2] = (self.board == EMPTY).astype(np.float32)
        return obs

    def action_mask(self) -> np.ndarray:
        return neighboring_action_mask(self.board, radius=2, opening_radius=1)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        if CPP_BACKEND_AVAILABLE:
            cpp_reset_env_state(self.env_id, self.board)

        row, col = action_to_coord(action)
        if not inside(row, col) or self.board[row, col] != EMPTY:
            self.done = True
            return StepResult(
                self.observation(),
                self.action_mask(),
                -TERMINAL_REWARD,
                True,
                {"illegal_move": True, "agent_result": "loss"},
            )

        reward_payload = cpp_evaluate_env_reward(self.env_id, row, col, self.agent_player)
        reward = float(reward_payload["reward"])
        info = reward_payload
        self.board[row, col] = self.agent_player
        cpp_apply_env_move(self.env_id, row, col, self.agent_player)

        if cpp_env_done(self.env_id):
            self.done = True
            winner = cpp_env_winner(self.env_id)
            if winner == self.agent_player:
                info["winner"] = self.agent_player
                info["agent_result"] = "win"
            else:
                info["draw"] = True
                info["agent_result"] = "draw"
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        opponent_action = None
        if FORCE_FOUR_TRAINING_UNTIL > 0 and self.current_update <= FORCE_FOUR_TRAINING_UNTIL:
            if _has_live_three(self.board, -self.agent_player):
                forced = _find_move_to_make_four(self.board, -self.agent_player)
                if forced is not None:
                    opponent_action = forced

        if opponent_action is None:
            opponent_action = call_bot_action(self.opponent, self.board.copy(), -self.agent_player, self.rng)
        opp_row, opp_col = action_to_coord(opponent_action)
        if self.board[opp_row, opp_col] != EMPTY:
            fallback = int(self.rng.choice(empty_actions(self.board)))
            opp_row, opp_col = action_to_coord(fallback)
            opponent_action = fallback

        self.board[opp_row, opp_col] = -self.agent_player
        cpp_apply_env_move(self.env_id, opp_row, opp_col, -self.agent_player)
        opp_patterns = classify_move(self.board, opp_row, opp_col, -self.agent_player)
        info["opponent_action"] = opponent_action
        info["opponent_pattern"] = opp_patterns
        if cpp_env_done(self.env_id):
            self.done = True
            winner = cpp_env_winner(self.env_id)
            if winner == -self.agent_player:
                info["winner"] = -self.agent_player
                info["agent_result"] = "loss"
                return StepResult(
                    self.observation(),
                    self.action_mask(),
                    reward - TERMINAL_REWARD,
                    True,
                    info,
                )
            info["draw"] = True
            info["agent_result"] = "draw"
            return StepResult(
                self.observation(),
                self.action_mask(),
                reward,
                True,
                info,
            )

        return StepResult(self.observation(), self.action_mask(), reward, False, info)


def call_bot_action(opponent, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    if hasattr(opponent, "next_action"):
        return int(opponent.next_action(board, player, rng))
    if hasattr(opponent, "select_action"):
        return int(opponent.select_action(board, player, rng))
    raise TypeError("opponent must implement next_action(board, player, rng) or select_action(board, player, rng)")


def _subproc_vector_worker(connection, env_specs: list[dict]) -> None:
    from bots import create_bot

    envs = [
        GomokuEnv(
            opponent=create_bot(name=spec["bot_name"], difficulty=(spec["bot_difficulty"] or None)),
            seed=int(spec["seed"]),
        )
        for spec in env_specs
    ]
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                obs, masks = zip(*(env.reset() for env in envs))
                connection.send((np.stack(obs), np.stack(masks)))
                continue
            
            if command == "set_update":
                update = int(payload)
                for env in envs:
                    env.current_update = update
                connection.send(True)
                continue

            if command == "step":
                actions = payload
                next_obs = []
                next_masks = []
                rewards = []
                dones = []
                infos: list[dict] = []
                for env, action in zip(envs, actions):
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

                connection.send(
                    (
                        np.stack(next_obs),
                        np.stack(next_masks),
                        np.asarray(rewards, dtype=np.float32),
                        np.asarray(dones, dtype=np.float32),
                        infos,
                    )
                )
                continue

            if command == "close":
                break

            raise ValueError(f"unsupported worker command: {command}")
    finally:
        connection.close()


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


class SubprocVectorEnv:
    def __init__(self, env_specs: list[dict], num_workers: int = 8, envs_per_worker: int = 2):
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if envs_per_worker <= 0:
            raise ValueError("envs_per_worker must be positive")
        if len(env_specs) == 0:
            raise ValueError("env_specs must not be empty")

        self.ctx = mp.get_context("spawn")
        self.parents = []
        self.processes = []
        self.worker_sizes: list[int] = []
        self.num_envs = len(env_specs)

        for start in range(0, len(env_specs), envs_per_worker):
            chunk = env_specs[start : start + envs_per_worker]
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_subproc_vector_worker,
                args=(child_conn, chunk),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.processes.append(process)
            self.worker_sizes.append(len(chunk))
    
    def set_current_update(self, update: int) -> None:
        for parent in self.parents:
            parent.send(("set_update", int(update)))
        for parent in self.parents:
            parent.recv()

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for parent in self.parents:
            parent.send(("reset", None))
        results = [parent.recv() for parent in self.parents]
        obs = [item[0] for item in results]
        masks = [item[1] for item in results]
        return np.concatenate(obs, axis=0), np.concatenate(masks, axis=0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")

        offset = 0
        for parent, worker_size in zip(self.parents, self.worker_sizes):
            parent.send(("step", actions[offset : offset + worker_size].tolist()))
            offset += worker_size

        results = [parent.recv() for parent in self.parents]
        next_obs = np.concatenate([item[0] for item in results], axis=0)
        next_masks = np.concatenate([item[1] for item in results], axis=0)
        rewards = np.concatenate([item[2] for item in results], axis=0)
        dones = np.concatenate([item[3] for item in results], axis=0)
        infos: list[dict] = []
        for item in results:
            infos.extend(item[4])
        return next_obs, next_masks, rewards, dones, infos

    def close(self) -> None:
        for parent in self.parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for parent in self.parents:
            try:
                parent.close()
            except OSError:
                pass
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
