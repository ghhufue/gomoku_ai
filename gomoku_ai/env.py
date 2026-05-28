from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Iterable

import numpy as np

from gomoku_ai.cpp_backend import (
    BACKEND_AVAILABLE as CPP_BACKEND_AVAILABLE,
    apply_env_move as cpp_apply_env_move,
    clear_env_state as cpp_clear_env_state,
    evaluate_env_reward as cpp_evaluate_env_reward,
    reset_env_state as cpp_reset_env_state,
)


BOARD_SIZE = 15
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
WIN_LENGTH = 5

EMPTY = 0
BLACK = 1
WHITE = -1

DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass
class EnvConfig:
    reward_mode: str = "terminal"
    terminal_reward: float = 1.0
    shape_alpha: float = 0.0
    use_action_mask: bool = True
    candidate_radius: int = 2
    opening_radius: int = 1


@dataclass
class StepResult:
    observation: np.ndarray
    action_mask: np.ndarray
    reward: float
    done: bool
    info: dict


def action_to_coord(action: int) -> tuple[int, int]:
    return divmod(int(action), BOARD_SIZE)


def coord_to_action(row: int, col: int) -> int:
    return int(row) * BOARD_SIZE + int(col)


def inside(row: int, col: int) -> bool:
    return 0 <= int(row) < BOARD_SIZE and 0 <= int(col) < BOARD_SIZE


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


def full_action_mask(board: np.ndarray) -> np.ndarray:
    return (board.reshape(-1) == EMPTY)


def check_win(board: np.ndarray, row: int, col: int, player: int) -> bool:
    for dr, dc in DIRS:
        count = 1
        for sign in (-1, 1):
            current_row = row + sign * dr
            current_col = col + sign * dc
            while inside(current_row, current_col) and board[current_row, current_col] == player:
                count += 1
                current_row += sign * dr
                current_col += sign * dc
        if count >= WIN_LENGTH:
            return True
    return False


def board_full(board: np.ndarray) -> bool:
    return not np.any(board == EMPTY)


def call_bot_action(opponent, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    if hasattr(opponent, "next_action"):
        return int(opponent.next_action(board, player, rng))
    if hasattr(opponent, "select_action"):
        return int(opponent.select_action(board, player, rng))
    raise TypeError("opponent must implement next_action(board, player, rng) or select_action(board, player, rng)")


class GomokuEnv:
    """Single-agent Gomoku environment used by all workshop training stages."""

    _next_env_id = 0

    def __init__(
        self,
        opponent,
        seed: int | None = None,
        config: EnvConfig | None = None,
    ):
        self.opponent = opponent
        self.config = config or EnvConfig()
        if self.config.reward_mode not in {"terminal", "shaped"}:
            raise ValueError("reward_mode must be 'terminal' or 'shaped'")
        if self.config.reward_mode == "shaped" and self.config.shape_alpha > 0.0 and not CPP_BACKEND_AVAILABLE:
            raise RuntimeError("C++ backend is required for shaped rewards")

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

    def set_shape_alpha(self, shape_alpha: float) -> None:
        self.config.shape_alpha = float(shape_alpha)

    def observation(self) -> np.ndarray:
        current = self.agent_player
        obs = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.board == current).astype(np.float32)
        obs[1] = (self.board == -current).astype(np.float32)
        obs[2] = (self.board == EMPTY).astype(np.float32)
        return obs

    def action_mask(self) -> np.ndarray:
        if not self.config.use_action_mask:
            return np.ones(BOARD_AREA, dtype=bool)
        return neighboring_action_mask(
            self.board,
            radius=self.config.candidate_radius,
            opening_radius=self.config.opening_radius,
        )

    def _shape_reward(self, row: int, col: int) -> tuple[float, dict]:
        if self.config.reward_mode != "shaped" or self.config.shape_alpha <= 0.0:
            return 0.0, {}
        payload = cpp_evaluate_env_reward(self.env_id, row, col, self.agent_player)
        raw_reward = float(payload.get("reward", 0.0))
        return self.config.shape_alpha * raw_reward, payload

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        if CPP_BACKEND_AVAILABLE:
            cpp_reset_env_state(self.env_id, self.board)

        row, col = action_to_coord(action)
        if not inside(row, col) or self.board[row, col] != EMPTY:
            self.done = True
            reward = -self.config.terminal_reward
            return StepResult(
                self.observation(),
                self.action_mask(),
                reward,
                True,
                {
                    "illegal_move": True,
                    "agent_result": "loss",
                    "reward_components": {"terminal_reward": reward, "shape_reward": 0.0, "total_reward": reward},
                },
            )

        shape_reward, shape_payload = self._shape_reward(row, col)
        terminal_reward = 0.0
        info = dict(shape_payload)
        self.board[row, col] = self.agent_player
        if CPP_BACKEND_AVAILABLE:
            cpp_apply_env_move(self.env_id, row, col, self.agent_player)

        if check_win(self.board, row, col, self.agent_player):
            self.done = True
            terminal_reward = self.config.terminal_reward
            reward = terminal_reward + shape_reward
            info.update({"winner": self.agent_player, "agent_result": "win"})
            info["reward_components"] = {
                "terminal_reward": terminal_reward,
                "shape_reward": shape_reward,
                "total_reward": reward,
            }
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        if board_full(self.board):
            self.done = True
            reward = shape_reward
            info.update({"draw": True, "agent_result": "draw"})
            info["reward_components"] = {
                "terminal_reward": 0.0,
                "shape_reward": shape_reward,
                "total_reward": reward,
            }
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        opponent_action = call_bot_action(self.opponent, self.board.copy(), -self.agent_player, self.rng)
        opp_row, opp_col = action_to_coord(opponent_action)
        if self.board[opp_row, opp_col] != EMPTY:
            fallback = int(self.rng.choice(empty_actions(self.board)))
            opp_row, opp_col = action_to_coord(fallback)
            opponent_action = fallback

        self.board[opp_row, opp_col] = -self.agent_player
        if CPP_BACKEND_AVAILABLE:
            cpp_apply_env_move(self.env_id, opp_row, opp_col, -self.agent_player)
        info["opponent_action"] = int(opponent_action)

        if check_win(self.board, opp_row, opp_col, -self.agent_player):
            self.done = True
            terminal_reward = -self.config.terminal_reward
            reward = terminal_reward + shape_reward
            info.update({"winner": -self.agent_player, "agent_result": "loss"})
            info["reward_components"] = {
                "terminal_reward": terminal_reward,
                "shape_reward": shape_reward,
                "total_reward": reward,
            }
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        if board_full(self.board):
            self.done = True
            reward = shape_reward
            info.update({"draw": True, "agent_result": "draw"})
            info["reward_components"] = {
                "terminal_reward": 0.0,
                "shape_reward": shape_reward,
                "total_reward": reward,
            }
            return StepResult(self.observation(), self.action_mask(), reward, True, info)

        info["reward_components"] = {
            "terminal_reward": 0.0,
            "shape_reward": shape_reward,
            "total_reward": shape_reward,
        }
        return StepResult(self.observation(), self.action_mask(), shape_reward, False, info)


def _subproc_vector_worker(connection, env_specs: list[dict]) -> None:
    from bots import create_bot

    envs = [
        GomokuEnv(
            opponent=create_bot(name=spec["bot_name"], difficulty=(spec["bot_difficulty"] or None)),
            seed=int(spec["seed"]),
            config=spec["env_config"],
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

            if command == "set_shape_alpha":
                for env in envs:
                    env.set_shape_alpha(float(payload))
                connection.send(True)
                continue

            if command == "step":
                next_obs = []
                next_masks = []
                rewards = []
                dones = []
                infos: list[dict] = []
                for env, action in zip(envs, payload):
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

    def set_shape_alpha(self, shape_alpha: float) -> None:
        for env in self.envs:
            env.set_shape_alpha(shape_alpha)


class SubprocVectorEnv:
    def __init__(self, env_specs: list[dict], num_workers: int = 4, envs_per_worker: int = 2):
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

    def set_shape_alpha(self, shape_alpha: float) -> None:
        for parent in self.parents:
            parent.send(("set_shape_alpha", float(shape_alpha)))
        for parent in self.parents:
            parent.recv()

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for parent in self.parents:
            parent.send(("reset", None))
        results = [parent.recv() for parent in self.parents]
        return np.concatenate([item[0] for item in results]), np.concatenate([item[1] for item in results])

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
