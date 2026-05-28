from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gomoku_ai.cpp_backend import score_reward_candidates
from gomoku_ai.env import EMPTY, action_to_coord, immediate_winning_actions, neighboring_action_mask


@dataclass(frozen=True)
class TacticalSearchConfig:
    candidate_top_k: int = 24
    opponent_top_k: int = 24
    reply_top_k: int = 4
    opponent_weight: float = 0.85
    followup_weight: float = 0.35
    policy_weight: float = 0.03
    opening_policy_weight: float = 0.01
    opening_extension_bonus: float = 20.0
    opening_self_distance_bonus: float = 5.0
    immediate_win_score: float = 1_000_000.0
    immediate_block_score: float = 500_000.0
    opponent_win_penalty: float = 750_000.0
    followup_win_score: float = 500_000.0


def _policy_value(policy_logits: np.ndarray | None, action: int) -> float:
    if policy_logits is None:
        return 0.0
    value = float(policy_logits[int(action)])
    if not np.isfinite(value):
        return -1_000_000.0
    return value


def _best_by_policy(actions: list[int], policy_logits: np.ndarray | None) -> int:
    return max(actions, key=lambda action: _policy_value(policy_logits, action))


def _score_candidates(board: np.ndarray, actions: list[int], player: int) -> list[tuple[float, int]]:
    if not actions:
        return []
    scored = score_reward_candidates(board, actions, player)
    return sorted(
        ((float(item["reward"]), int(item["action"])) for item in scored),
        key=lambda item: item[0],
        reverse=True,
    )


def _manhattan(row: int, col: int, point: np.ndarray) -> int:
    return abs(row - int(point[0])) + abs(col - int(point[1]))


def _candidate_actions(board: np.ndarray) -> list[int]:
    mask = neighboring_action_mask(board, radius=2, opening_radius=1)
    actions = np.flatnonzero(mask).astype(np.int64).tolist()
    if not actions:
        actions = np.flatnonzero(board.reshape(-1) == EMPTY).astype(np.int64).tolist()
    return actions


def _best_reply_score(board: np.ndarray, player: int, limit: int) -> float:
    actions = _candidate_actions(board)
    winning_actions = immediate_winning_actions(board.copy(), player, actions)
    if winning_actions:
        return 750_000.0
    scored = _score_candidates(board, actions, player)
    if limit > 0:
        scored = scored[:limit]
    return float(scored[0][0]) if scored else 0.0


def _select_opening_extension(
    board: np.ndarray,
    player: int,
    policy_logits: np.ndarray | None,
    cfg: TacticalSearchConfig,
) -> tuple[int, str, dict[str, float]] | None:
    own = np.argwhere(board == player)
    opponent = np.argwhere(board == -player)
    if len(own) != 1 or len(opponent) != 1 or np.count_nonzero(board != EMPTY) != 2:
        return None

    actions = _candidate_actions(board)
    scored = _score_candidates(board, actions, player)
    best_action = int(scored[0][1])
    best_score = -float("inf")
    best_payload: dict[str, float] = {}
    own_point = own[0]
    opponent_point = opponent[0]

    for reward_score, action in scored:
        row, col = action_to_coord(action)
        own_distance = _manhattan(row, col, own_point)
        opponent_distance = _manhattan(row, col, opponent_point)
        extension_bonus = cfg.opening_extension_bonus if 3 <= opponent_distance <= 4 else 0.0
        score = (
            float(reward_score)
            + extension_bonus
            + cfg.opening_self_distance_bonus * min(own_distance, 6)
            + cfg.opening_policy_weight * _policy_value(policy_logits, action)
        )
        if score > best_score:
            best_score = score
            best_action = int(action)
            best_payload = {
                "opening_score": float(score),
                "opening_reward_score": float(reward_score),
                "opening_own_distance": float(own_distance),
                "opening_opponent_distance": float(opponent_distance),
            }

    return best_action, "model_opening_extension", best_payload


def select_tactical_search_action(
    board: np.ndarray,
    player: int,
    policy_logits: np.ndarray | None = None,
    config: TacticalSearchConfig | None = None,
) -> tuple[int, str, dict[str, float]]:
    cfg = config or TacticalSearchConfig()
    board = np.asarray(board, dtype=np.int8)

    winning_actions = immediate_winning_actions(board.copy(), player)
    if winning_actions:
        action = _best_by_policy(winning_actions, policy_logits)
        return action, "model_tactical_win", {"search_score": cfg.immediate_win_score}

    blocking_actions = immediate_winning_actions(board.copy(), -player)
    blocking_set = set(blocking_actions)

    if not blocking_actions:
        opening_action = _select_opening_extension(board, player, policy_logits, cfg)
        if opening_action is not None:
            return opening_action

    base_actions = _candidate_actions(board)
    scored_self = _score_candidates(board, base_actions, player)
    if cfg.candidate_top_k > 0:
        scored_self = scored_self[: cfg.candidate_top_k]
    if blocking_actions:
        present = {action for _, action in scored_self}
        for action in blocking_actions:
            if action not in present:
                scored_self.append((cfg.immediate_block_score, int(action)))

    best_action = int(scored_self[0][1]) if scored_self else int(base_actions[0])
    best_score = -float("inf")
    best_opp_score = 0.0

    for self_score, action in scored_self:
        row, col = action_to_coord(action)
        if board[row, col] != EMPTY:
            continue

        next_board = board.copy()
        next_board[row, col] = player
        opponent_actions = _candidate_actions(next_board)
        opponent_wins = immediate_winning_actions(next_board.copy(), -player, opponent_actions)
        if opponent_wins:
            opponent_score = cfg.opponent_win_penalty
            followup_score = 0.0
        else:
            scored_opp = _score_candidates(next_board, opponent_actions, -player)
            if cfg.opponent_top_k > 0:
                scored_opp = scored_opp[: cfg.opponent_top_k]
            opponent_score = scored_opp[0][0] if scored_opp else 0.0
            reply_scores: list[float] = []
            for _, opponent_action in scored_opp[: max(1, cfg.reply_top_k)]:
                opp_row, opp_col = action_to_coord(opponent_action)
                if next_board[opp_row, opp_col] != EMPTY:
                    continue
                reply_board = next_board.copy()
                reply_board[opp_row, opp_col] = -player
                own_wins = immediate_winning_actions(reply_board.copy(), player)
                if own_wins:
                    reply_scores.append(cfg.followup_win_score)
                    continue
                reply_scores.append(_best_reply_score(reply_board, player, cfg.opponent_top_k))
            followup_score = min(reply_scores) if reply_scores else 0.0

        block_bonus = cfg.immediate_block_score if action in blocking_set else 0.0
        policy_bonus = cfg.policy_weight * _policy_value(policy_logits, action)
        search_score = (
            float(self_score)
            + block_bonus
            - cfg.opponent_weight * opponent_score
            + cfg.followup_weight * followup_score
            + policy_bonus
        )
        if search_score > best_score:
            best_score = search_score
            best_action = int(action)
            best_opp_score = float(opponent_score)

    source = "model_tactical_search"
    if best_action in blocking_set:
        source = "model_tactical_block"
    return best_action, source, {"search_score": float(best_score), "opponent_reply_score": best_opp_score}
