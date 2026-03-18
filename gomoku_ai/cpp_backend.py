from __future__ import annotations

try:
    from gomoku_ai import _cpp_backend as _backend
except ImportError:
    _backend = None


BACKEND_AVAILABLE = _backend is not None


def classify_move_counts(board, row: int, col: int, player: int):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    if not hasattr(_backend, "classify_move_counts"):
        raise RuntimeError("classify_move_counts is no longer exposed by the C++ backend")
    return _backend.classify_move_counts(board, row, col, player)


def immediate_winning_actions(board, player: int, actions=None):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    if not hasattr(_backend, "immediate_winning_actions"):
        raise RuntimeError("immediate_winning_actions is no longer exposed by the C++ backend")
    return list(_backend.immediate_winning_actions(board, player, actions))


def threat_summary(board, player: int, actions=None):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    if not hasattr(_backend, "threat_summary"):
        raise RuntimeError("threat_summary is no longer exposed by the C++ backend")
    return dict(_backend.threat_summary(board, player, actions))


def affected_actions(board, row: int, col: int, radius: int = 5):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    if not hasattr(_backend, "affected_actions"):
        raise RuntimeError("affected_actions is no longer exposed by the C++ backend")
    return list(_backend.affected_actions(board, row, col, radius))


def classify_move_shape_details(board, row: int, col: int, player: int, scales: dict[str, float]):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    if not hasattr(_backend, "classify_move_shape_details"):
        raise RuntimeError("classify_move_shape_details is no longer exposed by the C++ backend")
    return _backend.classify_move_shape_details(board, row, col, player, scales)


def extract_local_move_events(board, row: int, col: int, player: int, radius: int = 5):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    if not hasattr(_backend, "extract_local_move_events"):
        raise RuntimeError("extract_local_move_events is no longer exposed by the C++ backend")
    return _backend.extract_local_move_events(board, row, col, player, radius)


def evaluate_reward(board_before, row: int, col: int, player: int):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return dict(_backend.evaluate_reward(board_before, row, col, player))


def evaluate_env_reward(env_id: int, row: int, col: int, player: int):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return dict(_backend.evaluate_env_reward(env_id, row, col, player))


def score_classic_candidates(board_before, actions, player: int, thread_batch_size: int = 10):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return list(_backend.score_classic_candidates(board_before, actions, player, thread_batch_size))


def score_reward_candidates(board_before, actions, player: int, thread_batch_size: int = 10):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return list(_backend.score_reward_candidates(board_before, actions, player, thread_batch_size))


def reset_env_state(env_id: int, board):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    _backend.reset_env_state(env_id, board)


def clear_env_state(env_id: int):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    _backend.clear_env_state(env_id)


def apply_env_move(env_id: int, row: int, col: int, player: int):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    _backend.apply_env_move(env_id, row, col, player)


def env_done(env_id: int) -> bool:
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return bool(_backend.env_done(env_id))


def env_winner(env_id: int) -> int:
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return int(_backend.env_winner(env_id))


def list_state_values():
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return list(_backend.list_state_values())


def decode_reward_events(events):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    if not hasattr(_backend, "decode_reward_events"):
        raise RuntimeError("decode_reward_events is not exposed by the C++ backend")
    return list(_backend.decode_reward_events(events))


def debug_encode_direction_side_states(states):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return dict(_backend.debug_encode_direction_side_states(states))


def debug_decode_direction_lookup_key(lookup_key: int):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return dict(_backend.debug_decode_direction_lookup_key(lookup_key))


def debug_classify_direction_side_states(states):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return list(_backend.debug_classify_direction_side_states(states))
