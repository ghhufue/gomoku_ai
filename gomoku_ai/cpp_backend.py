from __future__ import annotations

try:
    from gomoku_ai import _cpp_backend as _backend
except ImportError:
    _backend = None


BACKEND_AVAILABLE = _backend is not None


def classify_move_counts(board, row: int, col: int, player: int):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return _backend.classify_move_counts(board, row, col, player)


def immediate_winning_actions(board, player: int, actions=None):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return list(_backend.immediate_winning_actions(board, player, actions))


def threat_summary(board, player: int, actions=None):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return dict(_backend.threat_summary(board, player, actions))


def affected_actions(board, row: int, col: int, radius: int = 5):
    if _backend is None:
        raise RuntimeError("C++ backend is not available")
    return list(_backend.affected_actions(board, row, col, radius))
