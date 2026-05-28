from __future__ import annotations

import numpy as np
import pytest

from bots import create_bot
from gomoku_ai.cpp_backend import BACKEND_AVAILABLE
from gomoku_ai.env import BLACK, EMPTY, EnvConfig, GomokuEnv, action_to_coord, coord_to_action


def make_env(**kwargs) -> GomokuEnv:
    return GomokuEnv(
        opponent=create_bot(name="random"),
        seed=123,
        config=EnvConfig(**kwargs),
    )


def test_reset_returns_observation_and_mask() -> None:
    env = make_env(use_action_mask=True)
    obs, mask = env.reset()
    assert obs.shape == (3, 15, 15)
    assert mask.shape == (225,)
    assert mask.dtype == bool
    assert mask.any()


def test_action_mask_excludes_occupied_points() -> None:
    env = make_env(use_action_mask=True)
    _, mask = env.reset()
    action = int(np.flatnonzero(mask)[0])
    row, col = action_to_coord(action)
    env.board[row, col] = BLACK
    next_mask = env.action_mask()
    assert not bool(next_mask[action])


def test_illegal_move_ends_episode_as_loss() -> None:
    env = make_env(use_action_mask=True)
    env.reset()
    env.board[7, 7] = BLACK
    result = env.step(coord_to_action(7, 7))
    assert result.done
    assert result.reward < 0
    assert result.info["illegal_move"] is True
    assert result.info["agent_result"] == "loss"


def test_terminal_win_reward() -> None:
    env = make_env(use_action_mask=False)
    env.reset()
    env.agent_player = BLACK
    env.board.fill(EMPTY)
    env.board[7, 3:7] = BLACK
    result = env.step(coord_to_action(7, 7))
    assert result.done
    assert result.reward > 0
    assert result.info["agent_result"] == "win"


@pytest.mark.skipif(not BACKEND_AVAILABLE, reason="C++ backend is not built")
def test_shaped_reward_uses_cpp_backend() -> None:
    env = make_env(reward_mode="shaped", shape_alpha=0.05, use_action_mask=True)
    _, mask = env.reset()
    action = int(np.flatnonzero(mask)[0])
    result = env.step(action)
    assert "shape_reward" in result.info["reward_components"]
