from __future__ import annotations

import numpy as np

from bots import create_bot
from gomoku_ai.cpp_backend import decode_reward_events
from gomoku_ai.env import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    GomokuEnv,
    WHITE,
    classify_move,
    classify_move_counts,
    coord_to_action,
    evaluate_reward,
    neighboring_action_mask,
)


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


def test_decode_reward_events_returns_readable_names() -> None:
    decoded = decode_reward_events([(272, 2), (100528, -1)])

    assert decoded[0]["side"] == "self"
    assert decoded[0]["state_name"] == "live_one"
    assert decoded[0]["count"] == 2
    assert decoded[1]["side"] == "opponent"
    assert decoded[1]["state_name"] == "live_two"
    assert decoded[1]["count"] == -1


def test_env_illegal_move_ends_episode() -> None:
    env = GomokuEnv(opponent=StaticOpponent(coord_to_action(0, 1)), seed=1)
    env.board[0, 0] = BLACK
    env.done = False

    result = env.step(coord_to_action(0, 0))

    assert result.done is True
    assert result.info["illegal_move"] is True
    assert result.info["agent_result"] == "loss"


def test_env_reports_win_before_opponent_turn() -> None:
    from gomoku_ai.cpp_backend import reset_env_state

    env = GomokuEnv(opponent=StaticOpponent(coord_to_action(0, 0)), seed=1)
    env.agent_player = BLACK
    env.done = False
    env.board.fill(EMPTY)
    place_many(env.board, [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])
    reset_env_state(env.env_id, env.board)

    result = env.step(coord_to_action(7, 7))

    assert result.done is True
    assert result.info["agent_result"] == "win"
    assert result.reward > 0.0


def test_neighboring_action_mask_limits_single_stone_to_radius_one() -> None:
    board = empty_board()
    board[7, 7] = BLACK

    mask = neighboring_action_mask(board, radius=2, opening_radius=1).reshape(BOARD_SIZE, BOARD_SIZE)

    assert mask[6, 6]
    assert mask[7, 8]
    assert mask[8, 7]
    assert not mask[5, 7]
    assert not mask[7, 9]


def test_neighboring_action_mask_uses_radius_two_after_multiple_stones() -> None:
    board = empty_board()
    place_many(board, [(7, 7, BLACK), (9, 9, -BLACK)])

    mask = neighboring_action_mask(board, radius=2, opening_radius=1).reshape(BOARD_SIZE, BOARD_SIZE)

    assert mask[5, 7]
    assert mask[11, 9]
    assert mask[7, 5]
    assert not mask[4, 7]
    assert not mask[12, 9]


def test_reward_driven_bot_can_be_constructed() -> None:
    bot = create_bot(name="reward_driven_hard")

    assert bot is not None


# === 4-channel observation tests ===

def test_observation_has_4_channels() -> None:
    env = GomokuEnv(opponent=create_bot(name="random"), seed=42)
    obs, _ = env.reset()
    assert obs.shape == (4, 15, 15), f"Expected (4, 15, 15), got {obs.shape}"


def test_observation_channel_0_is_current_player_stones() -> None:
    env = GomokuEnv(opponent=create_bot(name="random"), seed=42)
    env.board.fill(EMPTY)
    env.board[7, 7] = env.agent_player
    env.board[3, 3] = -env.agent_player
    obs = env.observation()
    assert obs[0, 7, 7] == 1.0, "Channel 0 should mark current player's stone"
    assert obs[0, 3, 3] == 0.0, "Channel 0 should not mark opponent's stone"


def test_observation_channel_1_is_opponent_stones() -> None:
    env = GomokuEnv(opponent=create_bot(name="random"), seed=42)
    env.board.fill(EMPTY)
    env.board[7, 7] = env.agent_player
    env.board[3, 3] = -env.agent_player
    obs = env.observation()
    assert obs[1, 3, 3] == 1.0, "Channel 1 should mark opponent's stone"
    assert obs[1, 7, 7] == 0.0, "Channel 1 should not mark current player's stone"


def test_observation_channel_2_last_move_all_zeros_on_first_move() -> None:
    env = GomokuEnv(opponent=create_bot(name="random"), seed=42)
    env.board.fill(EMPTY)
    env.last_move = None
    obs = env.observation()
    assert np.all(obs[2] == 0.0), "Channel 2 should be all zeros when no last move"


def test_observation_channel_2_last_move_after_agent_move() -> None:
    env = GomokuEnv(opponent=StaticOpponent(coord_to_action(0, 0)), seed=42)
    env.board.fill(EMPTY)
    env.done = False
    env.agent_player = BLACK
    env.last_move = None
    from gomoku_ai.cpp_backend import reset_env_state

    reset_env_state(env.env_id, env.board)
    # place a stone that would lead to a continuing game
    result = env.step(coord_to_action(7, 7))
    if not result.done:
        # last_move should be the opponent's move (7,7 was agent, then opponent moved)
        obs = env.observation()
        assert obs[2].sum() == 1.0, "Channel 2 should have exactly one 1.0 for last move"
        # The last_move should be the opponent's position


def test_observation_channel_3_is_black_constant_plane() -> None:
    env = GomokuEnv(opponent=create_bot(name="random"), seed=42)
    env.board.fill(EMPTY)
    env.agent_player = BLACK
    obs = env.observation()
    assert np.all(obs[3] == 1.0), "Channel 3 should be all 1.0 for black player"

    env.agent_player = WHITE
    obs = env.observation()
    assert np.all(obs[3] == 0.0), "Channel 3 should be all 0.0 for white player"


def test_observation_maintains_player_perspective_after_switch() -> None:
    env = GomokuEnv(opponent=create_bot(name="random"), seed=42)
    env.board.fill(EMPTY)
    env.board[7, 7] = BLACK
    env.board[3, 3] = WHITE

    # From BLACK's perspective
    env.agent_player = BLACK
    obs = env.observation()
    assert obs[0, 7, 7] == 1.0, "BLACK's stones should be in channel 0"
    assert obs[1, 3, 3] == 1.0, "WHITE's stones should be in channel 1"

    # From WHITE's perspective
    env.agent_player = WHITE
    obs = env.observation()
    assert obs[0, 3, 3] == 1.0, "WHITE's stones should be in channel 0"
    assert obs[1, 7, 7] == 1.0, "BLACK's stones should be in channel 1"


# === Reward scaling tests ===

def test_terminal_reward_is_scaled() -> None:
    from gomoku_ai.env import TERMINAL_WIN_REWARD, TERMINAL_LOSS_REWARD

    assert TERMINAL_WIN_REWARD == 1.0
    assert TERMINAL_LOSS_REWARD == -1.0


def test_env_step_info_contains_reward_breakdown() -> None:
    env = GomokuEnv(opponent=StaticOpponent(coord_to_action(0, 0)), seed=42)
    env.board.fill(EMPTY)
    env.done = False
    env.agent_player = BLACK
    env.last_move = None
    from gomoku_ai.cpp_backend import reset_env_state

    reset_env_state(env.env_id, env.board)
    result = env.step(coord_to_action(7, 7))

    assert "terminal_reward" in result.info, "info should contain terminal_reward"
    assert "raw_auxiliary_reward" in result.info, "info should contain raw_auxiliary_reward"


# === Opponent pool tests ===

def test_opponent_pool_switches_bots() -> None:
    bots = [create_bot(name="random"), create_bot(name="reward_driven_hard")]
    weights = [0.5, 0.5]
    env = GomokuEnv(
        opponent=create_bot(name="random"),
        seed=42,
        opponent_pool=bots,
        opponent_weights=weights,
    )
    # Reset a few times and check that opponent changes
    seen_names = set()
    for _ in range(10):
        obs, mask = env.reset()
        seen_names.add(env.opponent.name)
    assert len(seen_names) > 1, f"Expected multiple bots, got {seen_names}"
