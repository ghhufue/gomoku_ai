from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bots.factory import available_bots, create_bot, resolve_bot_name
from gomoku_ai.env import BOARD_SIZE, EMPTY, action_to_coord


class LocalBotBridgeError(Exception):
    """Raised when a local Bot bridge request cannot be fulfilled."""


@dataclass
class BotMoveResult:
    row: int
    col: int
    bot_name: str
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.col

    @property
    def y(self) -> int:
        return self.row

    def to_godot_response(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "x": self.x,
            "y": self.y,
            "bot_name": self.bot_name,
            "debug": self.debug,
        }


class LocalBotBridge:
    """Bridge Godot HTTP move requests to gomoku_ai.bots implementations."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._bots: dict[str, Any] = {}

    def list_bots(self) -> list[str]:
        return available_bots()

    def wake_bot(self, bot_name: str) -> str:
        resolved_name = self._resolve_bot_name(bot_name)
        self._get_or_create_bot(resolved_name)
        return resolved_name

    def request_move(
        self,
        board: list[list[int]],
        current_player: int,
        bot_name: str,
    ) -> BotMoveResult:
        resolved_name = self.wake_bot(bot_name)
        board_array = self._to_board_array(board)
        self._validate_player(current_player)

        bot = self._get_or_create_bot(resolved_name)
        action = int(bot.next_action(board_array.copy(), int(current_player), self.rng))
        row, col = action_to_coord(action)

        self._validate_move(board_array, row, col)
        return BotMoveResult(
            row=row,
            col=col,
            bot_name=resolved_name,
            debug={"bridge": "gomoku_ai.bots", "action": action},
        )

    def _resolve_bot_name(self, bot_name: str) -> str:
        if not bot_name or not bot_name.strip():
            raise LocalBotBridgeError("bot_name is required for the Bot bridge")

        try:
            return resolve_bot_name(bot_name)
        except ValueError as exc:
            raise LocalBotBridgeError(
                f"unsupported bot '{bot_name}'. Use local_inference_service for models or MoveEngines."
            ) from exc

    def _get_or_create_bot(self, bot_name: str):
        bot = self._bots.get(bot_name)
        if bot is None:
            bot = create_bot(name=bot_name)
            self._bots[bot_name] = bot
        return bot

    def _to_board_array(self, board: list[list[int]]) -> np.ndarray:
        board_array = np.asarray(board, dtype=np.int8)
        if board_array.shape != (BOARD_SIZE, BOARD_SIZE):
            raise LocalBotBridgeError(f"board must be {BOARD_SIZE}x{BOARD_SIZE}")

        allowed = np.isin(board_array, [EMPTY, 1, -1])
        if not bool(allowed.all()):
            raise LocalBotBridgeError("board contains unsupported cell values")

        return board_array

    def _validate_player(self, current_player: int) -> None:
        if current_player not in (1, -1):
            raise LocalBotBridgeError("current_player must be 1 or -1")

    def _validate_move(self, board: np.ndarray, row: int, col: int) -> None:
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise LocalBotBridgeError("bot returned an out-of-bounds move")
        if int(board[row, col]) != EMPTY:
            raise LocalBotBridgeError("bot returned an occupied move")
