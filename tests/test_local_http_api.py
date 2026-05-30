from __future__ import annotations

from fastapi.testclient import TestClient

from bots.local_http_api import app
from bots.local_http_bridge import LocalBotBridge
from gomoku_ai.env import BOARD_SIZE


def empty_board() -> list[list[int]]:
    return [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def test_local_bot_bridge_returns_godot_coordinates() -> None:
    bridge = LocalBotBridge(seed=123)

    result = bridge.request_move(empty_board(), current_player=1, bot_name="random")

    assert 0 <= result.row < BOARD_SIZE
    assert 0 <= result.col < BOARD_SIZE
    assert result.y == result.row
    assert result.x == result.col
    assert result.bot_name == "random"


def test_bot_bridge_lists_and_runs_bot() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    bots = client.get("/bots")
    assert bots.status_code == 200
    assert "random" in bots.json()["bots"]

    wake = client.post("/wake", json={"bot_name": "random"})
    assert wake.status_code == 200

    move = client.post(
        "/bot_move",
        json={"board": empty_board(), "current_player": 1, "bot_name": "random"},
    )
    assert move.status_code == 200
    payload = move.json()
    assert 0 <= payload["row"] < BOARD_SIZE
    assert 0 <= payload["col"] < BOARD_SIZE
    assert payload["x"] == payload["col"]
    assert payload["y"] == payload["row"]
