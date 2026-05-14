from fastapi.testclient import TestClient

from bots.local_http_api import app
from bots.local_http_bridge import LocalBotBridge, LocalBotBridgeError
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


def test_local_bot_bridge_rejects_unknown_bot() -> None:
    bridge = LocalBotBridge(seed=123)

    try:
        bridge.request_move(empty_board(), current_player=1, bot_name="trained_model")
    except LocalBotBridgeError as exc:
        assert "local_inference_service" in str(exc)
    else:
        raise AssertionError("expected LocalBotBridgeError")


def test_local_http_api_bot_move() -> None:
    client = TestClient(app)

    response = client.post(
        "/bot_move",
        json={
            "board": empty_board(),
            "current_player": 1,
            "bot_name": "random",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["row"] < BOARD_SIZE
    assert 0 <= payload["col"] < BOARD_SIZE
    assert payload["x"] == payload["col"]
    assert payload["y"] == payload["row"]
    assert payload["bot_name"] == "random"


def test_local_http_api_bots() -> None:
    client = TestClient(app)

    response = client.get("/bots")

    assert response.status_code == 200
    assert "random" in response.json()["bots"]
