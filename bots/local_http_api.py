from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from bots.local_http_bridge import LocalBotBridge, LocalBotBridgeError


class BotMoveRequest(BaseModel):
    board: list[list[int]]
    current_player: int
    bot_name: str = Field(min_length=1)


class WakeRequest(BaseModel):
    bot_name: str = Field(min_length=1)


bridge = LocalBotBridge()
app = FastAPI(title="Gomoku Local Bot Bridge")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "gomoku_ai.bots"}


@app.get("/bots")
def bots() -> dict[str, list[str]]:
    return {"bots": bridge.list_bots()}


@app.post("/wake")
def wake(request: WakeRequest) -> dict[str, str]:
    try:
        bot_name = bridge.wake_bot(request.bot_name)
    except LocalBotBridgeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ready", "bot_name": bot_name}


@app.post("/bot_move")
def bot_move(request: BotMoveRequest) -> dict[str, Any]:
    try:
        result = bridge.request_move(
            board=request.board,
            current_player=request.current_player,
            bot_name=request.bot_name,
        )
    except LocalBotBridgeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result.to_godot_response()


def main() -> None:
    uvicorn.run("bots.local_http_api:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    main()
