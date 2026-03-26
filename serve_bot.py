from pathlib import Path
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from bots.factory import create_bot
from gomoku_ai.env import call_bot_action, BOARD_SIZE

app = FastAPI()

class BotRequest(BaseModel):
    board: list[list[int]]
    current_player: int
    bot_name: str = "classic_rule"
    bot_difficulty: str | None = None

@app.post("/bot_move")
def bot_move(req: BotRequest):
    board = np.array(req.board, dtype=np.int8)
    bot = create_bot(name=req.bot_name, difficulty=req.bot_difficulty)
    rng = np.random.default_rng()
    action = call_bot_action(bot, board, req.current_player, rng)

    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    return {"action": int(action), "row": int(row), "col": int(col)}