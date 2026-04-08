from pathlib import Path
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from bots.factory import create_bot
from bots.trained_bot import TrainedBot
from gomoku_ai.env import call_bot_action, BOARD_SIZE

app = FastAPI()

class BotRequest(BaseModel):
    board: list[list[int]]
    current_player: int
    bot_name: str = "classic_rule"
    bot_difficulty: str | None = None
    model_path: str | None = None

@app.post("/bot_move")
def bot_move(req: BotRequest):
    board = np.array(req.board, dtype=np.int8)
    if req.model_path:
        bot = TrainedBot(model_path=req.model_path)
    else:
        bot = create_bot(name=req.bot_name, difficulty=req.bot_difficulty)

    rng = np.random.default_rng()
    action = call_bot_action(bot, board, req.current_player, rng)

    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    return {"action": int(action), "row": int(row), "col": int(col)}

@app.get("/list_runs")
def list_runs():
    """列出所有可用的训练模型"""
    runs_dir = Path(__file__).parent / "runs"
    models = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            final_model = run_dir / "final_model.pt"
            best_model = run_dir / "checkpoints" / "best_model.pt"
            if final_model.exists():
                models.append({
                    "run_id": run_dir.name,
                    "label": run_dir.name + " (final)",
                    "path": str(final_model)
                })
            if best_model.exists():
                models.append({
                    "run_id": run_dir.name,
                    "label": run_dir.name + " (best)",
                    "path": str(best_model)
                })
    return {"models": models}