from __future__ import annotations

from pathlib import Path

import torch

from gomoku_ai.checkpoint_opponent import CheckpointOpponent
from gomoku_ai.env import BOARD_SIZE, EMPTY
from gomoku_ai.model import ActorCriticNet, model_preset_config
from scripts.train_common import AlphaSchedule, StageConfig, to_jsonable


def test_alpha_schedule_decays_after_warmup() -> None:
    schedule = AlphaSchedule(start=0.2, end=0.02, warmup_fraction=0.3, decay_fraction=0.4)
    assert schedule.value(1, 100) == 0.2
    assert schedule.value(90, 100) == 0.02
    middle = schedule.value(50, 100)
    assert 0.02 < middle < 0.2


def test_stage_config_is_jsonable() -> None:
    stage = StageConfig(
        stage_name="test",
        description="test stage",
        reward_mode="terminal",
        shape_alpha=0.0,
        use_action_mask=True,
    )
    payload = to_jsonable({"stage": stage, "path": Path("runs/test")})
    assert payload["path"] == "runs\\test" or payload["path"] == "runs/test"


def test_resolve_device_checkpoint_compatible() -> None:
    tensor = torch.tensor([1.0])
    assert float(tensor.item()) == 1.0


def test_checkpoint_opponent_selects_legal_action(tmp_path) -> None:
    model = ActorCriticNet(model_preset_config("small"))
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
        },
        checkpoint,
    )
    opponent = CheckpointOpponent(checkpoint, device="cpu")
    board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.int8).numpy()
    board[7, 7] = 1
    action = opponent.next_action(board, -1, torch.Generator())  # type: ignore[arg-type]
    row, col = divmod(action, BOARD_SIZE)
    assert board[row, col] == EMPTY
