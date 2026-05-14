from __future__ import annotations

import json
from pathlib import Path

from datetime import datetime

from scripts.evaluate import build_export_session_dir, build_match_payload, derive_model_label, export_match_outputs
from tools.replay_record import load_record


def sample_game(seed: int, move_row: int, move_col: int) -> dict[str, object]:
    return {
        "game_index": 1,
        "name": f"game_seed_{seed}",
        "description": "sample exported game",
        "seed": seed,
        "agent_player": "white",
        "bot_player": "black",
        "bot_name": "rule",
        "result": "loss",
        "illegal_move": False,
        "winner": "black",
        "total_reward": -10.0,
        "num_moves": 2,
        "moves": [
            {"move_number": 1, "player": "black", "color": "black", "source": "bot", "action": 112, "row": 7, "col": 7},
            {"move_number": 2, "player": "white", "color": "white", "source": "model", "action": move_row * 15 + move_col, "row": move_row, "col": move_col},
        ],
        "final_board_ascii": "board",
    }


def test_build_match_payload_assigns_game_indices() -> None:
    payload = build_match_payload(
        checkpoint_path=Path("runs/demo/final_model.pt"),
        device="cpu",
        games=[sample_game(123, 7, 5), sample_game(124, 7, 6)],
    )

    assert payload["schema_version"] == 2
    assert payload["record_type"] == "gomoku_match_record"
    assert payload["bot_name"] == "rule"
    assert payload["games"][0]["game_index"] == 1
    assert payload["games"][1]["game_index"] == 2


def test_derive_model_label_uses_run_name_and_checkpoint_name() -> None:
    assert derive_model_label(Path("runs/demo_run/final_model.pt")) == "demo_run_final_model"
    assert derive_model_label(Path("runs/demo_run/checkpoints/best_model.pt")) == "demo_run_best_model"


def test_build_export_session_dir_wraps_output_dir() -> None:
    export_dir = build_export_session_dir(
        Path("outputs/evaluation"),
        Path("runs/demo_run/checkpoints/best_model.pt"),
        now=datetime(2026, 3, 15, 16, 30, 45),
    )

    assert export_dir.as_posix() == "outputs/evaluation/20260315_163045_demo_run_best_model"


def test_exported_match_record_is_replay_compatible(tmp_path) -> None:
    games = [sample_game(123, 7, 5)]
    checkpoint_path = Path("runs/demo/final_model.pt")
    payload = build_match_payload(checkpoint_path, "cpu", games)
    output_paths = export_match_outputs(
        payload=payload,
        games=games,
        output_dir=tmp_path,
        filename="match_record",
        export_text=True,
        export_visual=True,
    )

    json_path = tmp_path / "match_record.json"
    assert json_path in output_paths
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["record_type"] == "gomoku_match_record"

    record = load_record(json_path)
    assert record.name == "game_seed_123"
    assert len(record.steps) == 2
    assert Path(record.metadata["checkpoint"]).as_posix() == "runs/demo/final_model.pt"
