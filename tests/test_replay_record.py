from __future__ import annotations

import json

from tools.replay_record import load_record


def test_load_record_supports_simple_record_payload(tmp_path) -> None:
    record_path = tmp_path / "simple.json"
    record_path.write_text(
        json.dumps(
            {
                "name": "simple_case",
                "description": "simple record payload",
                "moves": [
                    {"row": 7, "col": 7, "color": "black", "source": "sample"},
                    {"row": 7, "col": 8, "color": "white", "source": "sample"},
                ],
            }
        ),
        encoding="utf-8",
    )

    record = load_record(record_path)

    assert record.name == "simple_case"
    assert len(record.steps) == 2
    assert record.steps[0].player_name == "black"
    assert record.steps[1].player_name == "white"
    assert record.metadata["game_count"] == 1


def test_load_record_supports_play_match_payload(tmp_path) -> None:
    record_path = tmp_path / "match.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "record_type": "gomoku_match_record",
                "checkpoint": "runs/demo/final_model.pt",
                "device": "cpu",
                "games": [
                    {
                        "game_index": 1,
                        "name": "game_seed_1",
                        "description": "game one",
                        "moves": [
                            {"move_number": 1, "player": "black", "source": "bot", "action": 112, "row": 7, "col": 7},
                            {"move_number": 2, "player": "white", "source": "model", "action": 113, "row": 7, "col": 8},
                        ],
                    },
                    {
                        "game_index": 2,
                        "name": "game_seed_2",
                        "description": "game two",
                        "moves": [
                            {"move_number": 1, "player": "black", "source": "bot", "action": 96, "row": 6, "col": 6},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    record = load_record(record_path, game_index=2)

    assert record.name == "game_seed_2"
    assert len(record.steps) == 1
    assert record.steps[0].player_name == "black"
    assert record.metadata["checkpoint"] == "runs/demo/final_model.pt"
    assert record.metadata["selected_game_index"] == 2
    assert record.metadata["game_count"] == 2
