from __future__ import annotations

import json
from pathlib import Path

from gomoku_ai.cpp_backend import BACKEND_AVAILABLE, debug_decode_direction_lookup_key, debug_encode_direction_side_states


CASE_FILE = Path(__file__).resolve().parent.parent / "cpp" / "test" / "direction_pattern_cases.json"

STATE_MAP = {
    "_": 0,
    "X": 1,
    "O": 2,
}


def line_to_side_states(line: str) -> list[int]:
    assert len(line) == 11
    assert line[5] in {"_", "X"}

    normalized = line[:5] + "X" + line[6:]
    return [STATE_MAP[char] for char in normalized[:5] + normalized[6:]]


def load_cases() -> list[dict[str, object]]:
    return json.loads(CASE_FILE.read_text(encoding="utf-8"))


def test_direction_pattern_cases_roundtrip_decode_exact_side_states() -> None:
    if not BACKEND_AVAILABLE:
        return

    for case in load_cases():
        side_states = line_to_side_states(str(case["line"]))
        encoded = debug_encode_direction_side_states(side_states)
        decoded = debug_decode_direction_lookup_key(int(encoded["lookup_key"]))

        assert decoded["side_states"] == side_states, case["name"]
        assert decoded["lookup_key"] == encoded["lookup_key"], case["name"]
