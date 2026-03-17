from __future__ import annotations

from gomoku_ai.cpp_backend import list_state_values


def test_state_values_are_exposed() -> None:
    values = list_state_values()

    assert values
    assert any(item["name"] == "live_one" for item in values)
    assert any(item["name"] == "sleep_four" for item in values)
