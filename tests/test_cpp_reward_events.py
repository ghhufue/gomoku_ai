from __future__ import annotations

from gomoku_ai.cpp_backend import BACKEND_AVAILABLE, list_state_values


def test_cpp_state_value_registry_is_exposed() -> None:
    if not BACKEND_AVAILABLE:
        return

    events = list_state_values()

    assert events
    assert any(event["name"] == "live_two" for event in events)
    assert any(event["name"] == "live_two_gap1" for event in events)
    assert any(event["name"] == "live_three_gap2" for event in events)
    assert any(event["name"] == "sleep_four_gap1" for event in events)
    assert any(event["name"] == "dead_four_gap1" for event in events)
