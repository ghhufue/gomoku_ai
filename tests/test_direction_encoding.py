from __future__ import annotations

from gomoku_ai.cpp_backend import (
    BACKEND_AVAILABLE,
    debug_classify_direction_side_states,
    debug_decode_direction_lookup_key,
    debug_encode_direction_side_states,
)


def test_direction_encoding_roundtrip() -> None:
    if not BACKEND_AVAILABLE:
        return

    encoded = debug_encode_direction_side_states([0, 0, 1, 1, 0, 1, 0, 2, 0, 0])
    decoded = debug_decode_direction_lookup_key(int(encoded["lookup_key"]))

    assert decoded["lookup_key"] == encoded["lookup_key"]
    assert decoded["effective_length"] == encoded["effective_length"]
    assert decoded["left_trimmed_empty"] == encoded["left_trimmed_empty"]
    assert decoded["right_trimmed_empty"] == encoded["right_trimmed_empty"]
    assert decoded["side_states"] == [0, 0, 1, 1, 0, 1, 0, 2, 0, 0]


def test_shorter_effective_sequence_gets_smaller_lookup_bucket() -> None:
    if not BACKEND_AVAILABLE:
        return

    short_key = debug_encode_direction_side_states([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])["lookup_key"]
    long_key = debug_encode_direction_side_states([0, 1, 1, 1, 0, 1, 2, 0, 0, 0])["lookup_key"]

    assert short_key < long_key


def test_direction_event_debug_api_is_exposed() -> None:
    if not BACKEND_AVAILABLE:
        return

    events = debug_classify_direction_side_states([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])

    assert "live_three" in events
