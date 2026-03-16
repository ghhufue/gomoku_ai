from __future__ import annotations

from pathlib import Path

from tools.cli import make_parser


def test_cli_replay_defaults() -> None:
    parser = make_parser()
    args = parser.parse_args(["replay"])

    assert args.command == "replay"
    assert args.record is None
    assert args.name is None
    assert args.game == 1
    assert args.list is False


def test_cli_replay_accepts_record_and_game() -> None:
    parser = make_parser()
    args = parser.parse_args(["replay", "--record", "outputs/evaluation/demo/match_record.json", "--game", "3"])

    assert args.command == "replay"
    assert args.record == Path("outputs/evaluation/demo/match_record.json")
    assert args.game == 3


def test_cli_replay_accepts_builtin_record_name() -> None:
    parser = make_parser()
    args = parser.parse_args(["replay", "--name", "block_live_three"])

    assert args.command == "replay"
    assert args.name == "block_live_three"
