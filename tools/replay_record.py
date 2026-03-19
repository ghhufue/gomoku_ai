from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.cpp_backend import BACKEND_AVAILABLE as CPP_BACKEND_AVAILABLE, decode_reward_events, list_state_values
from gomoku_ai.env import BLACK, BOARD_SIZE, EMPTY, WHITE, evaluate_reward
from utils.board_printer import print_board


RECORDS_DIR = Path(__file__).resolve().parent / "game_records"
COLOR_TO_PLAYER = {"black": BLACK, "white": WHITE}


@dataclass(frozen=True)
class ReplayStep:
    index: int
    player: int
    player_name: str
    row: int
    col: int
    source: str
    reward: float
    reward_components: dict[str, float]
    board_before: np.ndarray
    board_after: np.ndarray
    info: dict[str, object]


@dataclass(frozen=True)
class LoadedRecord:
    name: str
    description: str
    steps: tuple[ReplayStep, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class RushFourBoard:
    group_index: int
    symmetry: str
    board: np.ndarray
    black_counts: tuple[int, ...]
    white_counts: tuple[int, ...]


def list_records() -> list[Path]:
    return sorted(RECORDS_DIR.glob("*.json"))


def normalize_single_game_payload(payload: dict[str, object], path: Path, game_index: int = 1) -> dict[str, object]:
    if "games" in payload:
        games = payload.get("games")
        if not isinstance(games, list) or not games:
            raise ValueError(f"record has no games: {path}")
        selected_index = game_index - 1
        if selected_index < 0 or selected_index >= len(games):
            raise ValueError(f"game index out of range: {game_index}")
        game = dict(games[selected_index])
        game.setdefault("name", f"{path.stem}_game{game_index:02d}")
        game.setdefault("description", "game extracted from multi-game match record")
        game["_record_metadata"] = {
            "schema_version": payload.get("schema_version"),
            "record_type": payload.get("record_type"),
            "checkpoint": payload.get("checkpoint"),
            "device": payload.get("device"),
            "selected_game_index": game_index,
            "game_count": len(games),
        }
        return game

    if "moves" in payload:
        normalized = dict(payload)
        normalized.setdefault("name", path.stem)
        normalized.setdefault("description", "")
        normalized["_record_metadata"] = {
            "schema_version": payload.get("schema_version", 1),
            "record_type": payload.get("record_type", "gomoku_record"),
            "selected_game_index": 1,
            "game_count": 1,
        }
        return normalized

    raise ValueError(f"unsupported record format: {path}")


def normalize_move(move: dict[str, object], fallback_index: int) -> dict[str, object]:
    row = int(move["row"])
    col = int(move["col"])
    player_name = str(move.get("color", move.get("player", ""))).lower()
    if player_name not in COLOR_TO_PLAYER:
        raise ValueError(f"unsupported player/color value: {player_name}")
    action = int(move.get("action", row * BOARD_SIZE + col))
    return {
        "move_number": int(move.get("move_number", fallback_index)),
        "player": player_name,
        "color": player_name,
        "source": str(move.get("source", "record")),
        "action": action,
        "row": row,
        "col": col,
    }


def load_record(path: Path, game_index: int = 1) -> LoadedRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    game_payload = normalize_single_game_payload(payload, path, game_index=game_index)
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    steps: list[ReplayStep] = []

    for index, raw_move in enumerate(game_payload["moves"], start=1):
        move = normalize_move(raw_move, fallback_index=index)
        row = int(move["row"])
        col = int(move["col"])
        player_name = str(move["color"]).lower()
        player = COLOR_TO_PLAYER[player_name]
        board_before = board.copy()
        board[row, col] = player
        board_after = board.copy()
        reward, info = evaluate_reward(board_before, row, col, player)
        reward_components = {
            "reward": float(info.get("reward", reward)),
            "offense_score": float(info.get("offense_score", 0.0)),
            "defense_score": float(info.get("defense_score", 0.0)),
        }
        steps.append(
            ReplayStep(
                index=index,
                player=player,
                player_name=player_name,
                row=row,
                col=col,
                source=str(move["source"]),
                reward=float(reward),
                reward_components=reward_components,
                board_before=board_before,
                board_after=board_after,
                info=info,
            )
        )

    return LoadedRecord(
        name=str(game_payload.get("name", path.stem)),
        description=str(game_payload.get("description", "")),
        steps=tuple(steps),
        metadata=dict(game_payload.get("_record_metadata", {})),
    )


def load_rush_four_group(path: Path) -> tuple[RushFourBoard, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"rush four group must be a non-empty list: {path}")

    items: list[RushFourBoard] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"rush four item #{index} must be an object")
        board_payload = item.get("board")
        if not isinstance(board_payload, list):
            raise ValueError(f"rush four item #{index} is missing board")
        board = np.asarray(board_payload, dtype=np.int8)
        if board.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"rush four item #{index} board must have shape ({BOARD_SIZE}, {BOARD_SIZE})")
        black_counts = tuple(int(value) for value in item.get("black_counts", []))
        white_counts = tuple(int(value) for value in item.get("white_counts", []))
        items.append(
            RushFourBoard(
                group_index=int(item.get("group_index", 0)),
                symmetry=str(item.get("symmetry", f"item_{index}")),
                board=board,
                black_counts=black_counts,
                white_counts=white_counts,
            )
        )
    return tuple(items)


def tracked_state_names() -> tuple[str, ...]:
    if not CPP_BACKEND_AVAILABLE:
        return ()
    values = list_state_values()
    names = [str(item["name"]) for item in values]
    names.append("five")
    return tuple(names)


TRACKED_STATE_NAMES = tracked_state_names()


def format_state_counts(counts: tuple[int, ...]) -> str:
    if not counts:
        return "-"
    parts: list[str] = []
    for index, value in enumerate(counts):
        if value == 0:
            continue
        if index < len(TRACKED_STATE_NAMES):
            parts.append(f"{TRACKED_STATE_NAMES[index]}={value}")
        else:
            parts.append(f"state_{index}={value}")
    return ", ".join(parts) if parts else "-"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def render_step(record: LoadedRecord, cursor: int) -> None:
    step = record.steps[cursor]
    clear_screen()
    print(f"[record] {record.name}")
    if record.description:
        print(f"[desc] {record.description}")
    if record.metadata:
        checkpoint = record.metadata.get("checkpoint")
        if checkpoint:
            print(f"[checkpoint] {checkpoint}")
        device = record.metadata.get("device")
        if device:
            print(f"[device] {device}")
        selected_game_index = record.metadata.get("selected_game_index")
        game_count = record.metadata.get("game_count")
        if selected_game_index and game_count:
            print(f"[game] {selected_game_index}/{game_count}")
    print(f"[step] {step.index}/{len(record.steps)}")
    print(f"[move] {step.player_name} -> ({step.row}, {step.col}) source={step.source}")
    print(f"[reward] total={step.reward:.2f}")
    print("[board_before]")
    print_board(step.board_before)
    print("[board_after]")
    print_board(step.board_after, last_move=(step.row, step.col))
    print("[reward_components]")
    for key, value in step.reward_components.items():
        print(f"  - {key}: {value:.2f}")
    print("[special_rewards]")
    for key, value in step.info.get("special_rewards", {}).items():
        print(f"  - {key}: {float(value):.2f}")
    print("[events]")
    raw_events = step.info.get("events", [])
    if CPP_BACKEND_AVAILABLE:
        try:
            decoded_events = decode_reward_events(raw_events)
        except Exception:
            decoded_events = []
    else:
        decoded_events = []
    if decoded_events:
        for event in decoded_events:
            print(
                "  - "
                f"{event['side']} {event['state_name']} x{int(event['count'])} "
                f"(event_id={int(event['event_id'])})"
            )
    else:
        for event in raw_events:
            print(f"  - {event}")
    print()
    print("←/→ 切换步数，A/D 也可，Q 退出。")


def read_key() -> str:
    if os.name == "nt":
        import msvcrt

        first = msvcrt.getwch()
        if first in ("\x00", "\xe0"):
            second = msvcrt.getwch()
            return {"K": "left", "M": "right"}.get(second, "")
        if first.lower() == "a":
            return "left"
        if first.lower() == "d":
            return "right"
        if first.lower() == "q":
            return "quit"
        return ""

    try:
        import termios
        import tty
    except ImportError:
        value = input("输入 left/right/q: ").strip().lower()
        return {"left": "left", "right": "right", "q": "quit"}.get(value, "")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first == "\x1b":
            second = sys.stdin.read(1)
            third = sys.stdin.read(1)
            if second == "[" and third == "D":
                return "left"
            if second == "[" and third == "C":
                return "right"
        if first.lower() == "a":
            return "left"
        if first.lower() == "d":
            return "right"
        if first.lower() == "q":
            return "quit"
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def replay_record(record: LoadedRecord) -> int:
    if not record.steps:
        print("record has no moves")
        return 1

    cursor = 0
    while True:
        render_step(record, cursor)
        key = read_key()
        if key == "quit":
            return 0
        if key == "left":
            cursor = max(0, cursor - 1)
        elif key == "right":
            cursor = min(len(record.steps) - 1, cursor + 1)


def first_item_per_group(items: tuple[RushFourBoard, ...]) -> tuple[RushFourBoard, ...]:
    selected: list[RushFourBoard] = []
    seen_groups: set[int] = set()
    for item in items:
        if item.group_index in seen_groups:
            continue
        selected.append(item)
        seen_groups.add(item.group_index)
    return tuple(selected)


def render_rush_four_board(items: tuple[RushFourBoard, ...], cursor: int) -> None:
    clear_screen()
    current = items[cursor]
    print(f"[group] {cursor + 1}/{len(items)} group_index={current.group_index} symmetry={current.symmetry}")
    print_board(current.board)
    print(f"[black] {format_state_counts(current.black_counts)}")
    print(f"[white] {format_state_counts(current.white_counts)}")
    print()
    print("Left/Right or A/D to switch, Q to quit")


def replay_rush_four_group(items: tuple[RushFourBoard, ...]) -> int:
    if not items:
        print("rush four group is empty")
        return 1

    groups = first_item_per_group(items)
    if not groups:
        print("rush four group has no displayable groups")
        return 1

    cursor = 0
    while True:
        render_rush_four_board(groups, cursor)
        key = read_key()
        if key == "quit":
            return 0
        if key == "left":
            cursor = max(0, cursor - 1)
        elif key == "right":
            cursor = min(len(groups) - 1, cursor + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a record and inspect per-move reward.")
    parser.add_argument("--record", type=Path, default=None, help="Path to record JSON under tests/game_records or any JSON file.")
    parser.add_argument("--name", type=str, default=None, help="Record name under tests/game_records without .json.")
    parser.add_argument("--game", type=int, default=1, help="Game index to replay when the record contains `games`.")
    parser.add_argument(
        "--rush-four-group",
        type=Path,
        default=None,
        help="Path to a generated rush-four group JSON, such as outputs/build/rush_four_group.json.",
    )
    parser.add_argument("--list", action="store_true", help="List available built-in records.")
    return parser


def resolve_record(args: argparse.Namespace) -> Path | None:
    if args.record is not None:
        return args.record.resolve()
    if args.name:
        return (RECORDS_DIR / f"{args.name}.json").resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for path in list_records():
            print(path.stem)
        return 0

    if args.rush_four_group is not None:
        group_path = args.rush_four_group.resolve()
        if not group_path.exists():
            print(f"rush four group not found: {group_path}")
            return 1
        try:
            items = load_rush_four_group(group_path)
        except ValueError as exc:
            print(str(exc))
            return 1
        return replay_rush_four_group(items)

    record_path = resolve_record(args)
    if record_path is None:
        parser.print_help()
        return 1
    if not record_path.exists():
        print(f"record not found: {record_path}")
        return 1

    try:
        record = load_record(record_path, game_index=args.game)
    except ValueError as exc:
        print(str(exc))
        return 1
    return replay_record(record)


if __name__ == "__main__":
    raise SystemExit(main())
