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

from gomoku_ai.env import BLACK, BOARD_SIZE, EMPTY, WHITE, RewardConfig, evaluate_shape_reward
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


def list_records() -> list[Path]:
    return sorted(RECORDS_DIR.glob("*.json"))


def load_record(path: Path, reward_config: RewardConfig | None = None) -> LoadedRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    steps: list[ReplayStep] = []
    cfg = reward_config or RewardConfig()

    for index, move in enumerate(payload["moves"], start=1):
        row = int(move["row"])
        col = int(move["col"])
        player_name = str(move["color"]).lower()
        player = COLOR_TO_PLAYER[player_name]
        board_before = board.copy()
        board[row, col] = player
        board_after = board.copy()
        reward, info = evaluate_shape_reward(board_before, board_after, row, col, player, reward_config=cfg)
        reward_components = {key: float(value) for key, value in info.get("reward_components", {}).items()}
        steps.append(
            ReplayStep(
                index=index,
                player=player,
                player_name=player_name,
                row=row,
                col=col,
                source=str(move.get("source", "record")),
                reward=float(reward),
                reward_components=reward_components,
                board_before=board_before,
                board_after=board_after,
                info=info,
            )
        )

    return LoadedRecord(
        name=str(payload.get("name", path.stem)),
        description=str(payload.get("description", "")),
        steps=tuple(steps),
    )


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def render_step(record: LoadedRecord, cursor: int) -> None:
    step = record.steps[cursor]
    clear_screen()
    print(f"[record] {record.name}")
    if record.description:
        print(f"[desc] {record.description}")
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
    print("[self_pattern]")
    print(f"  - {step.info.get('self_pattern')}")
    print("[opp_threats_before]")
    print(f"  - {step.info.get('opp_threats_before')}")
    print("[opp_threats_after]")
    print(f"  - {step.info.get('opp_threats_after')}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a record and inspect per-move reward.")
    parser.add_argument("--record", type=Path, default=None, help="Path to record JSON under tests/game_records or any JSON file.")
    parser.add_argument("--name", type=str, default=None, help="Record name under tests/game_records without .json.")
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

    record_path = resolve_record(args)
    if record_path is None:
        parser.print_help()
        return 1
    if not record_path.exists():
        print(f"record not found: {record_path}")
        return 1

    record = load_record(record_path)
    return replay_record(record)


if __name__ == "__main__":
    raise SystemExit(main())
