from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import BLACK, BOARD_SIZE, RewardConfig, WHITE, evaluate_shape_reward
from scripts.train import build_reward_config, load_reward_config
from utils.board_printer import print_board


@dataclass(frozen=True)
class ShapeCase:
    name: str
    description: str
    stones_before: tuple[tuple[int, int, int], ...]
    move: tuple[int, int]
    player: int


SHAPE_CASES: tuple[ShapeCase, ...] = (
    ShapeCase("live_two_contiguous", "活二：连续两子，系数应为 1.0。", ((7, 6, BLACK),), (7, 7), BLACK),
    ShapeCase("live_two_gap1", "活二：中间空 1 格，系数应为 0.5。", ((7, 5, BLACK),), (7, 7), BLACK),
    ShapeCase("live_two_gap2", "活二：中间空 2 格，系数应为 0.15。", ((7, 4, BLACK),), (7, 7), BLACK),
    ShapeCase("live_three_contiguous", "活三：连续三子，系数应为 1.0。", ((7, 5, BLACK), (7, 6, BLACK)), (7, 7), BLACK),
    ShapeCase("live_three_gap1", "活三：中间空 1 格，系数应为 0.7。", ((7, 5, BLACK), (7, 7, BLACK)), (7, 8), BLACK),
    ShapeCase("live_three_gap2", "活三：中间空 2 格，系数应为 0.3。", ((7, 5, BLACK), (7, 6, BLACK)), (7, 9), BLACK),
    ShapeCase("sleep_three_contiguous", "死三：连续三子，系数应为 1.0。", ((7, 4, WHITE), (7, 5, BLACK), (7, 6, BLACK)), (7, 7), BLACK),
    ShapeCase("sleep_three_gap1", "死三：中间空 1 格，系数应为 0.7。", ((7, 4, WHITE), (7, 5, BLACK), (7, 7, BLACK)), (7, 8), BLACK),
    ShapeCase("sleep_three_gap2", "死三：中间空 2 格，系数应为 0.3。", ((7, 4, WHITE), (7, 5, BLACK), (7, 6, BLACK)), (7, 9), BLACK),
    ShapeCase("live_four_contiguous", "活四：连续四子，系数应为 1.0。", ((7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)), (7, 7), BLACK),
    ShapeCase("live_four_gap1", "活四：中间空 1 格，系数应为 0.5。", ((7, 4, BLACK), (7, 5, BLACK), (7, 7, BLACK)), (7, 8), BLACK),
    ShapeCase("rush_four_contiguous", "死四：连续四子，系数应为 1.0。", ((7, 3, WHITE), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)), (7, 7), BLACK),
    ShapeCase("rush_four_gap1", "死四：中间空 1 格，系数应为 0.5。", ((7, 3, WHITE), (7, 4, BLACK), (7, 5, BLACK), (7, 7, BLACK)), (7, 8), BLACK),
)


def empty_board() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


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

    value = input("输入 left/right/q: ").strip().lower()
    return {"left": "left", "right": "right", "q": "quit"}.get(value, "")


def build_case(case: ShapeCase, reward_config: RewardConfig) -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    board_before = empty_board()
    for row, col, player in case.stones_before:
        board_before[row, col] = player
    board_after = board_before.copy()
    row, col = case.move
    board_after[row, col] = case.player
    reward, info = evaluate_shape_reward(board_before, board_after, row, col, case.player, reward_config=reward_config)
    return board_before, board_after, reward, info


def render_case(case: ShapeCase, index: int, total: int, reward_config: RewardConfig) -> None:
    board_before, board_after, reward, info = build_case(case, reward_config)
    row, col = case.move
    clear_screen()
    print(f"[shape] {index + 1}/{total} {case.name}")
    print(f"[desc] {case.description}")
    print(f"[move] row={row} col={col} player={'black' if case.player == BLACK else 'white'}")
    print(f"[reward] total={reward:.2f}")
    print("[board_before]")
    print_board(board_before)
    print("[board_after]")
    print_board(board_after, last_move=(row, col))
    print("[reward_components]")
    for key, value in info.get("reward_components", {}).items():
        print(f"  - {key}: {float(value):.2f}")
    print("[move_shape_details.family_totals]")
    for key, value in info.get("move_shape_details", {}).get("family_totals", {}).items():
        print(f"  - {key}: {float(value):.2f}")
    print("[move_shape_details.active_directions]")
    for key, value in info.get("move_shape_details", {}).get("active_directions", {}).items():
        print(f"  - {key}: {int(value)}")
    print()
    print("←/→ 切换棋形，A/D 也可，Q 退出。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse detailed shape reward cases.")
    parser.add_argument("--list", action="store_true", help="List built-in shape cases.")
    parser.add_argument("--reward", type=Path, default=Path("configs/reward.toml"), help="Reward TOML path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for idx, case in enumerate(SHAPE_CASES, start=1):
            print(f"{idx}. {case.name} - {case.description}")
        return 0

    reward_config = build_reward_config(load_reward_config(args.reward.resolve()))
    cursor = 0
    while True:
        render_case(SHAPE_CASES[cursor], cursor, len(SHAPE_CASES), reward_config)
        key = read_key()
        if key == "quit":
            return 0
        if key == "left":
            cursor = max(0, cursor - 1)
        elif key == "right":
            cursor = min(len(SHAPE_CASES) - 1, cursor + 1)


if __name__ == "__main__":
    raise SystemExit(main())
