from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.cpp_backend import write_rush_four_opening_group


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build rush-four dataset JSON via the C++ backend.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/build/precompute/rush_four_group.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="How many base groups to generate. Each group expands to 8 symmetry variants.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Use 0 to let the backend choose a random seed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.groups <= 0:
        parser.error("--groups must be positive")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rush_four_opening_group(str(output_path), int(args.groups), int(args.seed))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
