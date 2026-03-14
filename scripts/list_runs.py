from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List tracked training runs.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--limit", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index_path = args.run_root / "index.json"
    if not index_path.exists():
        print(f"run index not found: {index_path}")
        return

    registry = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"latest_run={registry.get('latest_run')}")
    print(f"best_run={registry.get('best_run')}")
    print("--- runs ---")

    for entry in registry.get("runs", [])[: args.limit]:
        print(
            "status={status} update={last_update} latest_win_rate={latest_win_rate} "
            "best_win_rate={best_win_rate} run_dir={run_dir}".format(**entry)
        )


if __name__ == "__main__":
    main()
