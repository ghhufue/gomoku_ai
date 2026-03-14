from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_registry(index_path: Path) -> dict:
    if not index_path.exists():
        return {"runs": [], "latest_run": None, "best_run": None}
    return json.loads(index_path.read_text(encoding="utf-8"))


def write_registry(index_path: Path, registry: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def recompute_registry_fields(registry: dict) -> dict:
    runs = registry.get("runs", [])
    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    registry["runs"] = runs
    registry["latest_run"] = runs[0]["run_dir"] if runs else None

    completed_runs = [run for run in runs if run.get("status") == "completed"]
    if completed_runs:
        best_run = max(
            completed_runs,
            key=lambda item: (
                float(item.get("best_win_rate")) if item.get("best_win_rate") is not None else -1.0,
                -(float(item.get("best_avg_steps")) if item.get("best_avg_steps") is not None else float("inf")),
            ),
        )
        registry["best_run"] = best_run["run_dir"]
    else:
        registry["best_run"] = None
    return registry


def cleanup_latest_run(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    index_path = run_root / "index.json"
    registry = load_registry(index_path)
    latest_run = registry.get("latest_run")

    if not latest_run:
        print(f"no latest run found under {run_root}")
        return 1

    latest_path = Path(latest_run)
    if not latest_path.exists():
        registry["runs"] = [run for run in registry.get("runs", []) if run.get("run_dir") != latest_run]
        write_registry(index_path, recompute_registry_fields(registry))
        print(f"latest run path was missing, cleaned stale index entry: {latest_run}")
        return 0

    if not args.yes:
        print("refusing to delete without --yes")
        print(f"latest run: {latest_path}")
        return 2

    shutil.rmtree(latest_path)
    registry["runs"] = [run for run in registry.get("runs", []) if run.get("run_dir") != latest_run]
    write_registry(index_path, recompute_registry_fields(registry))
    print(f"deleted latest run: {latest_path}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run management tools for Gomoku AI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    help_parser = subparsers.add_parser("help", help="Show help for the tool or a subcommand.")
    help_parser.add_argument("topic", nargs="?", default=None, help="Optional subcommand name.")

    cleanup_parser = subparsers.add_parser(
        "cleanup-latest-run",
        help="Delete the most recent run directory under runs/ and update runs/index.json.",
    )
    cleanup_parser.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
    cleanup_parser.add_argument("--yes", action="store_true", help="Actually perform the deletion.")

    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    if args.command in (None, "help"):
        if getattr(args, "topic", None):
            topic_parser = make_parser()
            topic_parser.parse_args([args.topic, "--help"])
            return 0
        parser.print_help()
        return 0

    if args.command == "cleanup-latest-run":
        return cleanup_latest_run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
