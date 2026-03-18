from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tomllib

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


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def clear_directory(path: Path, preserve_root: bool = True) -> int:
    if not path.exists():
        if preserve_root:
            path.mkdir(parents=True, exist_ok=True)
        return 0

    removed = 0
    for child in path.iterdir():
        remove_path(child)
        removed += 1
    return removed


def clear_outputs_preserving_child_dirs(outputs_root: Path) -> tuple[int, int]:
    outputs_root.mkdir(parents=True, exist_ok=True)
    cleared_directories = 0
    removed_entries = 0

    for child in outputs_root.iterdir():
        if child.is_dir() and not child.is_symlink():
            removed_entries += clear_directory(child, preserve_root=True)
            cleared_directories += 1
            continue
        remove_path(child)
        removed_entries += 1
    return cleared_directories, removed_entries


def cleanup_workspace_artifacts(args: argparse.Namespace) -> int:
    outputs_root = args.outputs_root.resolve()
    log_root = args.log_root.resolve()
    run_root = args.run_root.resolve()

    if not args.yes:
        print("refusing to delete without --yes")
        print(f"outputs root: {outputs_root}")
        print(f"log root: {log_root}")
        print(f"run root: {run_root}")
        return 2

    outputs_dirs, outputs_removed = clear_outputs_preserving_child_dirs(outputs_root)
    logs_removed = clear_directory(log_root, preserve_root=True)
    runs_removed = clear_directory(run_root, preserve_root=True)

    print(
        "cleanup complete: outputs_dirs_preserved={outputs_dirs} outputs_entries_removed={outputs_removed} "
        "log_entries_removed={logs_removed} run_entries_removed={runs_removed}".format(
            outputs_dirs=outputs_dirs,
            outputs_removed=outputs_removed,
            logs_removed=logs_removed,
            runs_removed=runs_removed,
        )
    )
    return 0


def run_test_command(args: argparse.Namespace) -> int:
    from tests.test_env import main as test_env_main

    if args.list_cases:
        return test_env_main(["--list-cases"])

    argv: list[str] = []
    for case_id in args.case_id:
        argv.extend(["--case-id", str(case_id)])
    for case_name in args.case:
        argv.extend(["--case", case_name])
    return test_env_main(argv)


def run_evaluate_command(args: argparse.Namespace) -> int:
    from scripts.evaluate import main as evaluate_main

    argv: list[str] = []
    if args.checkpoint is not None:
        argv.extend(["--checkpoint", str(args.checkpoint)])
    argv.extend(["--games", str(args.games)])
    argv.extend(["--device", args.device])
    argv.extend(["--seed", str(args.seed)])
    argv.extend(["--num-seeds", str(args.num_seeds)])
    argv.extend(["--bot", args.bot])
    if args.bot_difficulty is not None:
        argv.extend(["--bot-difficulty", args.bot_difficulty])
    argv.extend(["--output-dir", str(args.output_dir)])
    argv.extend(["--filename", args.filename])

    if args.quiet_games:
        argv.append("--quiet-games")

    if args.export_record:
        argv.append("--export-record")
    else:
        argv.append("--no-export-record")

    if args.export_text:
        argv.append("--export-text")
    else:
        argv.append("--no-export-text")

    if args.export_visual:
        argv.append("--export-visual")
    else:
        argv.append("--no-export-visual")

    evaluate_main(argv)
    return 0


def run_replay_command(args: argparse.Namespace) -> int:
    from tools.replay_record import main as replay_main

    argv: list[str] = []
    if args.record is not None:
        argv.extend(["--record", str(args.record)])
    if args.name is not None:
        argv.extend(["--name", args.name])
    argv.extend(["--game", str(args.game)])
    if args.list:
        argv.append("--list")
    return int(replay_main(argv))


REWARD_FIELD_LABELS = {
    "terminal_reward": "终局奖励：当前落子直接成五时给分。",
    "live_four_reward": "活四奖励：形成活四时给分。",
    "rush_four_reward": "冲四奖励：形成冲四时给分。",
    "critical_reward": "关键奖励：解除对手立即制胜点时给分。",
    "shape_reward": "棋形奖励：形成活三或削弱对手活三时给分。",
    "sleep_three_reward": "眠三奖励：形成眠三时给分。",
    "probe_reward": "试探奖励：形成活二时给分。",
    "step_penalty": "基础步惩罚：每步固定扣分，抑制无意义拖延。",
    "unresolved_winning_threat_penalty": "未解除对手立即制胜威胁时的重罚。",
    "unresolved_four_threat_penalty": "未缓解对手四威胁时的惩罚。",
    "unresolved_live_three_threat_penalty": "未缓解对手活三前驱威胁时的惩罚。",
    "summary_winning_actions_weight": "threat_summary 中立即制胜点的权重。",
    "summary_live_four_weight": "threat_summary 中活四的权重。",
    "summary_rush_four_weight": "threat_summary 中冲四的权重。",
    "summary_live_three_weight": "threat_summary 中活三的权重。",
    "summary_sleep_three_weight": "threat_summary 中眠三的权重。",
    "summary_live_two_weight": "threat_summary 中活二的权重。",
    "live_two_contiguous_scale": "活二中连续两子的缩放系数。",
    "live_two_gap1_scale": "活二中间隔 1 格时的缩放系数。",
    "live_two_gap2_scale": "活二中间隔 2 格时的缩放系数。",
    "live_three_contiguous_scale": "活三中连续三子的缩放系数。",
    "live_three_gap1_scale": "活三中间隔 1 格时的缩放系数。",
    "live_three_gap2_scale": "活三中间隔 2 格时的缩放系数。",
    "sleep_three_contiguous_scale": "死三中连续三子的缩放系数。",
    "sleep_three_gap1_scale": "死三中间隔 1 格时的缩放系数。",
    "sleep_three_gap2_scale": "死三中间隔 2 格时的缩放系数。",
    "live_four_contiguous_scale": "活四中连续四子的缩放系数。",
    "live_four_gap1_scale": "活四中间隔 1 格时的缩放系数。",
    "rush_four_contiguous_scale": "死四中连续四子的缩放系数。",
    "rush_four_gap1_scale": "死四中间隔 1 格时的缩放系数。",
    "offense_delta_scale": "我方威胁增量转成奖励时的缩放系数。",
    "offense_delta_limit": "我方进攻增量奖励的上限。",
    "defense_delta_scale": "对手威胁下降转成奖励时的缩放系数。",
    "defense_delta_limit": "防守增量奖励的上限。",
    "double_live_three_bonus": "形成双活三时追加的奖励。",
    "block_live_four_bonus": "成功削弱对手四威胁时追加的奖励。",
}

REWARD_SECTION_TITLES = {
    "base": "Base",
    "defense": "Defense",
    "offense": "Offense",
    "weights": "Weights",
}


def normalize_reward_sections(reward_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    reward_root = reward_payload.get("reward", reward_payload)
    sections: dict[str, dict[str, object]] = {}
    for section_name in ("base", "defense", "offense", "weights"):
        section_payload = reward_root.get(section_name, {})
        if isinstance(section_payload, dict):
            sections[section_name] = section_payload
    if sections:
        return sections
    return {"base": reward_payload}


def build_reward_table_markdown(reward_payload: dict[str, object], source_path: Path) -> str:
    grouped_payload = normalize_reward_sections(reward_payload)
    lines = [
        "# Reward 配置表",
        "",
        f"- 源文件：`{source_path}`",
        "",
    ]
    for section_name, section_payload in grouped_payload.items():
        lines.extend(
            [
                f"## `{section_name}` / {REWARD_SECTION_TITLES.get(section_name, section_name.title())}",
                "",
                "| 参数名 | 当前值 | 含义 |",
                "|---|---:|---|",
            ]
        )
        for key, value in section_payload.items():
            meaning = REWARD_FIELD_LABELS.get(key, "未补充说明。")
            lines.append(f"| `{key}` | `{value}` | {meaning} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_reward_table_command(args: argparse.Namespace) -> int:
    reward_path = args.reward.resolve()
    if not reward_path.exists():
        print(f"reward config not found: {reward_path}")
        return 1

    reward_payload = tomllib.loads(reward_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{reward_path.stem}_table.md"
    markdown = build_reward_table_markdown(reward_payload, reward_path)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"generated reward table: {output_path}")
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

    cleanup_artifacts_parser = subparsers.add_parser(
        "cleanup-artifacts",
        help="Clear outputs/, logs/, and runs/ artifacts while preserving the root directories.",
    )
    cleanup_artifacts_parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Outputs root directory.")
    cleanup_artifacts_parser.add_argument("--log-root", type=Path, default=Path("logs"), help="Logs root directory.")
    cleanup_artifacts_parser.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
    cleanup_artifacts_parser.add_argument("--yes", action="store_true", help="Actually perform the deletion.")

    test_parser = subparsers.add_parser(
        "test",
        help="Run env tests, or print a single reward case with board/reward breakdown.",
    )
    test_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Named debug case from tests/test_env.py. Repeatable.",
    )
    test_parser.add_argument(
        "--case-id",
        action="append",
        type=int,
        default=[],
        help="Numbered test case from tests/TEST_CASES.md. Repeatable.",
    )
    test_parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available debug cases for `--case`.",
    )

    reward_table_parser = subparsers.add_parser(
        "reward-table",
        help="Convert a reward TOML config into a Markdown table under outputs/visual.",
    )
    reward_table_parser.add_argument(
        "--reward",
        type=Path,
        default=Path("configs/reward.toml"),
        help="Reward TOML path.",
    )
    reward_table_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visual"),
        help="Output directory for generated Markdown.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint against a bot, with optional replay-compatible exports.",
    )
    evaluate_parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint path. Defaults to latest final_model.pt.")
    evaluate_parser.add_argument("--games", type=int, default=50, help="Games per seed.")
    evaluate_parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda.")
    evaluate_parser.add_argument("--seed", type=int, default=123, help="Base seed.")
    evaluate_parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds to aggregate.")
    evaluate_parser.add_argument("--bot", type=str, default="rule", help="Opponent bot name from configs/bots.toml.")
    evaluate_parser.add_argument("--bot-difficulty", type=str, default=None, help="Opponent difficulty from configs/bots.toml.")
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"), help="Export directory.")
    evaluate_parser.add_argument("--filename", type=str, default="match_record", help="Export filename stem.")
    evaluate_parser.add_argument("--quiet-games", action="store_true", help="Suppress per-game logs.")
    evaluate_parser.add_argument("--export-record", action="store_true", default=True, help="Export replay-compatible JSON.")
    evaluate_parser.add_argument("--no-export-record", action="store_false", dest="export_record", help="Disable replay-compatible JSON export.")
    evaluate_parser.add_argument("--export-text", action="store_true", default=True, help="Export text report.")
    evaluate_parser.add_argument("--no-export-text", action="store_false", dest="export_text", help="Disable text report export.")
    evaluate_parser.add_argument("--export-visual", action="store_true", default=True, help="Export visual JSON.")
    evaluate_parser.add_argument("--no-export-visual", action="store_false", dest="export_visual", help="Disable visual JSON export.")

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a record and inspect per-move reward.",
    )
    replay_parser.add_argument("--record", type=Path, default=None, help="Path to record JSON.")
    replay_parser.add_argument("--name", type=str, default=None, help="Built-in record name without .json.")
    replay_parser.add_argument("--game", type=int, default=1, help="Game index for multi-game records.")
    replay_parser.add_argument("--list", action="store_true", help="List available built-in records.")

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

    if args.command == "cleanup-artifacts":
        return cleanup_workspace_artifacts(args)

    if args.command == "test":
        return run_test_command(args)

    if args.command == "reward-table":
        return run_reward_table_command(args)

    if args.command == "evaluate":
        return run_evaluate_command(args)

    if args.command == "replay":
        return run_replay_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
