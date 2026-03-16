from __future__ import annotations

from pathlib import Path

from tools.cli import cleanup_workspace_artifacts, make_parser


def test_cli_cleanup_artifacts_defaults() -> None:
    parser = make_parser()
    args = parser.parse_args(["cleanup-artifacts"])

    assert args.command == "cleanup-artifacts"
    assert args.outputs_root == Path("outputs")
    assert args.log_root == Path("logs")
    assert args.run_root == Path("runs")
    assert args.yes is False


def test_cleanup_artifacts_requires_yes(tmp_path: Path, capsys) -> None:
    outputs_root = tmp_path / "outputs"
    log_root = tmp_path / "logs"
    run_root = tmp_path / "runs"

    parser = make_parser()
    args = parser.parse_args(
        [
            "cleanup-artifacts",
            "--outputs-root",
            str(outputs_root),
            "--log-root",
            str(log_root),
            "--run-root",
            str(run_root),
        ]
    )

    code = cleanup_workspace_artifacts(args)
    captured = capsys.readouterr()

    assert code == 2
    assert "refusing to delete without --yes" in captured.out


def test_cleanup_artifacts_preserves_outputs_child_dirs_and_clears_runs_and_logs(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    evaluation_dir = outputs_root / "evaluation"
    visual_dir = outputs_root / "visual"
    nested_file = evaluation_dir / "session1" / "match.json"
    second_nested_file = visual_dir / "report.md"
    direct_output_file = outputs_root / "top_level.txt"
    log_root = tmp_path / "logs"
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo_run"
    log_file = log_root / "train.log"
    run_file = run_dir / "final_model.pt"

    nested_file.parent.mkdir(parents=True, exist_ok=True)
    second_nested_file.parent.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    nested_file.write_text("x", encoding="utf-8")
    second_nested_file.write_text("y", encoding="utf-8")
    direct_output_file.write_text("z", encoding="utf-8")
    log_file.write_text("log", encoding="utf-8")
    run_file.write_text("run", encoding="utf-8")

    parser = make_parser()
    args = parser.parse_args(
        [
            "cleanup-artifacts",
            "--outputs-root",
            str(outputs_root),
            "--log-root",
            str(log_root),
            "--run-root",
            str(run_root),
            "--yes",
        ]
    )

    code = cleanup_workspace_artifacts(args)

    assert code == 0
    assert outputs_root.is_dir()
    assert evaluation_dir.is_dir()
    assert visual_dir.is_dir()
    assert list(evaluation_dir.iterdir()) == []
    assert list(visual_dir.iterdir()) == []
    assert set(outputs_root.iterdir()) == {evaluation_dir, visual_dir}
    assert log_root.is_dir()
    assert list(log_root.iterdir()) == []
    assert run_root.is_dir()
    assert list(run_root.iterdir()) == []
