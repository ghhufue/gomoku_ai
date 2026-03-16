from __future__ import annotations

from tools.cli import make_parser


def test_cli_evaluate_defaults() -> None:
    parser = make_parser()
    args = parser.parse_args(["evaluate"])

    assert args.command == "evaluate"
    assert args.bot == "rule"
    assert str(args.output_dir).endswith("outputs\\evaluation") or str(args.output_dir).endswith("outputs/evaluation")
    assert args.export_record is True
    assert args.export_text is True
    assert args.export_visual is True


def test_cli_evaluate_can_disable_visual_export() -> None:
    parser = make_parser()
    args = parser.parse_args(["evaluate", "--bot", "random", "--no-export-visual"])

    assert args.bot == "random"
    assert args.export_visual is False
