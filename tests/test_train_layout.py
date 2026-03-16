from __future__ import annotations

from pathlib import Path

from scripts.train import infer_run_dir_from_resume_path, parse_bool_flag, resolve_run_layout


def test_infer_run_dir_from_checkpoint_path() -> None:
    artifacts = {
        "checkpoint_dir": "checkpoints",
        "final_model_name": "final_model.pt",
    }
    resume_path = Path("runs/demo_run/checkpoints/checkpoint_update_0200.pt")

    run_dir = infer_run_dir_from_resume_path(resume_path, artifacts)

    assert run_dir == Path("runs/demo_run").resolve()


def test_infer_run_dir_from_final_model_path() -> None:
    artifacts = {
        "checkpoint_dir": "checkpoints",
        "final_model_name": "final_model.pt",
    }
    resume_path = Path("runs/demo_run/final_model.pt")

    run_dir = infer_run_dir_from_resume_path(resume_path, artifacts)

    assert run_dir == Path("runs/demo_run").resolve()


def test_resolve_run_layout_reuses_resume_run_dir_by_default() -> None:
    config = {
        "runtime": {
            "resume_from": Path("runs/demo_run/checkpoints/checkpoint_update_0200.pt").resolve(),
            "run_name": None,
            "run_root": Path("runs").resolve(),
            "start_new_branch": False,
        },
        "artifacts": {
            "checkpoint_dir": "checkpoints",
            "tensorboard_dir": "tensorboard",
            "final_model_name": "final_model.pt",
            "manifest_name": "manifest.json",
            "latest_eval_name": "latest_eval.json",
        },
    }

    layout = resolve_run_layout(config)

    assert layout["run_dir"] == Path("runs/demo_run").resolve()
    assert layout["checkpoint_dir"] == Path("runs/demo_run/checkpoints").resolve()
    assert layout["history_dir"] == Path("runs/demo_run/history").resolve()


def test_resolve_run_layout_starts_new_branch_when_enabled() -> None:
    config = {
        "runtime": {
            "resume_from": Path("runs/demo_run/checkpoints/checkpoint_update_0200.pt").resolve(),
            "run_name": "branch_run",
            "run_root": Path("runs").resolve(),
            "start_new_branch": True,
        },
        "artifacts": {
            "checkpoint_dir": "checkpoints",
            "tensorboard_dir": "tensorboard",
            "final_model_name": "final_model.pt",
            "manifest_name": "manifest.json",
            "latest_eval_name": "latest_eval.json",
        },
    }

    layout = resolve_run_layout(config)

    assert layout["run_dir"] == Path("runs/branch_run").resolve()
    assert layout["checkpoint_dir"] == Path("runs/branch_run/checkpoints").resolve()
    assert layout["history_dir"] == Path("runs/branch_run/history").resolve()


def test_parse_bool_flag_supports_common_values() -> None:
    assert parse_bool_flag(None, default=False) is False
    assert parse_bool_flag(None, default=True) is True
    assert parse_bool_flag("true") is True
    assert parse_bool_flag("false") is False
    assert parse_bool_flag("yes") is True
    assert parse_bool_flag("0") is False
