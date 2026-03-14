from __future__ import annotations

import json
from pathlib import Path


def load_registry(index_path: Path) -> dict:
    if not index_path.exists():
        return {"runs": [], "latest_run": None, "best_run": None}
    return json.loads(index_path.read_text(encoding="utf-8"))


def write_registry(index_path: Path, payload: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_manifest(manifest: dict) -> dict:
    latest_eval = manifest.get("latest_eval") or {}
    best_eval = manifest.get("best_eval") or {}
    paths = manifest.get("paths", {})
    return {
        "run_dir": manifest.get("run_dir"),
        "created_at": manifest.get("created_at"),
        "status": manifest.get("status"),
        "device": manifest.get("device"),
        "seed": manifest.get("seed"),
        "last_update": manifest.get("last_update"),
        "latest_checkpoint": manifest.get("latest_checkpoint"),
        "final_model": paths.get("final_model"),
        "best_model": paths.get("best_model"),
        "latest_win_rate": latest_eval.get("win_rate"),
        "latest_avg_steps": latest_eval.get("avg_steps"),
        "best_win_rate": best_eval.get("win_rate"),
        "best_avg_steps": best_eval.get("avg_steps"),
    }


def registry_sort_key(entry: dict) -> tuple[float, float]:
    best_win_rate = entry.get("best_win_rate")
    best_avg_steps = entry.get("best_avg_steps")
    return (
        float(best_win_rate) if best_win_rate is not None else -1.0,
        -(float(best_avg_steps) if best_avg_steps is not None else float("inf")),
    )


def update_registry(index_path: Path, manifest: dict) -> dict:
    registry = load_registry(index_path)
    entry = summarize_manifest(manifest)
    runs = [run for run in registry.get("runs", []) if run.get("run_dir") != entry["run_dir"]]
    runs.append(entry)
    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    registry["runs"] = runs
    registry["latest_run"] = runs[0]["run_dir"] if runs else None

    completed_runs = [run for run in runs if run.get("status") == "completed"]
    if completed_runs:
        best_entry = max(completed_runs, key=registry_sort_key)
        registry["best_run"] = best_entry["run_dir"]
    else:
        registry["best_run"] = None

    write_registry(index_path, registry)
    return registry
