from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect tensor statistics inside a training checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path containing model_state_dict.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many parameter tensors to show in the detailed table.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="name",
        choices=["name", "numel", "abs_max", "std"],
        help="Sort key for the detailed table.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to outputs/inspection/<checkpoint_stem>_stats.json.",
    )
    return parser


def format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "()"
    return "x".join(str(dim) for dim in shape)


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, object]:
    data = tensor.detach().to(dtype=torch.float32, device="cpu")
    flattened = data.reshape(-1)
    return {
        "name": name,
        "shape": list(data.shape),
        "dtype": str(tensor.dtype),
        "numel": int(data.numel()),
        "min": float(flattened.min().item()),
        "max": float(flattened.max().item()),
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "abs_max": float(flattened.abs().max().item()),
        "l2_norm": float(torch.linalg.vector_norm(flattened).item()),
        "zero_fraction": float((flattened == 0).float().mean().item()),
    }


def load_model_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint does not contain model_state_dict: {checkpoint_path}")
    return model_state


def build_summary(stats: list[dict[str, object]]) -> dict[str, object]:
    total_params = sum(int(item["numel"]) for item in stats)
    global_min = min(float(item["min"]) for item in stats)
    global_max = max(float(item["max"]) for item in stats)
    largest_abs = max(stats, key=lambda item: float(item["abs_max"]))
    largest_std = max(stats, key=lambda item: float(item["std"]))
    largest_tensor = max(stats, key=lambda item: int(item["numel"]))
    weighted_mean = sum(float(item["mean"]) * int(item["numel"]) for item in stats) / max(1, total_params)
    weighted_zero_fraction = (
        sum(float(item["zero_fraction"]) * int(item["numel"]) for item in stats) / max(1, total_params)
    )
    return {
        "tensor_count": len(stats),
        "total_params": total_params,
        "global_min": global_min,
        "global_max": global_max,
        "weighted_mean": weighted_mean,
        "weighted_zero_fraction": weighted_zero_fraction,
        "largest_abs_max_tensor": largest_abs["name"],
        "largest_abs_max": largest_abs["abs_max"],
        "largest_std_tensor": largest_std["name"],
        "largest_std": largest_std["std"],
        "largest_tensor_by_numel": largest_tensor["name"],
        "largest_tensor_numel": largest_tensor["numel"],
    }


def sort_stats(stats: list[dict[str, object]], sort_by: str, descending: bool) -> list[dict[str, object]]:
    key_map = {
        "name": lambda item: str(item["name"]),
        "numel": lambda item: int(item["numel"]),
        "abs_max": lambda item: float(item["abs_max"]),
        "std": lambda item: float(item["std"]),
    }
    return sorted(stats, key=key_map[sort_by], reverse=descending)


def print_summary(checkpoint_path: Path, summary: dict[str, object]) -> None:
    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(f"tensor_count: {summary['tensor_count']}")
    print(f"total_params: {summary['total_params']}")
    print(f"global_min: {summary['global_min']:.6f}")
    print(f"global_max: {summary['global_max']:.6f}")
    print(f"weighted_mean: {summary['weighted_mean']:.6f}")
    print(f"weighted_zero_fraction: {summary['weighted_zero_fraction']:.6f}")
    print(
        "largest_abs_max: {value:.6f} ({name})".format(
            value=float(summary["largest_abs_max"]),
            name=summary["largest_abs_max_tensor"],
        )
    )
    print(
        "largest_std: {value:.6f} ({name})".format(
            value=float(summary["largest_std"]),
            name=summary["largest_std_tensor"],
        )
    )
    print(
        "largest_tensor_by_numel: {name} ({numel})".format(
            name=summary["largest_tensor_by_numel"],
            numel=int(summary["largest_tensor_numel"]),
        )
    )


def print_table(stats: list[dict[str, object]], limit: int) -> None:
    print("")
    print("name | shape | numel | min | max | mean | std | abs_max | zero_fraction")
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for item in stats[:limit]:
        print(
            "{name} | {shape} | {numel} | {min:.6f} | {max:.6f} | {mean:.6f} | {std:.6f} | {abs_max:.6f} | {zero_fraction:.6f}".format(
                name=item["name"],
                shape=format_shape(tuple(int(dim) for dim in item["shape"])),
                numel=int(item["numel"]),
                min=float(item["min"]),
                max=float(item["max"]),
                mean=float(item["mean"]),
                std=float(item["std"]),
                abs_max=float(item["abs_max"]),
                zero_fraction=float(item["zero_fraction"]),
            )
        )


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        print(f"checkpoint not found: {checkpoint_path}")
        return 1

    model_state = load_model_state(checkpoint_path)
    stats = [tensor_stats(name, tensor) for name, tensor in model_state.items()]
    summary = build_summary(stats)
    ordered_stats = sort_stats(stats, sort_by=args.sort_by, descending=args.descending)

    print_summary(checkpoint_path, summary)
    print_table(ordered_stats, max(0, int(args.top)))

    payload = {
        "checkpoint": str(checkpoint_path),
        "summary": summary,
        "tensors": ordered_stats,
    }
    output_path = (
        args.json.resolve()
        if args.json is not None
        else (ROOT / "outputs" / "inspection" / f"{checkpoint_path.stem}_stats.json").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("")
    print(f"saved json: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
