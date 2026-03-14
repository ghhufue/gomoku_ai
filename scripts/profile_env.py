from __future__ import annotations

import argparse
import cProfile
import io
from pathlib import Path
import pstats
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import GomokuEnv
from gomoku_ai.rule_bot import RuleBasedBot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Gomoku environment rollout cost.")
    parser.add_argument("--steps", type=int, default=200, help="How many environment steps to execute.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--mode",
        choices=("benchmark", "cprofile"),
        default="benchmark",
        help="benchmark: wall-clock timing summary; cprofile: Python call profiling report.",
    )
    parser.add_argument("--sort-by", type=str, default="cumulative", help="pstats sort key for cprofile mode.")
    parser.add_argument("--top-k", type=int, default=30, help="How many functions to print in cprofile mode.")
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Optional .prof output path for cprofile mode.",
    )
    return parser


def run_env_steps(steps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    env = GomokuEnv(opponent=RuleBasedBot(), seed=seed)
    obs, mask = env.reset()

    start = perf_counter()
    episodes = 0
    illegal_moves = 0

    for _ in range(steps):
        legal_actions = np.flatnonzero(mask)
        action = int(rng.choice(legal_actions))
        result = env.step(action)
        obs = result.observation
        mask = result.action_mask
        if result.done:
            episodes += 1
            if result.info.get("illegal_move"):
                illegal_moves += 1
            obs, mask = env.reset()

    elapsed = perf_counter() - start
    return {
        "steps": float(steps),
        "episodes": float(episodes),
        "illegal_moves": float(illegal_moves),
        "elapsed_s": elapsed,
        "steps_per_sec": steps / max(elapsed, 1e-6),
    }


def benchmark_mode(args: argparse.Namespace) -> int:
    metrics = run_env_steps(steps=args.steps, seed=args.seed)
    print("--- environment benchmark ---")
    print(f"steps={int(metrics['steps'])}")
    print(f"episodes={int(metrics['episodes'])}")
    print(f"illegal_moves={int(metrics['illegal_moves'])}")
    print(f"elapsed_s={metrics['elapsed_s']:.3f}")
    print(f"steps_per_sec={metrics['steps_per_sec']:.2f}")
    return 0


def cprofile_mode(args: argparse.Namespace) -> int:
    profiler = cProfile.Profile()
    profiler.enable()
    metrics = run_env_steps(steps=args.steps, seed=args.seed)
    profiler.disable()

    if args.profile_out is not None:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.profile_out))
        print(f"saved profile to {args.profile_out}")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(args.sort_by)
    stats.print_stats(args.top_k)

    print("--- environment cprofile summary ---")
    print(f"steps={int(metrics['steps'])}")
    print(f"elapsed_s={metrics['elapsed_s']:.3f}")
    print(f"steps_per_sec={metrics['steps_per_sec']:.2f}")
    print(stream.getvalue())
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "benchmark":
        return benchmark_mode(args)
    return cprofile_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
