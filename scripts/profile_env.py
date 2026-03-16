from __future__ import annotations

import argparse
import cProfile
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pstats
import sys
from time import perf_counter
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gomoku_ai.env as env_mod
from bots import create_bot
from gomoku_ai.env import DEFAULT_REWARD_CONFIG, GomokuEnv, RewardConfig, VectorEnv
from gomoku_ai.model import ActorCriticNet, MODEL_PRESETS, model_preset_config
from gomoku_ai.ppo import PPOConfig, PPOTrainer


@dataclass
class CallMetric:
    calls: int = 0
    total_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.calls += 1
        self.total_s += elapsed_s


class ProfileCollector:
    def __init__(self):
        self.metrics: dict[str, CallMetric] = {}

    def record(self, name: str, elapsed_s: float) -> None:
        metric = self.metrics.setdefault(name, CallMetric())
        metric.add(elapsed_s)

    def snapshot(self) -> dict[str, dict[str, float]]:
        payload: dict[str, dict[str, float]] = {}
        for name, metric in sorted(self.metrics.items()):
            payload[name] = {
                "calls": float(metric.calls),
                "total_s": metric.total_s,
                "avg_ms": (metric.total_s / metric.calls * 1000.0) if metric.calls else 0.0,
            }
        return payload


class PatchManager:
    def __init__(self):
        self._restorers: list[tuple[object, str, object]] = []

    def patch_attr(self, owner: object, name: str, value: object) -> None:
        original = getattr(owner, name)
        self._restorers.append((owner, name, original))
        setattr(owner, name, value)

    def restore_all(self) -> None:
        while self._restorers:
            owner, name, original = self._restorers.pop()
            setattr(owner, name, original)


def wrap_callable(collector: ProfileCollector, name: str, fn):
    def wrapped(*args, **kwargs):
        start = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            collector.record(name, perf_counter() - start)

    return wrapped


def instrument_env_module(collector: ProfileCollector, patches: PatchManager) -> None:
    for name in (
        "evaluate_shape_reward",
        "threat_summary",
        "classify_move_counts",
        "classify_move_shape_details",
        "affected_actions",
        "apply_local_threat_delta",
        "summary_score",
        "move_pattern_reward",
    ):
        patches.patch_attr(env_mod, name, wrap_callable(collector, f"env.{name}", getattr(env_mod, name)))


def instrument_model(model: ActorCriticNet, collector: ProfileCollector, patches: PatchManager) -> None:
    for name in ("sample_action", "masked_distribution", "evaluate_actions", "forward"):
        original = getattr(model, name)
        patches.patch_attr(model, name, wrap_callable(collector, f"model.{name}", original))


def instrument_trainer(trainer: PPOTrainer, collector: ProfileCollector, patches: PatchManager) -> None:
    patches.patch_attr(trainer.env, "step", wrap_callable(collector, "vector_env.step", trainer.env.step))
    patches.patch_attr(trainer.optimizer, "step", wrap_callable(collector, "optimizer.step", trainer.optimizer.step))
    patches.patch_attr(
        trainer.optimizer,
        "zero_grad",
        wrap_callable(collector, "optimizer.zero_grad", trainer.optimizer.zero_grad),
    )


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def flatten_reward_payload(raw_reward: dict | None) -> dict | None:
    if raw_reward is None:
        return None
    if "reward" in raw_reward and isinstance(raw_reward["reward"], dict):
        raw_reward = raw_reward["reward"]
    flattened: dict[str, object] = {}
    for value in raw_reward.values():
        if isinstance(value, dict):
            flattened.update(value)
    return flattened or raw_reward


def build_reward_config(raw_reward: dict | None) -> RewardConfig:
    flattened = flatten_reward_payload(raw_reward)
    if flattened is None:
        return DEFAULT_REWARD_CONFIG
    defaults = DEFAULT_REWARD_CONFIG.__dict__
    payload = {key: flattened.get(key, defaults[key]) for key in defaults}
    return RewardConfig(**payload)


def load_train_config(config_path: Path) -> dict:
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def load_reward_config(config_path: Path | None) -> dict | None:
    if config_path is None or not config_path.exists():
        return None
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def build_envs(
    n_envs: int,
    seed: int,
    reward_config: RewardConfig,
    bot_name: str,
    bot_difficulty: str | None,
) -> VectorEnv:
    opponent = create_bot(name=bot_name, difficulty=bot_difficulty, reward_config=reward_config)
    return VectorEnv(
        GomokuEnv(
            opponent=opponent,
            seed=seed + idx,
            reward_config=reward_config,
        )
        for idx in range(n_envs)
    )


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def run_env_steps(steps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    env = GomokuEnv(opponent=create_bot(name="reward_driven_hard"), seed=seed)
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


def summarize_metrics(raw_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return raw_metrics


def profile_single_preset(
    preset: str,
    *,
    training_cfg: dict,
    opponent_cfg: dict,
    reward_config: RewardConfig,
    device: str,
    seed: int,
    verbose: bool = True,
) -> dict[str, object]:
    model_config = model_preset_config(preset)
    model = ActorCriticNet(model_config)
    env = build_envs(
        n_envs=int(training_cfg["n_envs"]),
        seed=seed,
        reward_config=reward_config,
        bot_name=str(opponent_cfg["bot_name"]),
        bot_difficulty=(str(opponent_cfg["bot_difficulty"]) or None),
    )
    trainer = PPOTrainer(
        model=model,
        env=env,
        config=PPOConfig(
            n_envs=int(training_cfg["n_envs"]),
            n_steps=int(training_cfg["n_steps"]),
            total_updates=int(training_cfg["updates"]),
            batch_size=int(training_cfg["batch_size"]),
            epochs=int(training_cfg["epochs"]),
            learning_rate=float(training_cfg["lr"]),
            device=device,
            show_progress=False,
        ),
        writer=None,
    )

    collector = ProfileCollector()
    patches = PatchManager()
    instrument_env_module(collector, patches)
    instrument_model(trainer.model, collector, patches)
    instrument_trainer(trainer, collector, patches)

    obs, masks = trainer.env.reset()
    rollout_stats_list: list[dict[str, float]] = []
    update_stats_list: list[dict[str, float]] = []

    try:
        preset_started_at = perf_counter()
        for update_idx in range(1, int(training_cfg["updates"]) + 1):
            rollout_start = perf_counter()
            rollout, obs, masks, rollout_stats = trainer.collect_rollout(obs, masks, update_idx)
            rollout_time_s = perf_counter() - rollout_start

            optimize_start = perf_counter()
            train_stats = trainer.update(rollout)
            optimize_time_s = perf_counter() - optimize_start
            total_update_time_s = rollout_time_s + optimize_time_s

            rollout_stats_list.append(
                {
                    "update": float(update_idx),
                    "rollout_time_s": rollout_time_s,
                    "episodes": float(rollout_stats["episodes"]),
                    "ep_rew_mean": float(rollout_stats["ep_rew_mean"]),
                    "wins": float(rollout_stats["wins"]),
                    "losses": float(rollout_stats["losses"]),
                    "draws": float(rollout_stats["draws"]),
                }
            )
            update_stats_list.append(
                {
                    "update": float(update_idx),
                    "optimize_time_s": optimize_time_s,
                    "policy_loss": float(train_stats["policy_loss"]),
                    "value_loss": float(train_stats["value_loss"]),
                    "entropy": float(train_stats["entropy"]),
                    "approx_kl": float(train_stats["approx_kl"]),
                }
            )
            if verbose:
                print(
                    "[preset={preset}] update {update}/{total} rollout={rollout:.3f}s optimize={optimize:.3f}s total={total_time:.3f}s".format(
                        preset=preset,
                        update=update_idx,
                        total=int(training_cfg["updates"]),
                        rollout=rollout_time_s,
                        optimize=optimize_time_s,
                        total_time=total_update_time_s,
                    )
                )
    finally:
        patches.restore_all()

    samples_per_update = int(training_cfg["n_envs"]) * int(training_cfg["n_steps"])
    total_rollout_time_s = sum(item["rollout_time_s"] for item in rollout_stats_list)
    total_optimize_time_s = sum(item["optimize_time_s"] for item in update_stats_list)
    total_time_s = total_rollout_time_s + total_optimize_time_s
    preset_elapsed_s = perf_counter() - preset_started_at

    if verbose:
        print(
            "[preset={preset}] done elapsed={elapsed:.3f}s avg_rollout={rollout:.3f}s avg_optimize={optimize:.3f}s samples_per_sec={fps:.2f}".format(
                preset=preset,
                elapsed=preset_elapsed_s,
                rollout=total_rollout_time_s / max(1, len(rollout_stats_list)),
                optimize=total_optimize_time_s / max(1, len(update_stats_list)),
                fps=(samples_per_update * int(training_cfg["updates"])) / max(total_time_s, 1e-6),
            )
        )

    return {
        "preset": preset,
        "model_config": model_config.to_dict(),
        "samples_per_update": float(samples_per_update),
        "updates": float(training_cfg["updates"]),
        "total_rollout_time_s": total_rollout_time_s,
        "total_optimize_time_s": total_optimize_time_s,
        "total_time_s": total_time_s,
        "avg_rollout_time_s": total_rollout_time_s / max(1, len(rollout_stats_list)),
        "avg_optimize_time_s": total_optimize_time_s / max(1, len(update_stats_list)),
        "samples_per_sec": (samples_per_update * int(training_cfg["updates"])) / max(total_time_s, 1e-6),
        "preset_elapsed_s": preset_elapsed_s,
        "rollout_includes_model_inference": True,
        "function_metrics": summarize_metrics(collector.snapshot()),
        "rollout_updates": rollout_stats_list,
        "optimize_updates": update_stats_list,
    }


def build_markdown_report(payload: dict[str, object]) -> str:
    lines = [
        "# Training Profile Report",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Device: `{payload['device']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Config: `{payload['config_path']}`",
        f"- Notes: rollout time includes model inference and environment stepping.",
        "",
    ]

    for result in payload["results"]:
        lines.extend(
            [
                f"## {result['preset']}",
                "",
                f"- Samples/update: `{int(result['samples_per_update'])}`",
                f"- Updates profiled: `{int(result['updates'])}`",
                f"- Avg rollout time: `{result['avg_rollout_time_s']:.4f}s`",
                f"- Avg optimize time: `{result['avg_optimize_time_s']:.4f}s`",
                f"- Total time: `{result['total_time_s']:.4f}s`",
                f"- Throughput: `{result['samples_per_sec']:.2f}` samples/s",
                "",
                "| Metric | Calls | Total(s) | Avg(ms) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, metric in result["function_metrics"].items():
            lines.append(
                f"| `{name}` | {int(metric['calls'])} | {metric['total_s']:.4f} | {metric['avg_ms']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def training_profile_mode(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    raw_config = load_train_config(config_path)
    runtime = raw_config["runtime"]
    training = raw_config["training"]
    opponent = raw_config.get("opponent", {})

    reward_config_path = Path(runtime.get("reward_config", "configs/reward.toml"))
    if not reward_config_path.is_absolute():
        reward_config_path = (ROOT / reward_config_path).resolve()
    reward_config = build_reward_config(load_reward_config(reward_config_path))

    training_cfg = {
        "n_envs": args.n_envs if args.n_envs is not None else training["n_envs"],
        "n_steps": args.n_steps if args.n_steps is not None else training["n_steps"],
        "updates": args.updates,
        "batch_size": args.batch_size if args.batch_size is not None else training["batch_size"],
        "epochs": args.epochs if args.epochs is not None else training["epochs"],
        "lr": args.lr if args.lr is not None else training["lr"],
    }
    opponent_cfg = {
        "bot_name": args.bot if args.bot is not None else opponent.get("bot_name", "reward_driven_hard"),
        "bot_difficulty": args.bot_difficulty if args.bot_difficulty is not None else opponent.get("bot_difficulty", ""),
    }
    device = resolve_device(args.device)
    presets = [item.strip() for item in args.presets.split(",") if item.strip()]
    unsupported = [item for item in presets if item not in MODEL_PRESETS]
    if unsupported:
        raise ValueError(f"unsupported presets: {unsupported}")

    results = []
    overall_start = perf_counter()
    for index, preset in enumerate(presets):
        print(
            "[{current}/{total}] profiling preset={preset} device={device} n_envs={n_envs} n_steps={n_steps} updates={updates}".format(
                current=index + 1,
                total=len(presets),
                preset=preset,
                device=device,
                n_envs=training_cfg["n_envs"],
                n_steps=training_cfg["n_steps"],
                updates=training_cfg["updates"],
            )
        )
        results.append(
            profile_single_preset(
                preset,
                training_cfg=training_cfg,
                opponent_cfg=opponent_cfg,
                reward_config=reward_config,
                device=device,
                seed=args.seed + index * 1000,
                verbose=True,
            )
        )
    overall_elapsed_s = perf_counter() - overall_start

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "seed": args.seed,
        "config_path": config_path,
        "training": training_cfg,
        "opponent": opponent_cfg,
        "overall_elapsed_s": overall_elapsed_s,
        "results": results,
    }

    json_path = output_dir / f"{timestamp}_training_profile.json"
    md_path = output_dir / f"{timestamp}_training_profile.md"
    latest_json_path = output_dir / "latest_training_profile.json"
    latest_md_path = output_dir / "latest_training_profile.md"
    json_text = json.dumps(to_jsonable(payload), indent=2)
    markdown = build_markdown_report(to_jsonable(payload))
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")
    latest_md_path.write_text(markdown, encoding="utf-8")

    print(f"saved: {json_path}")
    print(f"saved: {md_path}")
    print(f"saved: {latest_json_path}")
    print(f"saved: {latest_md_path}")
    print(f"overall elapsed: {overall_elapsed_s:.3f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Gomoku environment and PPO training performance.")
    parser.add_argument(
        "--mode",
        choices=("benchmark", "cprofile", "training-profile"),
        default="training-profile",
        help="benchmark: simple env throughput; cprofile: Python call report; training-profile: profile rollout/update by model preset.",
    )
    parser.add_argument("--steps", type=int, default=200, help="How many environment steps to execute for benchmark/cprofile.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--sort-by", type=str, default="cumulative", help="pstats sort key for cprofile mode.")
    parser.add_argument("--top-k", type=int, default=30, help="How many functions to print in cprofile mode.")
    parser.add_argument("--profile-out", type=Path, default=None, help="Optional .prof output path for cprofile mode.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.toml"), help="Training config path.")
    parser.add_argument("--device", type=str, default="auto", help="Device for training-profile mode.")
    parser.add_argument("--presets", type=str, default="small,base,large", help="Comma-separated model presets to profile.")
    parser.add_argument("--updates", type=int, default=1, help="How many PPO updates to profile per preset.")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of environments.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override rollout steps per environment.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override PPO batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override PPO epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--bot", type=str, default=None, help="Override opponent bot name.")
    parser.add_argument("--bot-difficulty", type=str, default=None, help="Override opponent difficulty.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/profile"), help="Profile report output directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "benchmark":
        return benchmark_mode(args)
    if args.mode == "cprofile":
        return cprofile_mode(args)
    return training_profile_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
