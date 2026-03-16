from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
import tomllib

import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import DEFAULT_REWARD_CONFIG, GomokuEnv, RewardConfig, VectorEnv
from gomoku_ai.model import (
    MODEL_PRESETS,
    ActorCriticNet,
    ModelConfig,
    model_config_from_checkpoint_payload,
    model_preset_config,
)
from gomoku_ai.ppo import PPOConfig, PPOTrainer
from gomoku_ai.run_registry import update_registry
from bots import create_bot
from scripts.evaluate import evaluate_across_seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the initial Gomoku PPO prototype.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--start-new-branch", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model-preset", type=str, choices=MODEL_PRESETS, default=None)
    parser.add_argument("--model-channels", type=int, default=None)
    parser.add_argument("--model-blocks", type=int, default=None)
    parser.add_argument("--policy-channels", type=int, default=None)
    parser.add_argument("--value-channels", type=int, default=None)
    parser.add_argument("--value-hidden-dim", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--eval-seeds", type=int, default=None)
    parser.add_argument("--show-progress", type=str, default=None)
    parser.add_argument("--bot", type=str, default=None)
    parser.add_argument("--bot-difficulty", type=str, default=None)
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def parse_bool_flag(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def build_model_config(raw_model: dict | None, args: argparse.Namespace) -> ModelConfig:
    raw_model = raw_model or {}
    preset_name = str(args.model_preset or raw_model.get("preset", "base")).lower()
    if preset_name == "custom":
        return ModelConfig(
            channels=int(args.model_channels if args.model_channels is not None else raw_model.get("channels", ModelConfig().channels)),
            blocks=int(args.model_blocks if args.model_blocks is not None else raw_model.get("blocks", ModelConfig().blocks)),
            policy_channels=int(
                args.policy_channels
                if args.policy_channels is not None
                else raw_model.get("policy_channels", ModelConfig().policy_channels)
            ),
            value_channels=int(
                args.value_channels
                if args.value_channels is not None
                else raw_model.get("value_channels", ModelConfig().value_channels)
            ),
            value_hidden_dim=int(
                args.value_hidden_dim
                if args.value_hidden_dim is not None
                else raw_model.get("value_hidden_dim", ModelConfig().value_hidden_dim)
            ),
        )

    base = model_preset_config(preset_name)
    return ModelConfig(
        channels=int(args.model_channels if args.model_channels is not None else base.channels),
        blocks=int(args.model_blocks if args.model_blocks is not None else base.blocks),
        policy_channels=int(args.policy_channels if args.policy_channels is not None else base.policy_channels),
        value_channels=int(args.value_channels if args.value_channels is not None else base.value_channels),
        value_hidden_dim=int(args.value_hidden_dim if args.value_hidden_dim is not None else base.value_hidden_dim),
    )


def save_checkpoint(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    path: Path,
    seed: int,
    history_tail: list[dict[str, float]],
    update: int,
    model_config: ModelConfig,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "history_tail": history_tail,
        "seed": seed,
        "update": update,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_config": model_config.to_dict(),
    }
    if extra is not None:
        payload["extra"] = {"model_config": model_config.to_dict(), **extra}
    else:
        payload["extra"] = {"model_config": model_config.to_dict()}
    torch.save(payload, path)


def load_config(config_path: Path) -> dict:
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def load_reward_config(config_path: Path | None) -> dict | None:
    if config_path is None:
        return None
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def flatten_reward_payload(raw_reward: dict | None) -> dict | None:
    if raw_reward is None:
        return None

    if "reward" in raw_reward and isinstance(raw_reward["reward"], dict):
        raw_reward = raw_reward["reward"]

    flattened: dict[str, object] = {}
    for value in raw_reward.values():
        if isinstance(value, dict):
            flattened.update(value)
    if flattened:
        return flattened
    return raw_reward


def build_reward_config(raw_reward: dict | None) -> RewardConfig:
    flattened_reward = flatten_reward_payload(raw_reward)
    if flattened_reward is None:
        return DEFAULT_REWARD_CONFIG

    defaults = DEFAULT_REWARD_CONFIG.__dict__
    payload = {key: flattened_reward.get(key, defaults[key]) for key in defaults}
    return RewardConfig(**payload)


def config_path(value: str) -> Path | None:
    return Path(value).resolve() if value else None


def infer_run_dir_from_resume_path(resume_path: Path, artifacts: dict) -> Path:
    resolved = resume_path.resolve()
    checkpoint_dir_name = str(artifacts["checkpoint_dir"])
    final_model_name = str(artifacts["final_model_name"])

    if resolved.parent.name == checkpoint_dir_name:
        return resolved.parent.parent
    if resolved.name == final_model_name:
        return resolved.parent
    return resolved.parent


def resolve_run_layout(config: dict) -> dict[str, Path]:
    runtime = config["runtime"]
    artifacts = config["artifacts"]
    resume_from = runtime["resume_from"]
    run_name = runtime["run_name"]
    run_root = runtime["run_root"]
    start_new_branch = bool(runtime.get("start_new_branch", False))

    if resume_from is not None and run_name is None and not start_new_branch:
        run_dir = infer_run_dir_from_resume_path(resume_from, artifacts)
    else:
        resolved_run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (run_root / resolved_run_name).resolve()

    checkpoint_dir = run_dir / artifacts["checkpoint_dir"]
    log_dir = run_dir / artifacts["tensorboard_dir"]
    history_dir = run_dir / str(artifacts.get("history_dir", "history"))
    save_path = run_dir / artifacts["final_model_name"]
    manifest_path = run_dir / artifacts["manifest_name"]
    latest_eval_path = run_dir / artifacts["latest_eval_name"]
    registry_path = run_dir.parent / "index.json"

    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "history_dir": history_dir,
        "save_path": save_path,
        "manifest_path": manifest_path,
        "latest_eval_path": latest_eval_path,
        "registry_path": registry_path,
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def persist_run_metadata(layout: dict[str, Path], manifest: dict) -> None:
    write_manifest(layout["manifest_path"], manifest)
    update_registry(layout["registry_path"], manifest)


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def should_keep_checkpoint(update: int, keep_milestone_every: int) -> bool:
    return keep_milestone_every > 0 and update % keep_milestone_every == 0


def prune_checkpoints(checkpoint_dir: Path, keep_last: int, keep_milestone_every: int, best_model_name: str) -> list[str]:
    if keep_last <= 0:
        return []

    periodic = sorted(checkpoint_dir.glob("checkpoint_update_*.pt"))
    if len(periodic) <= keep_last:
        return []

    recent = {path.name for path in periodic[-keep_last:]}
    removed: list[str] = []
    for path in periodic[:-keep_last]:
        stem = path.stem
        update = int(stem.rsplit("_", 1)[-1])
        if should_keep_checkpoint(update, keep_milestone_every):
            continue
        if path.name == best_model_name:
            continue
        path.unlink(missing_ok=True)
        removed.append(str(path))
    return removed


def print_training_strategy(config: dict[str, dict], layout: dict[str, Path]) -> None:
    runtime = config["runtime"]
    training = config["training"]
    model = config["model"]
    evaluation = config["evaluation"]
    opponent = config["opponent"]
    checkpoint = config["checkpoint"]
    artifacts = config["artifacts"]

    print("=== training strategy ===")
    print(
        "runtime: device={device} seed={seed} run_dir={run_dir}".format(
            device=runtime["device"],
            seed=runtime["seed"],
            run_dir=layout["run_dir"],
        )
    )
    print(
        "model: preset={preset} channels={channels} blocks={blocks} policy_channels={policy_channels} "
        "value_channels={value_channels} value_hidden_dim={value_hidden_dim}".format(**model)
    )
    print(
        "ppo: n_envs={n_envs} n_steps={n_steps} updates={updates} batch_size={batch_size} epochs={epochs} "
        "lr={lr} show_progress={show_progress}".format(**training)
    )
    print(
        "evaluation: eval_every={eval_every} eval_games={eval_games} eval_seeds={eval_seeds}".format(**evaluation)
    )
    print(
        "opponent: bot_name={bot_name} bot_difficulty={bot_difficulty}".format(**opponent)
    )
    print(
        "checkpoint: save_every={save_every} keep_best_model={keep_best_model} keep_final_model={keep_final_model} "
        "keep_last={keep_last} keep_milestone_every={keep_milestone_every}".format(**checkpoint)
    )
    print(
        "paths: checkpoint_dir={checkpoint_dir} tensorboard_dir={tensorboard_dir} final_model={final_model_name} "
        "latest_eval={latest_eval_name} history_dir={history_dir}".format(**artifacts)
    )
    print(
        "resume: resume_from={resume_from} start_new_branch={start_new_branch} reward_config={reward_config}".format(
            resume_from=runtime["resume_from"],
            start_new_branch=runtime["start_new_branch"],
            reward_config=runtime["reward_config"],
        )
    )
    print("=========================")


def format_update_message(stats: dict[str, float]) -> str:
    return (
        "update={update} episodes={episodes} ep_rew_mean={ep_rew_mean:.2f} "
        "wins={wins} losses={losses} draws={draws} value_loss={value_loss:.4f} entropy={entropy:.4f} "
        "rollout={rollout_time_s:.1f}s optimize={optimize_time_s:.1f}s fps={samples_per_sec:.1f}"
    ).format(**stats)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def append_csv_row(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def write_history_strategy(path: Path, config: dict[str, dict], layout: dict[str, Path]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "layout": {key: str(value) for key, value in layout.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


UPDATE_CSV_FIELDS = [
    "update",
    "episodes",
    "wins",
    "losses",
    "draws",
    "ep_rew_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "rollout_time_s",
    "optimize_time_s",
    "total_time_s",
    "samples_per_sec",
    "offense_bonus",
    "defense_bonus",
    "block_winning_bonus",
    "block_four_bonus",
    "unresolved_winning_penalty",
    "unresolved_four_penalty",
    "unresolved_live_three_penalty",
]


def main() -> None:
    args = build_parser().parse_args()
    raw_config = load_config(args.config)
    runtime = raw_config["runtime"]
    training = raw_config["training"]
    evaluation = raw_config["evaluation"]
    opponent_config = raw_config.get("opponent", {})
    artifacts = raw_config["artifacts"]
    checkpoint_policy = raw_config["checkpoint"]
    model_preset = str(args.model_preset or raw_config.get("model", {}).get("preset", "base")).lower()
    model_config = build_model_config(raw_config.get("model"), args)
    reward_config_path = config_path(runtime.get("reward_config"))
    if reward_config_path is None:
        reward_config_path = (args.config.parent / "reward.toml").resolve()
    reward = build_reward_config(load_reward_config(reward_config_path))

    config: dict[str, dict] = {
        "runtime": {
            "device": args.device or runtime["device"],
            "seed": int(args.seed if args.seed is not None else runtime["seed"]),
            "run_root": Path(runtime["run_root"]).resolve(),
            "run_name": args.run_name if args.run_name is not None else (runtime["run_name"] or None),
            "resume_from": args.resume_from.resolve() if args.resume_from is not None else config_path(runtime["resume_from"]),
            "start_new_branch": parse_bool_flag(args.start_new_branch, default=bool(runtime.get("start_new_branch", False))),
            "reward_config": reward_config_path,
        },
        "training": {
            "n_envs": int(args.n_envs if args.n_envs is not None else training["n_envs"]),
            "n_steps": int(args.n_steps if args.n_steps is not None else training["n_steps"]),
            "updates": int(args.updates if args.updates is not None else training["updates"]),
            "batch_size": int(args.batch_size if args.batch_size is not None else training["batch_size"]),
            "epochs": int(args.epochs if args.epochs is not None else training["epochs"]),
            "lr": float(args.lr if args.lr is not None else training["lr"]),
            "show_progress": (
                str(args.show_progress).lower() == "true"
                if args.show_progress is not None
                else bool(training.get("show_progress", True))
            ),
        },
        "model": {"preset": model_preset, **model_config.to_dict()},
        "evaluation": {
            "eval_every": int(args.eval_every if args.eval_every is not None else evaluation["eval_every"]),
            "eval_games": int(args.eval_games if args.eval_games is not None else evaluation["eval_games"]),
            "eval_seeds": int(args.eval_seeds if args.eval_seeds is not None else evaluation["eval_seeds"]),
        },
        "opponent": {
            "bot_name": str(args.bot if args.bot is not None else opponent_config.get("bot_name", "reward_driven_hard")),
            "bot_difficulty": str(
                args.bot_difficulty if args.bot_difficulty is not None else opponent_config.get("bot_difficulty", "")
            ),
        },
        "artifacts": {
            "tensorboard_dir": str(artifacts["tensorboard_dir"]),
            "checkpoint_dir": str(artifacts["checkpoint_dir"]),
            "history_dir": str(artifacts.get("history_dir", "history")),
            "final_model_name": str(artifacts["final_model_name"]),
            "latest_eval_name": str(artifacts["latest_eval_name"]),
            "manifest_name": str(artifacts["manifest_name"]),
        },
        "checkpoint": {
            "save_every": int(args.save_every if args.save_every is not None else checkpoint_policy["save_every"]),
            "keep_best_model": bool(checkpoint_policy["keep_best_model"]),
            "best_model_name": str(checkpoint_policy["best_model_name"]),
            "keep_final_model": bool(checkpoint_policy["keep_final_model"]),
            "keep_last": int(checkpoint_policy["keep_last"]),
            "keep_milestone_every": int(checkpoint_policy["keep_milestone_every"]),
        },
        "reward": reward.__dict__.copy(),
    }

    torch.manual_seed(config["runtime"]["seed"])
    device = resolve_device(config["runtime"]["device"])
    config["runtime"]["device"] = device
    layout = resolve_run_layout(config)
    print_training_strategy(config, layout)

    opponent = create_bot(
        name=config["opponent"]["bot_name"],
        difficulty=(config["opponent"]["bot_difficulty"] or None),
        reward_config=reward,
    )
    env = VectorEnv(
        GomokuEnv(
            opponent=opponent,
            seed=config["runtime"]["seed"] + idx,
            reward_config=reward,
        )
        for idx in range(config["training"]["n_envs"])
    )
    model = ActorCriticNet(model_config)
    ppo_config = PPOConfig(
        n_envs=config["training"]["n_envs"],
        n_steps=config["training"]["n_steps"],
        total_updates=config["training"]["updates"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["lr"],
        device=device,
        show_progress=config["training"]["show_progress"],
    )
    layout["run_dir"].mkdir(parents=True, exist_ok=True)
    layout["log_dir"].mkdir(parents=True, exist_ok=True)
    layout["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    layout["history_dir"].mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(layout["log_dir"]))
    trainer = PPOTrainer(model=model, env=env, config=ppo_config, writer=writer)
    best_eval: dict[str, float] | None = None
    history_tail: list[dict[str, float]] = []
    best_model_path = layout["checkpoint_dir"] / config["checkpoint"]["best_model_name"]
    latest_metrics_path = layout["latest_eval_path"]
    updates_jsonl_path = layout["history_dir"] / "updates.jsonl"
    updates_csv_path = layout["history_dir"] / "updates.csv"
    updates_log_path = layout["history_dir"] / "updates.log"
    evals_jsonl_path = layout["history_dir"] / "evals.jsonl"
    strategy_path = layout["history_dir"] / "training_strategy.json"
    start_update = 0
    write_history_strategy(strategy_path, config, layout)

    manifest: dict[str, object] = {
        "run_dir": str(layout["run_dir"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "seed": config["runtime"]["seed"],
        "paths": {
            "final_model": str(layout["save_path"]),
            "checkpoint_dir": str(layout["checkpoint_dir"]),
            "tensorboard_dir": str(layout["log_dir"]),
            "history_dir": str(layout["history_dir"]),
            "best_model": str(best_model_path),
            "latest_eval": str(latest_metrics_path),
        },
        "config": config,
        "status": "running",
        "last_update": start_update,
        "best_eval": None,
        "latest_eval": None,
        "latest_checkpoint": None,
        "resume_from": str(config["runtime"]["resume_from"]) if config["runtime"]["resume_from"] is not None else None,
        "removed_checkpoints": [],
    }
    persist_run_metadata(layout, manifest)

    if config["runtime"]["resume_from"] is not None:
        resume_payload = torch.load(config["runtime"]["resume_from"], map_location=trainer.config.device)
        resume_model_config = model_config_from_checkpoint_payload(resume_payload)
        if resume_model_config != model.config:
            model = ActorCriticNet(resume_model_config).to(device)
            trainer = PPOTrainer(model=model, env=env, config=ppo_config, writer=writer)
        trainer.model.load_state_dict(resume_payload["model_state_dict"])
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            trainer.optimizer.load_state_dict(optimizer_state)
        start_update = int(resume_payload.get("update", 0))
        history_tail = list(resume_payload.get("history_tail", []))
        resume_extra = resume_payload.get("extra", {})
        best_eval = resume_extra.get("best_eval") or resume_extra.get("eval_metrics")
        resumed_seed = resume_payload.get("seed")
        config["model"] = {"preset": "checkpoint", **resume_model_config.to_dict()}
        manifest["config"]["model"] = {"preset": "checkpoint", **resume_model_config.to_dict()}
        print(f"resumed from {config['runtime']['resume_from']} at update {start_update}")
        if resumed_seed is not None and resumed_seed != config["runtime"]["seed"]:
            print(f"resume checkpoint seed={resumed_seed}, current run seed={config['runtime']['seed']}")
        manifest["last_update"] = start_update
        manifest["best_eval"] = best_eval
        persist_run_metadata(layout, manifest)

    def on_update_end(update: int, stats: dict[str, float], current_model: ActorCriticNet) -> None:
        nonlocal best_eval
        history_tail.append(stats)
        del history_tail[:-10]
        manifest["last_update"] = update
        update_message = format_update_message(stats)
        append_jsonl(
            updates_jsonl_path,
            {
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "message": update_message,
                "stats": stats,
                "model": config["model"],
                "training": config["training"],
            },
        )
        append_csv_row(updates_csv_path, stats, UPDATE_CSV_FIELDS)
        with updates_log_path.open("a", encoding="utf-8") as handle:
            handle.write(update_message + "\n")

        if config["checkpoint"]["save_every"] > 0 and update % config["checkpoint"]["save_every"] == 0:
            checkpoint_path = layout["checkpoint_dir"] / f"checkpoint_update_{update:04d}.pt"
            save_checkpoint(
                model=current_model,
                optimizer=trainer.optimizer,
                config=ppo_config,
                path=checkpoint_path,
                seed=config["runtime"]["seed"],
                history_tail=history_tail,
                update=update,
                model_config=current_model.config,
                extra={"train_stats": stats},
            )
            manifest["latest_checkpoint"] = str(checkpoint_path)
            removed = prune_checkpoints(
                checkpoint_dir=layout["checkpoint_dir"],
                keep_last=config["checkpoint"]["keep_last"],
                keep_milestone_every=config["checkpoint"]["keep_milestone_every"],
                best_model_name=config["checkpoint"]["best_model_name"],
            )
            if removed:
                manifest["removed_checkpoints"].extend(removed)
            persist_run_metadata(layout, manifest)
            print(f"saved checkpoint: {checkpoint_path}")

        if config["evaluation"]["eval_every"] > 0 and update % config["evaluation"]["eval_every"] == 0:
            eval_seeds = [
                config["runtime"]["seed"] + update * 100 + idx
                for idx in range(config["evaluation"]["eval_seeds"])
            ]
            eval_metrics = evaluate_across_seeds(
                model=current_model,
                games=config["evaluation"]["eval_games"],
                device=device,
                seeds=eval_seeds,
                verbose=False,
            )
            eval_metrics["update"] = float(update)
            eval_metrics["ep_rew_mean"] = stats["ep_rew_mean"]
            writer.add_scalar("eval/win_rate", eval_metrics["win_rate"], update)
            writer.add_scalar("eval/avg_steps", eval_metrics["avg_steps"], update)
            writer.add_scalar("eval/illegal_moves", eval_metrics["illegal_moves"], update)
            writer.add_scalar("eval/win_rate_std", eval_metrics["win_rate_std"], update)
            writer.flush()

            latest_metrics_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
            manifest["latest_eval"] = eval_metrics
            append_jsonl(
                evals_jsonl_path,
                {
                    "recorded_at": datetime.now().isoformat(timespec="seconds"),
                    "metrics": eval_metrics,
                    "update_stats": stats,
                },
            )
            print(
                "eval update={update} win_rate={win_rate:.3f} std={win_rate_std:.3f} wins={wins:.0f} "
                "losses={losses:.0f} draws={draws:.0f} avg_steps={avg_steps:.2f}".format(**eval_metrics)
            )

            is_better = best_eval is None or eval_metrics["win_rate"] > best_eval["win_rate"]
            tie_break = (
                best_eval is not None
                and eval_metrics["win_rate"] == best_eval["win_rate"]
                and eval_metrics["avg_steps"] < best_eval["avg_steps"]
            )
            if config["checkpoint"]["keep_best_model"] and (is_better or tie_break):
                best_eval = eval_metrics
                save_checkpoint(
                    model=current_model,
                    optimizer=trainer.optimizer,
                    config=ppo_config,
                    path=best_model_path,
                    seed=config["runtime"]["seed"],
                    history_tail=history_tail,
                    update=update,
                    model_config=current_model.config,
                    extra={"eval_metrics": eval_metrics, "train_stats": stats},
                )
                manifest["best_eval"] = best_eval
                manifest["paths"]["best_model"] = str(best_model_path)
                print(f"updated best model: {best_model_path}")
            persist_run_metadata(layout, manifest)

    history = trainer.train(start_update=start_update, on_update_end=on_update_end)
    writer.close()

    if config["checkpoint"]["keep_final_model"]:
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            config=ppo_config,
            path=layout["save_path"],
            seed=config["runtime"]["seed"],
            history_tail=history[-10:],
            update=start_update + config["training"]["updates"],
            model_config=model.config,
            extra={"best_eval": best_eval},
        )
    manifest["status"] = "completed"
    manifest["last_update"] = start_update + config["training"]["updates"]
    manifest["best_eval"] = best_eval
    manifest["paths"]["final_model"] = str(layout["save_path"]) if config["checkpoint"]["keep_final_model"] else None
    persist_run_metadata(layout, manifest)
    if config["checkpoint"]["keep_final_model"]:
        print(f"saved checkpoint to {layout['save_path']}")


if __name__ == "__main__":
    main()
