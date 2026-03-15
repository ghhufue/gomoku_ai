from __future__ import annotations

import argparse
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
from gomoku_ai.model import ActorCriticNet
from gomoku_ai.ppo import PPOConfig, PPOTrainer
from gomoku_ai.run_registry import update_registry
from gomoku_ai.rule_bot import RuleBasedBot
from scripts.evaluate import evaluate_across_seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the initial Gomoku PPO prototype.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--eval-seeds", type=int, default=None)
    parser.add_argument("--show-progress", type=str, default=None)
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def save_checkpoint(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    path: Path,
    seed: int,
    history_tail: list[dict[str, float]],
    update: int,
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
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_resume_state(
    resume_path: Path,
    model: ActorCriticNet,
    trainer: PPOTrainer,
) -> dict:
    payload = torch.load(resume_path, map_location=trainer.config.device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        trainer.optimizer.load_state_dict(optimizer_state)
    return payload


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


def resolve_run_layout(config: dict) -> dict[str, Path]:
    runtime = config["runtime"]
    artifacts = config["artifacts"]
    resume_from = runtime["resume_from"]
    run_name = runtime["run_name"]
    run_root = runtime["run_root"]

    if resume_from is not None and run_name is None:
        run_dir = resume_from.resolve().parent.parent
    else:
        resolved_run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (run_root / resolved_run_name).resolve()

    checkpoint_dir = run_dir / artifacts["checkpoint_dir"]
    log_dir = run_dir / artifacts["tensorboard_dir"]
    save_path = run_dir / artifacts["final_model_name"]
    manifest_path = run_dir / artifacts["manifest_name"]
    latest_eval_path = run_dir / artifacts["latest_eval_name"]
    registry_path = run_dir.parent / "index.json"

    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
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


def main() -> None:
    args = build_parser().parse_args()
    raw_config = load_config(args.config)
    runtime = raw_config["runtime"]
    training = raw_config["training"]
    evaluation = raw_config["evaluation"]
    artifacts = raw_config["artifacts"]
    checkpoint_policy = raw_config["checkpoint"]
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
        "evaluation": {
            "eval_every": int(args.eval_every if args.eval_every is not None else evaluation["eval_every"]),
            "eval_games": int(args.eval_games if args.eval_games is not None else evaluation["eval_games"]),
            "eval_seeds": int(args.eval_seeds if args.eval_seeds is not None else evaluation["eval_seeds"]),
        },
        "artifacts": {
            "tensorboard_dir": str(artifacts["tensorboard_dir"]),
            "checkpoint_dir": str(artifacts["checkpoint_dir"]),
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
    print(f"using device: {device}")
    print(f"run directory: {layout['run_dir']}")
    print(
        "training config: n_envs={n_envs} n_steps={n_steps} updates={updates} batch_size={batch_size} epochs={epochs} eval_every={eval_every}".format(
            n_envs=config["training"]["n_envs"],
            n_steps=config["training"]["n_steps"],
            updates=config["training"]["updates"],
            batch_size=config["training"]["batch_size"],
            epochs=config["training"]["epochs"],
            eval_every=config["evaluation"]["eval_every"],
        )
    )

    opponent = RuleBasedBot()
    env = VectorEnv(
        GomokuEnv(
            opponent=opponent,
            seed=config["runtime"]["seed"] + idx,
            reward_config=reward,
        )
        for idx in range(config["training"]["n_envs"])
    )
    model = ActorCriticNet()
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
    writer = SummaryWriter(log_dir=str(layout["log_dir"]))
    trainer = PPOTrainer(model=model, env=env, config=ppo_config, writer=writer)
    best_eval: dict[str, float] | None = None
    history_tail: list[dict[str, float]] = []
    best_model_path = layout["checkpoint_dir"] / config["checkpoint"]["best_model_name"]
    latest_metrics_path = layout["latest_eval_path"]
    start_update = 0

    manifest: dict[str, object] = {
        "run_dir": str(layout["run_dir"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "seed": config["runtime"]["seed"],
        "paths": {
            "final_model": str(layout["save_path"]),
            "checkpoint_dir": str(layout["checkpoint_dir"]),
            "tensorboard_dir": str(layout["log_dir"]),
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
        resume_payload = load_resume_state(config["runtime"]["resume_from"], model=model, trainer=trainer)
        start_update = int(resume_payload.get("update", 0))
        history_tail = list(resume_payload.get("history_tail", []))
        resume_extra = resume_payload.get("extra", {})
        best_eval = resume_extra.get("best_eval") or resume_extra.get("eval_metrics")
        resumed_seed = resume_payload.get("seed")
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
