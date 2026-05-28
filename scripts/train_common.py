from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import EnvConfig, GomokuEnv, VectorEnv
from gomoku_ai.model import ActorCriticNet, model_preset_config
from gomoku_ai.opponents import MixedOpponentSampler
from gomoku_ai.ppo import PPOConfig, PPOTrainer


@dataclass
class AlphaSchedule:
    start: float
    end: float
    warmup_fraction: float = 0.3
    decay_fraction: float = 0.4

    def value(self, update: int, total_updates: int) -> float:
        progress = update / max(1, total_updates)
        if progress <= self.warmup_fraction:
            return self.start
        decay_end = self.warmup_fraction + self.decay_fraction
        if progress >= decay_end:
            return self.end
        local = (progress - self.warmup_fraction) / max(self.decay_fraction, 1e-8)
        return self.start + local * (self.end - self.start)


@dataclass
class StageConfig:
    stage_name: str
    description: str
    reward_mode: str
    shape_alpha: float
    use_action_mask: bool
    bot_pool: tuple[str, ...] = ("random",)
    bot_weights: tuple[float, ...] = (1.0,)
    checkpoint_paths: tuple[str, ...] = ()
    checkpoint_weight: float = 0.0
    alpha_schedule: AlphaSchedule | None = None


def build_parser(stage: StageConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=stage.description)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-preset", default="small")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Historical model checkpoint to include as an episode-sampled opponent.",
    )
    parser.add_argument("--checkpoint-weight", type=float, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def make_envs(stage: StageConfig, args: argparse.Namespace) -> VectorEnv:
    envs = []
    for index in range(args.n_envs):
        envs.append(
            GomokuEnv(
                opponent=MixedOpponentSampler(
                    stage.bot_pool,
                    stage.bot_weights,
                    checkpoint_paths=stage.checkpoint_paths,
                    checkpoint_weight=stage.checkpoint_weight,
                    device=resolve_device(args.device),
                ),
                seed=args.seed + index,
                config=EnvConfig(
                    reward_mode=stage.reward_mode,
                    shape_alpha=stage.shape_alpha,
                    use_action_mask=stage.use_action_mask,
                ),
            )
        )
    return VectorEnv(envs)


def resolve_run_dir(stage_name: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (ROOT / "runs" / stage_name / timestamp).resolve()


def save_checkpoint(path: Path, model: ActorCriticNet, trainer: PPOTrainer, update: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "update": update,
            "metadata": to_jsonable(metadata),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def run_stage(stage: StageConfig) -> None:
    parser = build_parser(stage)
    args = parser.parse_args()
    if args.opponent_checkpoint:
        stage.checkpoint_paths = tuple(str(path.resolve()) for path in args.opponent_checkpoint)
        if args.checkpoint_weight is not None:
            stage.checkpoint_weight = float(args.checkpoint_weight)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = resolve_run_dir(stage.stage_name, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticNet(model_preset_config(args.model_preset))
    env = make_envs(stage, args)
    ppo_config = PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_updates=args.updates,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        show_progress=not args.no_progress,
    )
    trainer = PPOTrainer(model=model, env=env, config=ppo_config)
    metadata = {
        "stage": asdict(stage),
        "args": vars(args),
        "device": device,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "training_config.json", metadata)
    history: list[dict[str, float]] = []

    def on_update_end(update: int, stats: dict[str, float], current_model: ActorCriticNet) -> None:
        if stage.alpha_schedule is not None:
            alpha = stage.alpha_schedule.value(update, args.updates)
            env.set_shape_alpha(alpha)
            stats["shape_alpha"] = alpha
        history.append(stats)
        write_json(run_dir / "history.json", {"updates": history})
        if args.save_every > 0 and update % args.save_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"checkpoint_update_{update:04d}.pt",
                current_model,
                trainer,
                update,
                metadata,
            )

    print(f"stage={stage.stage_name}")
    print(f"description={stage.description}")
    print(f"device={device} run_dir={run_dir}")
    print(f"reward_mode={stage.reward_mode} shape_alpha={stage.shape_alpha} action_mask={stage.use_action_mask}")
    print(f"bot_pool={stage.bot_pool} weights={stage.bot_weights}")
    trainer.train(on_update_end=on_update_end)
    save_checkpoint(run_dir / "final_model.pt", trainer.model, trainer, args.updates, metadata)
    write_json(run_dir / "history.json", {"updates": history})
    print(f"saved final checkpoint: {run_dir / 'final_model.pt'}")
