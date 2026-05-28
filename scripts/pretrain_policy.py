from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bots import create_bot
from gomoku_ai.cpp_backend import score_reward_candidates
from gomoku_ai.env import (
    BLACK,
    BOARD_AREA,
    BOARD_SIZE,
    EMPTY,
    action_to_coord,
    call_bot_action,
    neighboring_action_mask,
)
from gomoku_ai.model import MODEL_PRESETS, ActorCriticNet, model_preset_config


@dataclass(frozen=True)
class PretrainConfig:
    samples: int = 20000
    batch_size: int = 256
    learning_rate: float = 3e-4
    max_moves: int = 80
    score_temperature: float = 80.0
    candidate_top_k: int = 32
    seed: int = 20260527
    device: str = "auto"
    model_preset: str = "base"
    rollout_bots: tuple[str, ...] = ("random", "classic_rule", "reward_driven_medium")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain Gomoku policy from C++ tactical reward scores.")
    parser.add_argument("--samples", type=int, default=PretrainConfig.samples)
    parser.add_argument("--batch-size", type=int, default=PretrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=PretrainConfig.learning_rate)
    parser.add_argument("--max-moves", type=int, default=PretrainConfig.max_moves)
    parser.add_argument("--score-temperature", type=float, default=PretrainConfig.score_temperature)
    parser.add_argument("--candidate-top-k", type=int, default=PretrainConfig.candidate_top_k)
    parser.add_argument("--seed", type=int, default=PretrainConfig.seed)
    parser.add_argument("--device", type=str, default=PretrainConfig.device)
    parser.add_argument("--model-preset", type=str, choices=MODEL_PRESETS, default=PretrainConfig.model_preset)
    parser.add_argument("--rollout-bots", nargs="*", default=list(PretrainConfig.rollout_bots))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pretrain"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build_observation(board: np.ndarray, current_player: int, last_move: int | None) -> np.ndarray:
    obs = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    obs[0] = (board == current_player).astype(np.float32)
    obs[1] = (board == -current_player).astype(np.float32)
    if last_move is not None:
        row, col = action_to_coord(last_move)
        obs[2, row, col] = 1.0
    if current_player == BLACK:
        obs[3] = 1.0
    return obs


def has_five(board: np.ndarray, row: int, col: int, player: int) -> bool:
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            rr = row + dr * sign
            cc = col + dc * sign
            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == player:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= 5:
            return True
    return False


def make_soft_target(
    board: np.ndarray,
    current_player: int,
    mask: np.ndarray,
    *,
    temperature: float,
    candidate_top_k: int,
) -> np.ndarray:
    actions = np.flatnonzero(mask).astype(np.int64)
    if actions.size == 0:
        raise ValueError("cannot build target without legal actions")
    scored = score_reward_candidates(board, actions.tolist(), current_player)
    scored_pairs = sorted(
        ((float(item["reward"]), int(item["action"])) for item in scored),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = scored_pairs[: max(1, min(candidate_top_k, len(scored_pairs)))]
    scores = np.asarray([item[0] for item in selected], dtype=np.float64)
    logits = (scores - scores.max()) / max(1e-6, float(temperature))
    weights = np.exp(logits)
    probs = weights / weights.sum()
    target = np.zeros(BOARD_AREA, dtype=np.float32)
    for probability, (_, action) in zip(probs, selected):
        target[action] = float(probability)
    return target


def sample_rollout_bot(bots: list, rng: np.random.Generator):
    return bots[int(rng.integers(0, len(bots)))]


def collect_batch(
    *,
    batch_size: int,
    max_moves: int,
    bots: list,
    rng: np.random.Generator,
    temperature: float,
    candidate_top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    games = 0
    moves = 0

    while len(observations) < batch_size:
        games += 1
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        current_player = BLACK
        last_move: int | None = None
        black_bot = sample_rollout_bot(bots, rng)
        white_bot = sample_rollout_bot(bots, rng)

        for _ in range(max_moves):
            if len(observations) >= batch_size:
                break
            mask = neighboring_action_mask(board, radius=2, opening_radius=1)
            observations.append(build_observation(board, current_player, last_move))
            masks.append(mask.astype(bool))
            targets.append(
                make_soft_target(
                    board,
                    current_player,
                    mask,
                    temperature=temperature,
                    candidate_top_k=candidate_top_k,
                )
            )

            bot = black_bot if current_player == BLACK else white_bot
            action = call_bot_action(bot, board.copy(), current_player, rng)
            row, col = action_to_coord(action)
            if board[row, col] != EMPTY:
                legal = np.flatnonzero(board.reshape(-1) == EMPTY)
                if legal.size == 0:
                    break
                action = int(rng.choice(legal))
                row, col = action_to_coord(action)
            board[row, col] = current_player
            moves += 1
            last_move = action
            if has_five(board, row, col, current_player):
                break
            if not np.any(board == EMPTY):
                break
            current_player = -current_player

    return (
        np.stack(observations),
        np.stack(masks),
        np.stack(targets),
        {"games": float(games), "rollout_moves": float(moves)},
    )


def policy_loss_from_soft_targets(
    model: ActorCriticNet,
    observations: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, values = model(observations)
    masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.softmax(masked_logits, dim=-1)
    policy_loss = -(targets * log_probs).sum(dim=-1).mean()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    top1 = (torch.argmax(probs, dim=-1) == torch.argmax(targets, dim=-1)).float().mean()
    value_l2 = values.square().mean()
    loss = policy_loss + 0.001 * value_l2
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
        "target_top1": float(top1.detach().cpu().item()),
        "value_l2": float(value_l2.detach().cpu().item()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    config: PretrainConfig,
    model_config: dict[str, int],
    history_tail: list[dict[str, float]],
    steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {"pretrain": asdict(config)},
            "history_tail": history_tail[-10:],
            "seed": config.seed,
            "update": steps,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model_config": model_config,
            "extra": {
                "model_config": model_config,
                "pretrain": asdict(config),
                "checkpoint_kind": "tactical_policy_pretrain",
            },
        },
        path,
    )


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    config = PretrainConfig(
        samples=int(args.samples),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        max_moves=int(args.max_moves),
        score_temperature=float(args.score_temperature),
        candidate_top_k=int(args.candidate_top_k),
        seed=int(args.seed),
        device=resolve_device(args.device),
        model_preset=str(args.model_preset),
        rollout_bots=tuple(args.rollout_bots),
    )
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.resolve() / run_name
    history_path = run_dir / "history.jsonl"
    manifest_path = run_dir / "manifest.json"
    checkpoint_path = run_dir / "policy_pretrain.pt"

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    bots = [create_bot(name=name) for name in config.rollout_bots]
    model_config = model_preset_config(config.model_preset)
    model = ActorCriticNet(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    model.train()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "config": asdict(config),
        "model_config": model_config.to_dict(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history: list[dict[str, float]] = []
    samples_seen = 0
    update = 0
    started = perf_counter()
    while samples_seen < config.samples:
        batch_started = perf_counter()
        obs_np, masks_np, targets_np, batch_stats = collect_batch(
            batch_size=min(config.batch_size, config.samples - samples_seen),
            max_moves=config.max_moves,
            bots=bots,
            rng=rng,
            temperature=config.score_temperature,
            candidate_top_k=config.candidate_top_k,
        )
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=config.device)
        masks = torch.as_tensor(masks_np, dtype=torch.bool, device=config.device)
        targets = torch.as_tensor(targets_np, dtype=torch.float32, device=config.device)

        loss, metrics = policy_loss_from_soft_targets(model, obs, masks, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        update += 1
        samples_seen += obs_np.shape[0]
        elapsed = perf_counter() - started
        row = {
            "update": float(update),
            "samples_seen": float(samples_seen),
            "elapsed_s": elapsed,
            "batch_time_s": perf_counter() - batch_started,
            "samples_per_sec": samples_seen / max(elapsed, 1e-6),
            **batch_stats,
            **metrics,
        }
        history.append(row)
        append_jsonl(history_path, row)
        print(
            "pretrain update={update:.0f} samples={samples_seen:.0f}/{total} "
            "loss={policy_loss:.4f} top1={target_top1:.3f} entropy={entropy:.3f} "
            "fps={samples_per_sec:.1f}".format(total=config.samples, **row)
        )

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        model_config=model_config.to_dict(),
        history_tail=history,
        steps=samples_seen,
    )
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["samples_seen"] = samples_seen
    manifest["latest_metrics"] = history[-1] if history else None
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
