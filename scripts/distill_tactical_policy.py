from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from gomoku_ai.model import ActorCriticNet, model_config_from_checkpoint_payload
from gomoku_ai.opening import apply_random_opening_pairs_to_board
from gomoku_ai.tactical_policy import select_tactical_search_action


@dataclass(frozen=True)
class DistillConfig:
    samples: int = 50000
    batch_size: int = 256
    learning_rate: float = 1e-4
    epochs_per_batch: int = 4
    max_moves: int = 90
    seed: int = 20260527
    device: str = "auto"
    teacher_checkpoint: str = ""
    student_init: str = ""
    rollout_bots: tuple[str, ...] = ("random", "classic_rule", "reward_driven_medium", "reward_driven_hard")
    rollout_bot_weights: tuple[float, ...] = (0.15, 0.30, 0.35, 0.20)
    include_teacher_self_play: bool = True
    teacher_self_play_prob: float = 0.35
    opening_random_pairs: int = 0
    soft_targets: bool = True
    teacher_top_k: int = 12
    teacher_temperature: float = 2.0
    scorer_weight: float = 0.25
    entropy_coef: float = 0.01
    num_workers: int = 1
    worker_batch_size: int = 32
    worker_device: str = "cpu"
    save_every: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill tactical search choices back into the policy network.")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-init", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=DistillConfig.samples)
    parser.add_argument("--batch-size", type=int, default=DistillConfig.batch_size)
    parser.add_argument("--lr", type=float, default=DistillConfig.learning_rate)
    parser.add_argument("--epochs-per-batch", type=int, default=DistillConfig.epochs_per_batch)
    parser.add_argument("--max-moves", type=int, default=DistillConfig.max_moves)
    parser.add_argument("--seed", type=int, default=DistillConfig.seed)
    parser.add_argument("--device", type=str, default=DistillConfig.device)
    parser.add_argument("--rollout-bots", nargs="*", default=list(DistillConfig.rollout_bots))
    parser.add_argument("--rollout-bot-weights", type=float, nargs="*", default=list(DistillConfig.rollout_bot_weights))
    parser.add_argument("--no-teacher-self-play", action="store_true")
    parser.add_argument("--teacher-self-play-prob", type=float, default=DistillConfig.teacher_self_play_prob)
    parser.add_argument("--opening-random-pairs", type=int, default=DistillConfig.opening_random_pairs)
    parser.add_argument("--hard-targets", action="store_true", help="Use one-hot teacher targets instead of top-k soft targets.")
    parser.add_argument("--teacher-top-k", type=int, default=DistillConfig.teacher_top_k)
    parser.add_argument("--teacher-temperature", type=float, default=DistillConfig.teacher_temperature)
    parser.add_argument("--scorer-weight", type=float, default=DistillConfig.scorer_weight)
    parser.add_argument("--entropy-coef", type=float, default=DistillConfig.entropy_coef)
    parser.add_argument("--num-workers", type=int, default=DistillConfig.num_workers)
    parser.add_argument("--worker-batch-size", type=int, default=DistillConfig.worker_batch_size)
    parser.add_argument("--worker-device", type=str, default=DistillConfig.worker_device)
    parser.add_argument("--save-every", type=int, default=DistillConfig.save_every)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/distill"))
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


def model_logits(model: ActorCriticNet, obs: np.ndarray, mask: np.ndarray, device: str) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask[None, ...], dtype=torch.bool, device=device)
    with torch.no_grad():
        dist, _ = model.masked_distribution(obs_tensor, mask_tensor)
    return dist.logits.squeeze(0).detach().cpu().numpy().astype(np.float64)


def teacher_action(
    model: ActorCriticNet,
    board: np.ndarray,
    player: int,
    obs: np.ndarray,
    mask: np.ndarray,
    device: str,
) -> tuple[int, str]:
    logits = model_logits(model, obs, mask, device)
    action, source, _ = select_tactical_search_action(board, player, logits)
    return int(action), source


def teacher_soft_target(
    model: ActorCriticNet,
    board: np.ndarray,
    player: int,
    obs: np.ndarray,
    mask: np.ndarray,
    device: str,
    *,
    top_k: int,
    temperature: float,
    scorer_weight: float,
) -> tuple[np.ndarray, int, str]:
    logits = model_logits(model, obs, mask, device)
    action, source, _ = select_tactical_search_action(board, player, logits)
    target = np.zeros(BOARD_AREA, dtype=np.float32)
    legal_actions = np.flatnonzero(mask).astype(np.int64).tolist()
    if not legal_actions:
        target[action] = 1.0
        return target, int(action), source

    scorer_logits = np.zeros(BOARD_AREA, dtype=np.float64)
    scorer_logits.fill(-1e9)
    scored = score_reward_candidates(board, legal_actions, player)
    for item in scored:
        scorer_logits[int(item["action"])] = float(item["reward"])
    finite = np.isfinite(scorer_logits) & (scorer_logits > -1e8)
    if np.any(finite):
        values = scorer_logits[finite]
        scale = max(1.0, float(np.std(values)))
        scorer_logits[finite] = (values - float(np.max(values))) / scale

    mixed = np.asarray(logits, dtype=np.float64).copy()
    mixed[~mask.astype(bool)] = -1e9
    mixed = mixed + float(scorer_weight) * scorer_logits
    mixed[int(action)] = max(float(mixed[int(action)]), float(np.max(mixed[mask.astype(bool)])) + 2.0)

    selected = np.argsort(mixed)[::-1]
    selected = [int(item) for item in selected if mask[int(item)]][: max(1, int(top_k))]
    selected_logits = np.asarray([mixed[item] for item in selected], dtype=np.float64)
    selected_logits = (selected_logits - selected_logits.max()) / max(1e-6, float(temperature))
    probs = np.exp(selected_logits)
    probs = probs / probs.sum()
    for item, prob in zip(selected, probs):
        target[item] = float(prob)
    return target, int(action), source


def select_rollout_action(
    model: ActorCriticNet,
    board: np.ndarray,
    player: int,
    last_move: int | None,
    bot,
    rng: np.random.Generator,
    device: str,
    teacher_self_play: bool,
) -> int:
    if teacher_self_play:
        mask = neighboring_action_mask(board, radius=2, opening_radius=1)
        obs = build_observation(board, player, last_move)
        action, _ = teacher_action(model, board, player, obs, mask, device)
        return action
    return int(call_bot_action(bot, board.copy(), player, rng))


def sample_bot(bots: list, weights: np.ndarray, rng: np.random.Generator):
    return bots[int(rng.choice(len(bots), p=weights))]


def collect_batch(
    *,
    model: ActorCriticNet,
    batch_size: int,
    max_moves: int,
    bots: list,
    bot_weights: np.ndarray,
    rng: np.random.Generator,
    device: str,
    include_teacher_self_play: bool,
    teacher_self_play_prob: float,
    opening_random_pairs: int,
    soft_targets: bool,
    teacher_top_k: int,
    teacher_temperature: float,
    scorer_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    targets: list[np.ndarray] = []
    source_counts: dict[str, int] = {}
    games = 0
    moves = 0

    while len(observations) < batch_size:
        games += 1
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        current_player = BLACK
        last_move: int | None = None
        if opening_random_pairs > 0:
            last_move, current_player = apply_random_opening_pairs_to_board(
                board,
                opening_random_pairs,
                rng,
                first_player=BLACK,
            )
        black_bot = sample_bot(bots, bot_weights, rng)
        white_bot = sample_bot(bots, bot_weights, rng)
        teacher_black = include_teacher_self_play and rng.random() < teacher_self_play_prob
        teacher_white = include_teacher_self_play and rng.random() < teacher_self_play_prob

        for _ in range(max_moves):
            if len(observations) >= batch_size:
                break
            mask = neighboring_action_mask(board, radius=2, opening_radius=1)
            obs = build_observation(board, current_player, last_move)
            if soft_targets:
                target, action, source = teacher_soft_target(
                    model,
                    board,
                    current_player,
                    obs,
                    mask,
                    device,
                    top_k=teacher_top_k,
                    temperature=teacher_temperature,
                    scorer_weight=scorer_weight,
                )
                targets.append(target)
            else:
                action, source = teacher_action(model, board, current_player, obs, mask, device)
            observations.append(obs)
            masks.append(mask.astype(bool))
            actions.append(action)
            source_counts[source] = source_counts.get(source, 0) + 1

            bot = black_bot if current_player == BLACK else white_bot
            teacher_self_play = teacher_black if current_player == BLACK else teacher_white
            rollout_action = select_rollout_action(
                model,
                board,
                current_player,
                last_move,
                bot,
                rng,
                device,
                teacher_self_play,
            )
            row, col = action_to_coord(rollout_action)
            if board[row, col] != EMPTY:
                legal = np.flatnonzero(board.reshape(-1) == EMPTY)
                if legal.size == 0:
                    break
                rollout_action = int(rng.choice(legal))
                row, col = action_to_coord(rollout_action)
            board[row, col] = current_player
            moves += 1
            last_move = rollout_action
            if has_five(board, row, col, current_player):
                break
            if not np.any(board == EMPTY):
                break
            current_player = -current_player

    stats: dict[str, float] = {"games": float(games), "rollout_moves": float(moves)}
    for source, count in source_counts.items():
        stats[f"source_{source}"] = float(count)
    target_payload = np.stack(targets) if soft_targets else np.asarray(actions, dtype=np.int64)
    return np.stack(observations), np.stack(masks), target_payload, stats


_WORKER_MODEL: ActorCriticNet | None = None
_WORKER_BOTS: list | None = None
_WORKER_WEIGHTS: np.ndarray | None = None
_WORKER_CONFIG: DistillConfig | None = None
_WORKER_DEVICE: str = "cpu"


def _init_worker(
    teacher_checkpoint: str,
    config_payload: dict,
    model_config_payload: dict,
    bot_weights: list[float],
) -> None:
    global _WORKER_MODEL, _WORKER_BOTS, _WORKER_WEIGHTS, _WORKER_CONFIG, _WORKER_DEVICE
    _WORKER_CONFIG = DistillConfig(**config_payload)
    _WORKER_DEVICE = resolve_device(_WORKER_CONFIG.worker_device)
    payload = torch.load(Path(teacher_checkpoint), map_location=_WORKER_DEVICE)
    from gomoku_ai.model import ModelConfig

    model_config = ModelConfig.from_dict(model_config_payload)
    model = ActorCriticNet(model_config).to(_WORKER_DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _WORKER_MODEL = model
    _WORKER_BOTS = [create_bot(name=name) for name in _WORKER_CONFIG.rollout_bots]
    _WORKER_WEIGHTS = np.asarray(bot_weights, dtype=np.float64)


def _worker_collect_batch(seed: int, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if _WORKER_MODEL is None or _WORKER_BOTS is None or _WORKER_WEIGHTS is None or _WORKER_CONFIG is None:
        raise RuntimeError("worker is not initialized")
    rng = np.random.default_rng(seed)
    return collect_batch(
        model=_WORKER_MODEL,
        batch_size=batch_size,
        max_moves=_WORKER_CONFIG.max_moves,
        bots=_WORKER_BOTS,
        bot_weights=_WORKER_WEIGHTS,
        rng=rng,
        device=_WORKER_DEVICE,
        include_teacher_self_play=_WORKER_CONFIG.include_teacher_self_play,
        teacher_self_play_prob=_WORKER_CONFIG.teacher_self_play_prob,
        opening_random_pairs=_WORKER_CONFIG.opening_random_pairs,
        soft_targets=_WORKER_CONFIG.soft_targets,
        teacher_top_k=_WORKER_CONFIG.teacher_top_k,
        teacher_temperature=_WORKER_CONFIG.teacher_temperature,
        scorer_weight=_WORKER_CONFIG.scorer_weight,
    )


def merge_batch_results(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]],
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    obs_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    stats: dict[str, float] = {}
    remaining = limit
    for obs, masks, targets, batch_stats in results:
        take = min(remaining, obs.shape[0])
        if take <= 0:
            break
        obs_parts.append(obs[:take])
        mask_parts.append(masks[:take])
        target_parts.append(targets[:take])
        fraction = take / max(1, obs.shape[0])
        for key, value in batch_stats.items():
            stats[key] = stats.get(key, 0.0) + float(value) * fraction
        remaining -= take
    return np.concatenate(obs_parts), np.concatenate(mask_parts), np.concatenate(target_parts), stats


def collect_batch_parallel(
    executor: ProcessPoolExecutor,
    *,
    batch_size: int,
    worker_batch_size: int,
    num_workers: int,
    seed_base: int,
    update: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    tasks = []
    remaining = batch_size
    worker_index = 0
    while remaining > 0:
        size = min(worker_batch_size, remaining)
        seed = seed_base + update * 100_000 + worker_index
        tasks.append(executor.submit(_worker_collect_batch, seed, size))
        remaining -= size
        worker_index += 1
        if len(tasks) >= max(1, num_workers) and remaining > 0:
            continue
    results = [task.result() for task in tasks]
    return merge_batch_results(results, batch_size)


def train_step(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    obs_np: np.ndarray,
    masks_np: np.ndarray,
    targets_np: np.ndarray,
    device: str,
    entropy_coef: float,
) -> dict[str, float]:
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    masks = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
    logits, values = model(obs)
    masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    if targets_np.ndim == 2:
        targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        target_actions = torch.argmax(targets, dim=-1)
    else:
        target_actions = torch.as_tensor(targets_np, dtype=torch.int64, device=device)
        loss = nn.functional.cross_entropy(masked_logits, target_actions)
    value_l2 = values.square().mean()
    total_loss = loss + 0.001 * value_l2 - float(entropy_coef) * entropy
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        pred = torch.argmax(masked_logits, dim=-1)
        top1 = (pred == target_actions).float().mean()
    return {
        "loss": float(total_loss.detach().cpu().item()),
        "policy_loss": float(loss.detach().cpu().item()),
        "target_top1": float(top1.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
        "value_l2": float(value_l2.detach().cpu().item()),
    }


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    *,
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    config: DistillConfig,
    model_config: dict[str, int],
    history_tail: list[dict[str, float]],
    samples_seen: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {"distill": asdict(config)},
            "history_tail": history_tail[-10:],
            "seed": config.seed,
            "update": samples_seen,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model_config": model_config,
            "extra": {
                "model_config": model_config,
                "distill": asdict(config),
                "checkpoint_kind": "tactical_policy_distill",
            },
        },
        path,
    )


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    config = DistillConfig(
        samples=int(args.samples),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        epochs_per_batch=int(args.epochs_per_batch),
        max_moves=int(args.max_moves),
        seed=int(args.seed),
        device=device,
        teacher_checkpoint=str(teacher_checkpoint),
        student_init=str(args.student_init.resolve()) if args.student_init is not None else "",
        rollout_bots=tuple(args.rollout_bots),
        include_teacher_self_play=not args.no_teacher_self_play,
        teacher_self_play_prob=float(args.teacher_self_play_prob),
        opening_random_pairs=int(args.opening_random_pairs),
        soft_targets=not args.hard_targets,
        teacher_top_k=int(args.teacher_top_k),
        teacher_temperature=float(args.teacher_temperature),
        scorer_weight=float(args.scorer_weight),
        entropy_coef=float(args.entropy_coef),
        num_workers=int(args.num_workers),
        worker_batch_size=int(args.worker_batch_size),
        worker_device=str(args.worker_device),
        save_every=int(args.save_every),
    )
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if config.num_workers < 1:
        raise ValueError("num-workers must be at least 1")
    if config.worker_batch_size <= 0:
        raise ValueError("worker-batch-size must be positive")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.resolve() / run_name
    history_path = run_dir / "history.jsonl"
    manifest_path = run_dir / "manifest.json"
    checkpoint_path = run_dir / "distilled_model.pt"
    latest_checkpoint_path = run_dir / "latest_model.pt"

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    teacher_payload = torch.load(teacher_checkpoint, map_location=device)
    model_config = model_config_from_checkpoint_payload(teacher_payload)
    teacher_model = ActorCriticNet(model_config).to(device)
    teacher_model.load_state_dict(teacher_payload["model_state_dict"])
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    model = ActorCriticNet(model_config).to(device)
    student_payload = teacher_payload
    if args.student_init is not None:
        student_payload = torch.load(args.student_init.resolve(), map_location=device)
        student_config = model_config_from_checkpoint_payload(student_payload)
        if student_config != model_config:
            raise ValueError(f"student init model config {student_config} does not match teacher config {model_config}")
    model.load_state_dict(student_payload["model_state_dict"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    bots = [create_bot(name=name) for name in config.rollout_bots]
    weights = np.asarray(config.rollout_bot_weights, dtype=np.float64)
    if weights.size != len(bots):
        raise ValueError("rollout-bot-weights length must match rollout-bots length")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("rollout-bot-weights must be non-negative and sum to a positive value")
    weights = weights / weights.sum()

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "config": asdict(config),
        "model_config": model_config.to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history: list[dict[str, float]] = []
    samples_seen = 0
    update = 0
    started = perf_counter()
    executor: ProcessPoolExecutor | None = None
    if config.num_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=config.num_workers,
            initializer=_init_worker,
            initargs=(
                str(teacher_checkpoint),
                asdict(config),
                model_config.to_dict(),
                weights.tolist(),
            ),
        )

    try:
        while samples_seen < config.samples:
            batch_started = perf_counter()
            requested_batch_size = min(config.batch_size, config.samples - samples_seen)
            if executor is not None:
                obs_np, masks_np, targets_np, batch_stats = collect_batch_parallel(
                    executor,
                    batch_size=requested_batch_size,
                    worker_batch_size=config.worker_batch_size,
                    num_workers=config.num_workers,
                    seed_base=config.seed,
                    update=update,
                )
            else:
                obs_np, masks_np, targets_np, batch_stats = collect_batch(
                    model=teacher_model,
                    batch_size=requested_batch_size,
                    max_moves=config.max_moves,
                    bots=bots,
                    bot_weights=weights,
                    rng=rng,
                    device=device,
                    include_teacher_self_play=config.include_teacher_self_play,
                    teacher_self_play_prob=config.teacher_self_play_prob,
                    opening_random_pairs=config.opening_random_pairs,
                    soft_targets=config.soft_targets,
                    teacher_top_k=config.teacher_top_k,
                    teacher_temperature=config.teacher_temperature,
                    scorer_weight=config.scorer_weight,
                )
            epoch_metrics = [
                train_step(model, optimizer, obs_np, masks_np, targets_np, device, config.entropy_coef)
                for _ in range(max(1, config.epochs_per_batch))
            ]
            metrics = average_metrics(epoch_metrics)
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
                "distill update={update:.0f} samples={samples_seen:.0f}/{total} "
                "loss={policy_loss:.4f} top1={target_top1:.3f} entropy={entropy:.3f} "
                "fps={samples_per_sec:.1f}".format(total=config.samples, **row)
            )
            if config.save_every > 0 and update % config.save_every == 0:
                save_checkpoint(
                    latest_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    model_config=model_config.to_dict(),
                    history_tail=history,
                    samples_seen=samples_seen,
                )
                manifest["latest_checkpoint"] = str(latest_checkpoint_path)
                manifest["samples_seen"] = samples_seen
                manifest["latest_metrics"] = history[-1]
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"saved interim checkpoint: {latest_checkpoint_path}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        model_config=model_config.to_dict(),
        history_tail=history,
        samples_seen=samples_seen,
    )
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["samples_seen"] = samples_seen
    manifest["latest_metrics"] = history[-1] if history else None
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
