from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import (
    BLACK,
    BOARD_SIZE,
    GomokuEnv,
    WHITE,
    action_to_coord,
    call_bot_action,
)
from gomoku_ai.model import ActorCriticNet, model_config_from_checkpoint_payload
from gomoku_ai.opening import apply_random_opening_pairs_to_env, move_coord_payload
from gomoku_ai.tactical_policy import select_tactical_search_action
from bots import available_bots, available_difficulties, create_bot


PLAYER_LABEL = {
    BLACK: "black",
    WHITE: "white",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Gomoku PPO checkpoint against a configured bot, or run bot vs bot.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--bot", type=str, default="reward_driven_hard", choices=available_bots() + ["rule", "reward-driven"])
    parser.add_argument("--bot-difficulty", type=str, default=None, choices=available_difficulties())
    parser.add_argument("--bot-a", type=str, default=None, choices=available_bots() + ["rule", "reward-driven"])
    parser.add_argument("--bot-a-difficulty", type=str, default=None, choices=available_difficulties())
    parser.add_argument("--bot-b", type=str, default=None, choices=available_bots() + ["rule", "reward-driven"])
    parser.add_argument("--bot-b-difficulty", type=str, default=None, choices=available_difficulties())
    parser.add_argument("--export-record", action="store_true", default=True, help="Export replay-compatible match records.")
    parser.add_argument("--no-export-record", action="store_false", dest="export_record", help="Disable replay-compatible match record export.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--filename", type=str, default="match_record")
    parser.add_argument("--export-text", action="store_true", default=True, help="Export plain-text move report.")
    parser.add_argument("--no-export-text", action="store_false", dest="export_text", help="Disable plain-text move report export.")
    parser.add_argument("--export-visual", action="store_true", default=True, help="Export visual JSON payload.")
    parser.add_argument("--no-export-visual", action="store_false", dest="export_visual", help="Disable visual JSON export.")
    parser.add_argument("--quiet-games", action="store_true", help="Suppress per-game logs.")
    parser.add_argument(
        "--opening-random-pairs",
        type=int,
        default=0,
        help="Add this many random historical move pairs before each model game.",
    )
    parser.add_argument(
        "--no-tactical-guard",
        action="store_true",
        help="Disable tactical search guard for model move selection.",
    )
    # Strength evaluation mode
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "strength"],
                        help="Evaluation mode: standard (single bot) or strength (multi-bot rating).")
    parser.add_argument("--strength-bots", type=str, nargs="*",
                        default=["random", "classic_rule", "reward_driven_medium", "reward_driven_hard"],
                        help="Bots to evaluate against in strength mode.")
    parser.add_argument("--strength-games-per-bot", type=int, nargs="*",
                        default=[20, 20, 20, 20],
                        help="Number of games per bot in strength mode.")
    parser.add_argument("--strength-weights", type=float, nargs="*",
                        default=[1.0, 2.0, 4.0, 8.0],
                        help="Difficulty weights for each bot in strength mode.")
    parser.add_argument("--strength-seed", type=int, default=12345,
                        help="Base seed for strength evaluation.")
    parser.add_argument("--strength-output-json", type=Path, default=None,
                        help="Path to append strength evaluation JSONL result.")
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def choose_action(
    model: ActorCriticNet,
    obs: np.ndarray,
    mask: np.ndarray,
    device: str,
    board: np.ndarray | None = None,
    current_player: int | None = None,
    tactical_guard: bool = True,
) -> tuple[int, str]:
    obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask[None, ...], dtype=torch.bool, device=device)
    with torch.no_grad():
        dist, _ = model.masked_distribution(obs_tensor, mask_tensor)
        logits = dist.logits.squeeze(0).detach().cpu().numpy()

    if tactical_guard and board is not None and current_player is not None:
        action, source, _ = select_tactical_search_action(board, current_player, logits)
        return action, source

    return int(np.argmax(logits)), "model"


def build_opponent(bot_name: str, bot_difficulty: str | None = None):
    return create_bot(name=bot_name, difficulty=bot_difficulty)


def bot_label(bot_name: str | None, bot_difficulty: str | None = None) -> str:
    return bot_difficulty or str(bot_name)


def find_latest_checkpoint() -> Path:
    runs_dir = ROOT / "runs"
    candidates = [path for path in runs_dir.glob("*/final_model.pt") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no final_model.pt found under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint(checkpoint: Path | None) -> tuple[Path, str | None]:
    warning: str | None = None
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        if resolved.is_file():
            return resolved, None
        warning = f"checkpoint not found, fallback to latest final_model: {resolved}"
    return find_latest_checkpoint(), warning


def sanitize_path_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "artifact"


def derive_model_label(checkpoint_path: Path) -> str:
    resolved = checkpoint_path.resolve()
    stem = sanitize_path_fragment(resolved.stem)
    if resolved.parent.name == "checkpoints":
        run_name = sanitize_path_fragment(resolved.parent.parent.name)
        return f"{run_name}_{stem}"
    run_name = sanitize_path_fragment(resolved.parent.name)
    return f"{run_name}_{stem}"


def build_export_session_dir(output_root: Path, checkpoint_path: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    model_label = derive_model_label(checkpoint_path)
    return output_root / f"{timestamp}_{model_label}"


def build_export_session_dir_for_label(output_root: Path, label: str, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_{sanitize_path_fragment(label)}"


def board_to_ascii(board: np.ndarray) -> str:
    header = "   " + " ".join(f"{col:02d}" for col in range(BOARD_SIZE))
    rows = [header]
    for row in range(BOARD_SIZE):
        chars = []
        for col in range(BOARD_SIZE):
            value = int(board[row, col])
            if value == BLACK:
                chars.append("X")
            elif value == WHITE:
                chars.append("O")
            else:
                chars.append(".")
        rows.append(f"{row:02d} " + " ".join(chars))
    return "\n".join(rows)


def game_sequence_signature(game: dict[str, object], limit: int | None = None) -> tuple[tuple[object, ...], ...]:
    moves = game.get("moves", [])
    if limit is not None:
        moves = moves[:limit]
    return tuple(
        (
            move.get("player"),
            move.get("row"),
            move.get("col"),
            move.get("source"),
        )
        for move in moves
    )


def sequence_diversity(games: list[dict[str, object]]) -> dict[str, int]:
    if not games or not all("moves" in game for game in games):
        return {}
    return {
        "games": len(games),
        "unique_full_sequences": len({game_sequence_signature(game) for game in games}),
        "unique_first_6_plies": len({game_sequence_signature(game, 6) for game in games}),
        "unique_first_10_plies": len({game_sequence_signature(game, 10) for game in games}),
    }


def record_move(moves: list[dict[str, object]], player: int, action: int, source: str) -> None:
    row, col = action_to_coord(action)
    player_label = PLAYER_LABEL[player]
    moves.append(
        {
            "move_number": len(moves) + 1,
            "player": player_label,
            "color": player_label,
            "source": source,
            "action": int(action),
            "row": int(row),
            "col": int(col),
        }
    )


def record_opening_moves(moves: list[dict[str, object]], actions: list[int], first_player: int) -> None:
    player = first_player
    for action in actions:
        payload = move_coord_payload(action, player, "random_opening")
        payload["move_number"] = len(moves) + 1
        moves.append(payload)
        player = -player


def detect_opening_action(board: np.ndarray, player: int) -> int | None:
    positions = np.argwhere(board == player)
    if len(positions) != 1:
        return None
    row, col = positions[0]
    return int(row) * BOARD_SIZE + int(col)


def play_game(
    model: ActorCriticNet,
    device: str,
    seed: int,
    bot_name: str = "reward_driven_hard",
    bot_difficulty: str | None = None,
    verbose: bool = True,
    include_record: bool = False,
    tactical_guard: bool = True,
    opening_random_pairs: int = 0,
) -> dict[str, object]:
    env = GomokuEnv(opponent=build_opponent(bot_name, bot_difficulty), seed=seed)
    obs, mask = env.reset()

    moves: list[dict[str, object]] = []
    agent_player = env.agent_player
    bot_player = -agent_player

    if include_record:
        opening_action = detect_opening_action(env.board, bot_player)
        if opening_action is not None:
            record_move(moves, bot_player, opening_action, "bot")

    opening_actions = apply_random_opening_pairs_to_env(env, opening_random_pairs, env.rng)
    if opening_actions:
        if include_record:
            record_opening_moves(moves, opening_actions, agent_player)
        obs = env.observation()
        mask = env.action_mask()

    done = False
    result = None
    steps = 0
    total_reward = 0.0
    tactical_guard_uses = 0
    while not done:
        action, action_source = choose_action(
            model,
            obs,
            mask,
            device,
            board=env.board,
            current_player=agent_player,
            tactical_guard=tactical_guard,
        )
        if action_source != "model":
            tactical_guard_uses += 1
        result = env.step(action)
        if include_record:
            record_move(moves, agent_player, action, action_source)
            opponent_action = result.info.get("opponent_action")
            if opponent_action is not None:
                record_move(moves, bot_player, int(opponent_action), "bot")

        obs = result.observation
        mask = result.action_mask
        done = result.done
        total_reward += result.reward
        steps += 1

    assert result is not None
    outcome = result.info.get("agent_result")
    if verbose:
        print(f"game seed={seed} result={outcome} steps={steps} total_reward={total_reward:.2f}")

    game = {
        "game_index": 1,
        "name": f"game_seed_{seed}",
        "description": f"model vs {bot_difficulty or bot_name} bot match exported by scripts/evaluate.py",
        "seed": seed,
        "agent_player": PLAYER_LABEL[agent_player],
        "bot_player": PLAYER_LABEL[bot_player],
        "bot_name": bot_difficulty or bot_name,
        "result": outcome,
        "illegal_move": bool(result.info.get("illegal_move", False)),
        "winner": PLAYER_LABEL[result.info["winner"]] if "winner" in result.info else None,
        "total_reward": float(total_reward),
        "num_moves": len(moves) if include_record else int(steps),
        "tactical_guard_uses": tactical_guard_uses,
        "opening_random_pairs": int(opening_random_pairs),
    }
    if include_record:
        game["moves"] = moves
        game["final_board_ascii"] = board_to_ascii(env.board)
    return game


def play_bot_game(
    seed: int,
    bot_a_name: str,
    bot_b_name: str,
    bot_a_difficulty: str | None = None,
    bot_b_difficulty: str | None = None,
    verbose: bool = True,
    include_record: bool = False,
) -> dict[str, object]:
    bot_b = build_opponent(bot_b_name, bot_b_difficulty)
    env = GomokuEnv(opponent=bot_b, seed=seed)
    bot_a = build_opponent(bot_a_name, bot_a_difficulty)
    obs, mask = env.reset()

    moves: list[dict[str, object]] = []
    bot_a_player = env.agent_player
    bot_b_player = -bot_a_player
    bot_a_label = bot_label(bot_a_name, bot_a_difficulty)
    bot_b_label = bot_label(bot_b_name, bot_b_difficulty)

    if include_record:
        opening_action = detect_opening_action(env.board, bot_b_player)
        if opening_action is not None:
            record_move(moves, bot_b_player, opening_action, "bot_b")

    done = False
    result = None
    steps = 0
    total_reward = 0.0
    rng = env.rng
    while not done:
        action = call_bot_action(bot_a, env.board.copy(), bot_a_player, rng)
        result = env.step(action)
        if include_record:
            record_move(moves, bot_a_player, action, "bot_a")
            opponent_action = result.info.get("opponent_action")
            if opponent_action is not None:
                record_move(moves, bot_b_player, int(opponent_action), "bot_b")

        obs = result.observation
        mask = result.action_mask
        done = result.done
        total_reward += result.reward
        steps += 1

    assert result is not None
    outcome = result.info.get("agent_result")
    if verbose:
        print(f"game seed={seed} result={outcome} steps={steps} total_reward={total_reward:.2f}")

    game = {
        "game_index": 1,
        "name": f"game_seed_{seed}",
        "description": f"{bot_a_label} vs {bot_b_label} bot match exported by scripts/evaluate.py",
        "seed": seed,
        "agent_player": PLAYER_LABEL[bot_a_player],
        "bot_player": PLAYER_LABEL[bot_b_player],
        "bot_name": bot_b_label,
        "agent_name": bot_a_label,
        "mode": "bot_vs_bot",
        "result": outcome,
        "illegal_move": bool(result.info.get("illegal_move", False)),
        "winner": PLAYER_LABEL[result.info["winner"]] if "winner" in result.info else None,
        "total_reward": float(total_reward),
        "num_moves": len(moves) if include_record else int(steps),
    }
    if include_record:
        game["moves"] = moves
        game["final_board_ascii"] = board_to_ascii(env.board)
    return game


def evaluate_model(
    model: ActorCriticNet,
    games: int,
    device: str,
    seed: int,
    bot_name: str = "reward_driven_hard",
    bot_difficulty: str | None = None,
    verbose: bool = True,
    tactical_guard: bool = True,
    opening_random_pairs: int = 0,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0
    episode_lengths: list[int] = []

    for game_idx in range(games):
        game = play_game(
            model=model,
            device=device,
            seed=seed + game_idx,
            bot_name=bot_name,
            bot_difficulty=bot_difficulty,
            verbose=verbose,
            include_record=False,
            tactical_guard=tactical_guard,
            opening_random_pairs=opening_random_pairs,
        )
        outcome = game["result"]
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            draws += 1

        if game["illegal_move"]:
            illegal_moves += 1

        episode_lengths.append(int(game["num_moves"]))

    metrics = {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "illegal_moves": float(illegal_moves),
        "win_rate": wins / games,
        "avg_steps": sum(episode_lengths) / len(episode_lengths),
    }
    if was_training:
        model.train()
    return metrics


def evaluate_across_seeds(
    model: ActorCriticNet,
    games: int,
    device: str,
    seeds: list[int],
    bot_name: str = "reward_driven_hard",
    bot_difficulty: str | None = None,
    verbose: bool = True,
    tactical_guard: bool = True,
    opening_random_pairs: int = 0,
) -> dict[str, float]:
    runs: list[dict[str, float]] = []
    for seed in seeds:
        metrics = evaluate_model(
            model=model,
            games=games,
            device=device,
            seed=seed,
            bot_name=bot_name,
            bot_difficulty=bot_difficulty,
            verbose=verbose,
            tactical_guard=tactical_guard,
            opening_random_pairs=opening_random_pairs,
        )
        metrics["seed"] = float(seed)
        runs.append(metrics)
        if verbose:
            print(
                "seed={seed} win_rate={win_rate:.3f} wins={wins:.0f} losses={losses:.0f} "
                "draws={draws:.0f} avg_steps={avg_steps:.2f}".format(**metrics)
            )

    return {
        "games_per_seed": float(games),
        "num_seeds": float(len(seeds)),
        "bot_name": bot_difficulty or bot_name,
        "total_games": float(games * len(seeds)),
        "wins": float(sum(run["wins"] for run in runs)),
        "losses": float(sum(run["losses"] for run in runs)),
        "draws": float(sum(run["draws"] for run in runs)),
        "illegal_moves": float(sum(run["illegal_moves"] for run in runs)),
        "win_rate": float(np.mean([run["win_rate"] for run in runs])),
        "avg_steps": float(np.mean([run["avg_steps"] for run in runs])),
        "win_rate_std": float(np.std([run["win_rate"] for run in runs])),
        "runs": runs,
    }


def build_match_payload(checkpoint_path: Path, device: str, games: list[dict[str, object]]) -> dict[str, object]:
    payload = {
        "schema_version": 2,
        "record_type": "gomoku_match_record",
        "checkpoint": str(checkpoint_path),
        "device": device,
        "bot_name": games[0].get("bot_name") if games else None,
        "sequence_diversity": sequence_diversity(games),
        "games": games,
    }
    for index, game in enumerate(payload["games"], start=1):
        game["game_index"] = index
    return payload


def build_bot_match_payload(bot_a_name: str, bot_b_name: str, games: list[dict[str, object]]) -> dict[str, object]:
    payload = {
        "schema_version": 2,
        "record_type": "gomoku_match_record",
        "mode": "bot_vs_bot",
        "bot_a_name": bot_a_name,
        "bot_b_name": bot_b_name,
        "games": games,
    }
    for index, game in enumerate(payload["games"], start=1):
        game["game_index"] = index
    return payload


def render_text_report(games: list[dict[str, object]]) -> str:
    sections: list[str] = []
    for idx, game in enumerate(games, start=1):
        sections.append(
            "game={game} seed={seed} model={agent} bot={bot} result={result} illegal_move={illegal}".format(
                game=idx,
                seed=game["seed"],
                agent=game["agent_player"],
                bot=game["bot_player"],
                result=game["result"],
                illegal=game["illegal_move"],
            )
        )
        for move in game.get("moves", []):
            sections.append(
                "  move {move_number:02d}: {player} ({source}) -> ({row}, {col}) action={action}".format(**move)
            )
        if "final_board_ascii" in game:
            sections.append("  final board:")
            sections.append(str(game["final_board_ascii"]))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def build_visual_payload(game: dict[str, object]) -> dict[str, object]:
    moves = [
        {
            "row": move["row"],
            "col": move["col"],
            "color": move["player"],
        }
        for move in game.get("moves", [])
    ]
    return {
        "moves": moves,
        "analysis": {
            "winRate": {},
            "evaluations": [],
            "suggestions": [],
        },
    }


def export_match_outputs(
    payload: dict[str, object],
    games: list[dict[str, object]],
    output_dir: Path,
    filename: str,
    export_text: bool,
    export_visual: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    output_path = output_dir / f"{filename}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_paths.append(output_path)

    if export_text:
        text_path = output_dir / f"{filename}.txt"
        text_path.write_text(render_text_report(games), encoding="utf-8")
        output_paths.append(text_path)

    if export_visual:
        if len(games) == 1:
            visual_path = output_dir / f"{filename}_visual.json"
            visual_path.write_text(json.dumps(build_visual_payload(games[0]), indent=2), encoding="utf-8")
            output_paths.append(visual_path)
        else:
            for idx, game in enumerate(games, start=1):
                visual_path = output_dir / f"{filename}_game{idx:02d}_visual.json"
                visual_path.write_text(json.dumps(build_visual_payload(game), indent=2), encoding="utf-8")
                output_paths.append(visual_path)

    return output_paths


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> ActorCriticNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = model_config_from_checkpoint_payload(checkpoint)
    model = ActorCriticNet(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def print_summary(metrics: dict[str, float]) -> None:
    print("--- evaluation summary ---")
    print(f"num_seeds={int(metrics['num_seeds'])}")
    print(f"games_per_seed={int(metrics['games_per_seed'])}")
    print(f"total_games={int(metrics['total_games'])}")
    print(f"wins={int(metrics['wins'])}")
    print(f"losses={int(metrics['losses'])}")
    print(f"draws={int(metrics['draws'])}")
    print(f"illegal_moves={int(metrics['illegal_moves'])}")
    print(f"win_rate={metrics['win_rate']:.3f}")
    print(f"win_rate_std={metrics['win_rate_std']:.3f}")
    print(f"avg_steps={metrics['avg_steps']:.2f}")


def evaluate_strength(
    model: ActorCriticNet,
    device: str,
    checkpoint_path: Path,
    bots: list[str],
    games_per_bot: list[int],
    difficulty_weights: list[float],
    seed: int,
    verbose: bool = True,
    tactical_guard: bool = True,
    opening_random_pairs: int = 0,
) -> dict:
    """Evaluate model strength against a pool of bots.
    
    Returns a dict with per-bot results, raw_strength_score, 
    and normalized_strength_score_0_100.
    """
    if len(bots) != len(games_per_bot) or len(bots) != len(difficulty_weights):
        raise ValueError(
            f"bots ({len(bots)}), games_per_bot ({len(games_per_bot)}), "
            f"difficulty_weights ({len(difficulty_weights)}) must have same length"
        )

    was_training = model.training
    model.eval()
    total_games = sum(games_per_bot)
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_illegal = 0
    total_steps = 0
    
    bot_results: list[dict] = []
    raw_strength_score = 0.0
    total_weight = sum(difficulty_weights)
    
    current_seed = seed
    
    for bot_name, games, weight in zip(bots, games_per_bot, difficulty_weights):
        bot_wins = 0
        bot_losses = 0
        bot_draws = 0
        bot_illegal = 0
        bot_steps = 0
        
        for game_idx in range(games):
            game = play_game(
                model=model,
                device=device,
                seed=current_seed + game_idx,
                bot_name=bot_name,
                verbose=False,
                include_record=False,
                tactical_guard=tactical_guard,
                opening_random_pairs=opening_random_pairs,
            )
            outcome = game["result"]
            if outcome == "win":
                bot_wins += 1
            elif outcome == "loss":
                bot_losses += 1
            else:
                bot_draws += 1
            if game.get("illegal_move"):
                bot_illegal += 1
            bot_steps += int(game["num_moves"])
        
        current_seed += games
        g = float(games)
        win_rate = bot_wins / g if g > 0 else 0.0
        draw_rate = bot_draws / g if g > 0 else 0.0
        loss_rate = bot_losses / g if g > 0 else 0.0
        result_rate = win_rate + 0.5 * draw_rate
        contribution = result_rate * weight
        raw_strength_score += contribution
        
        bot_result = {
            "bot": bot_name,
            "games": int(g),
            "wins": bot_wins,
            "draws": bot_draws,
            "losses": bot_losses,
            "illegal_moves": bot_illegal,
            "win_rate": round(win_rate, 4),
            "draw_rate": round(draw_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "result_rate": round(result_rate, 4),
            "difficulty_weight": weight,
            "contribution_score": round(contribution, 4),
            "avg_steps": round(bot_steps / max(1, g), 2),
        }
        bot_results.append(bot_result)
        
        total_wins += bot_wins
        total_losses += bot_losses
        total_draws += bot_draws
        total_illegal += bot_illegal
        total_steps += bot_steps
        
        if verbose:
            print(
                f"  bot={bot_name:25s} games={int(g):3d} win={win_rate:.3f} "
                f"draw={draw_rate:.3f} loss={loss_rate:.3f} "
                f"result_rate={result_rate:.3f} weight={weight:.1f} "
                f"contrib={contribution:.3f}"
            )
    
    normalized_score = 100.0 * raw_strength_score / total_weight if total_weight > 0 else 0.0
    
    result = {
        "checkpoint": str(checkpoint_path),
        "total_games": total_games,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_draws": total_draws,
        "total_illegal_moves": total_illegal,
        "overall_win_rate": round(total_wins / max(1, total_games), 4),
        "overall_draw_rate": round(total_draws / max(1, total_games), 4),
        "overall_loss_rate": round(total_losses / max(1, total_games), 4),
        "avg_steps": round(total_steps / max(1, total_games), 2),
        "bot_results": bot_results,
        "raw_strength_score": round(raw_strength_score, 4),
        "normalized_strength_score_0_100": round(normalized_score, 2),
        "seed": seed,
        "difficulty_weights": difficulty_weights,
    }
    
    if was_training:
        model.train()
    return result


def print_strength_report(result: dict) -> None:
    """Print a formatted strength evaluation report."""
    print()
    print("=" * 70)
    print("  STRENGTH EVALUATION REPORT")
    print("=" * 70)
    print(f"  Checkpoint: {result['checkpoint']}")
    print(f"  Total games: {result['total_games']}")
    print(f"  Overall: W={result['total_wins']} D={result['total_draws']} "
          f"L={result['total_losses']} "
          f"win_rate={result['overall_win_rate']:.3f} "
          f"illegal={result['total_illegal_moves']}")
    print("-" * 70)
    print(f"  {'Bot':<25s} {'G':>4s} {'Win%':>7s} {'Draw%':>7s} {'Loss%':>7s} "
          f"{'Rate':>7s} {'Wgt':>5s} {'Contrib':>8s}")
    print("-" * 70)
    for br in result["bot_results"]:
        print(
            f"  {br['bot']:<25s} {br['games']:>4d} {br['win_rate']:>7.3f} "
            f"{br['draw_rate']:>7.3f} {br['loss_rate']:>7.3f} "
            f"{br['result_rate']:>7.3f} {br['difficulty_weight']:>5.1f} "
            f"{br['contribution_score']:>8.3f}"
        )
    print("-" * 70)
    print(f"  Raw Strength Score:      {result['raw_strength_score']:.4f}")
    print(f"  Normalized (0-100):      {result['normalized_strength_score_0_100']:.2f}")
    print("=" * 70)
    print()


def is_bot_vs_bot_mode(args: argparse.Namespace) -> bool:
    return args.checkpoint is None and args.bot_a is not None and args.bot_b is not None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet_games
    seeds = [args.seed + idx for idx in range(args.num_seeds)]
    default_games = build_parser().get_default("games")

    if is_bot_vs_bot_mode(args):
        games_to_play = 1 if args.games == default_games else args.games
        bot_games: list[dict[str, object]] = []
        wins = 0
        losses = 0
        draws = 0
        illegal_moves = 0
        steps: list[int] = []
        for seed in seeds:
            for game_idx in range(games_to_play):
                game = play_bot_game(
                    seed=seed + game_idx,
                    bot_a_name=args.bot_a,
                    bot_b_name=args.bot_b,
                    bot_a_difficulty=args.bot_a_difficulty,
                    bot_b_difficulty=args.bot_b_difficulty,
                    verbose=verbose,
                    include_record=args.export_record,
                )
                bot_games.append(game)
                if game["result"] == "win":
                    wins += 1
                elif game["result"] == "loss":
                    losses += 1
                else:
                    draws += 1
                if game["illegal_move"]:
                    illegal_moves += 1
                steps.append(int(game["num_moves"]))
        summary = {
            "num_seeds": float(len(seeds)),
            "games_per_seed": float(games_to_play),
            "total_games": float(len(bot_games)),
            "wins": float(wins),
            "losses": float(losses),
            "draws": float(draws),
            "illegal_moves": float(illegal_moves),
            "win_rate": wins / len(bot_games) if bot_games else 0.0,
            "win_rate_std": 0.0,
            "avg_steps": sum(steps) / len(steps) if steps else 0.0,
        }
        print("mode=bot_vs_bot")
        print(f"bot_a={bot_label(args.bot_a, args.bot_a_difficulty)}")
        print(f"bot_b={bot_label(args.bot_b, args.bot_b_difficulty)}")
        print_summary(summary)
        if args.export_record:
            export_dir = build_export_session_dir_for_label(
                args.output_dir.resolve(),
                f"{bot_label(args.bot_a, args.bot_a_difficulty)}_vs_{bot_label(args.bot_b, args.bot_b_difficulty)}",
            )
            output_paths = export_match_outputs(
                payload=build_bot_match_payload(
                    bot_label(args.bot_a, args.bot_a_difficulty),
                    bot_label(args.bot_b, args.bot_b_difficulty),
                    bot_games,
                ),
                games=bot_games,
                output_dir=export_dir,
                filename=Path(args.filename).stem or "match_record",
                export_text=args.export_text,
                export_visual=args.export_visual,
            )
            for path in output_paths:
                print(f"saved: {path}")
        return

    device = resolve_device(args.device)
    print(f"using device: {device}")
    checkpoint_path, warning = resolve_checkpoint(args.checkpoint)
    if warning is not None:
        print(f"warning: {warning}")
    
    if args.mode == "strength":
        # Strength evaluation mode
        print("mode=strength")
        model = load_model_from_checkpoint(checkpoint_path, device)
        bots = args.strength_bots
        games_per_bot = args.strength_games_per_bot
        weights = args.strength_weights
        # Pad/truncate to match bot count
        if len(games_per_bot) < len(bots):
            games_per_bot = games_per_bot + [20] * (len(bots) - len(games_per_bot))
        if len(weights) < len(bots):
            weights = weights + [1.0] * (len(bots) - len(weights))
        games_per_bot = games_per_bot[:len(bots)]
        weights = weights[:len(bots)]
        
        result = evaluate_strength(
            model=model,
            device=device,
            checkpoint_path=checkpoint_path,
            bots=bots,
            games_per_bot=games_per_bot,
            difficulty_weights=weights,
            seed=args.strength_seed,
            verbose=True,
            tactical_guard=not args.no_tactical_guard,
            opening_random_pairs=args.opening_random_pairs,
        )
        print_strength_report(result)
        
        if args.strength_output_json is not None:
            output_path = Path(args.strength_output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
            print(f"strength result appended to: {output_path}")
        
        if args.export_record:
            export_dir = build_export_session_dir(args.output_dir.resolve(), checkpoint_path)
            # Also save strength result to export dir
            strength_path = export_dir / "strength_eval.json"
            strength_path.parent.mkdir(parents=True, exist_ok=True)
            strength_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"saved: {strength_path}")
        return

    model = load_model_from_checkpoint(checkpoint_path, device)
    metrics = evaluate_across_seeds(
        model=model,
        games=args.games,
        device=device,
        seeds=seeds,
        bot_name=args.bot,
        bot_difficulty=args.bot_difficulty,
        verbose=verbose,
        tactical_guard=not args.no_tactical_guard,
        opening_random_pairs=args.opening_random_pairs,
    )
    print_summary(metrics)

    if args.export_record:
        games: list[dict[str, object]] = []
        for seed in seeds:
            for game_idx in range(args.games):
                games.append(
                    play_game(
                        model=model,
                        device=device,
                        seed=seed + game_idx,
                        bot_name=args.bot,
                        bot_difficulty=args.bot_difficulty,
                        verbose=False,
                        include_record=True,
                        tactical_guard=not args.no_tactical_guard,
                        opening_random_pairs=args.opening_random_pairs,
                    )
                )
        export_dir = build_export_session_dir(args.output_dir.resolve(), checkpoint_path)
        output_paths = export_match_outputs(
            payload=build_match_payload(checkpoint_path, device, games),
            games=games,
            output_dir=export_dir,
            filename=Path(args.filename).stem or "match_record",
            export_text=args.export_text,
            export_visual=args.export_visual,
        )
        for path in output_paths:
            print(f"saved: {path}")


if __name__ == "__main__":
    main()
