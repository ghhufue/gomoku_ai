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

from gomoku_ai.env import BLACK, BOARD_SIZE, GomokuEnv, WHITE, action_to_coord
from gomoku_ai.model import ActorCriticNet, model_config_from_checkpoint_payload
from gomoku_ai.rule_bot import RuleBasedBot


PLAYER_LABEL = {
    BLACK: "black",
    WHITE: "white",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Gomoku PPO checkpoint against the rule bot.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--bot", type=str, default="rule", choices=["rule", "random"])
    parser.add_argument("--export-record", action="store_true", default=True, help="Export replay-compatible match records.")
    parser.add_argument("--no-export-record", action="store_false", dest="export_record", help="Disable replay-compatible match record export.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--filename", type=str, default="match_record")
    parser.add_argument("--export-text", action="store_true", default=True, help="Export plain-text move report.")
    parser.add_argument("--no-export-text", action="store_false", dest="export_text", help="Disable plain-text move report export.")
    parser.add_argument("--export-visual", action="store_true", default=True, help="Export visual JSON payload.")
    parser.add_argument("--no-export-visual", action="store_false", dest="export_visual", help="Disable visual JSON export.")
    parser.add_argument("--quiet-games", action="store_true", help="Suppress per-game logs.")
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def choose_action(model: ActorCriticNet, obs: np.ndarray, mask: np.ndarray, device: str) -> int:
    obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask[None, ...], dtype=torch.bool, device=device)
    with torch.no_grad():
        dist, _ = model.masked_distribution(obs_tensor, mask_tensor)
        action = torch.argmax(dist.logits, dim=-1)
    return int(action.item())


class RandomBot:
    def select_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        empties = np.flatnonzero(board.reshape(-1) == 0)
        return int(rng.choice(empties))


def build_opponent(bot_name: str):
    if bot_name == "rule":
        return RuleBasedBot()
    if bot_name == "random":
        return RandomBot()
    raise ValueError(f"unsupported bot: {bot_name}")


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
    bot_name: str = "rule",
    verbose: bool = True,
    include_record: bool = False,
) -> dict[str, object]:
    env = GomokuEnv(opponent=build_opponent(bot_name), seed=seed)
    obs, mask = env.reset()

    moves: list[dict[str, object]] = []
    agent_player = env.agent_player
    bot_player = -agent_player

    if include_record:
        opening_action = detect_opening_action(env.board, bot_player)
        if opening_action is not None:
            record_move(moves, bot_player, opening_action, "bot")

    done = False
    result = None
    steps = 0
    total_reward = 0.0
    while not done:
        action = choose_action(model, obs, mask, device)
        result = env.step(action)
        if include_record:
            record_move(moves, agent_player, action, "model")
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
        "description": f"model vs {bot_name} bot match exported by scripts/evaluate.py",
        "seed": seed,
        "agent_player": PLAYER_LABEL[agent_player],
        "bot_player": PLAYER_LABEL[bot_player],
        "bot_name": bot_name,
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
    bot_name: str = "rule",
    verbose: bool = True,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0
    episode_lengths: list[int] = []

    for game_idx in range(games):
        game = play_game(model=model, device=device, seed=seed + game_idx, bot_name=bot_name, verbose=verbose, include_record=False)
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
    bot_name: str = "rule",
    verbose: bool = True,
) -> dict[str, float]:
    runs: list[dict[str, float]] = []
    for seed in seeds:
        metrics = evaluate_model(model=model, games=games, device=device, seed=seed, bot_name=bot_name, verbose=verbose)
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
        "bot_name": bot_name,
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
    checkpoint_path: Path,
    device: str,
    games: list[dict[str, object]],
    output_dir: Path,
    filename: str,
    export_text: bool,
    export_visual: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    output_path = output_dir / f"{filename}.json"
    payload = build_match_payload(checkpoint_path, device, games)
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    print(f"using device: {device}")
    checkpoint_path, warning = resolve_checkpoint(args.checkpoint)
    if warning is not None:
        print(f"warning: {warning}")
    model = load_model_from_checkpoint(checkpoint_path, device)
    verbose = not args.quiet_games
    seeds = [args.seed + idx for idx in range(args.num_seeds)]
    metrics = evaluate_across_seeds(
        model=model,
        games=args.games,
        device=device,
        seeds=seeds,
        bot_name=args.bot,
        verbose=verbose,
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
                        verbose=False,
                        include_record=True,
                    )
                )
        export_dir = build_export_session_dir(args.output_dir.resolve(), checkpoint_path)
        output_paths = export_match_outputs(
            checkpoint_path=checkpoint_path,
            device=device,
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
