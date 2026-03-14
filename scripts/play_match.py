from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import BLACK, BOARD_SIZE, EMPTY, WHITE, GomokuEnv, action_to_coord
from gomoku_ai.rule_bot import RuleBasedBot
from scripts.evaluate import choose_action, load_model_from_checkpoint, resolve_device


PLAYER_LABEL = {
    BLACK: "black",
    WHITE: "white",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play model-vs-bot matches and export move records.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--filename", type=str, default="match_record")
    parser.add_argument("--export-visual", type=str, default="true")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    return parser


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
    moves.append(
        {
            "move_number": len(moves) + 1,
            "player": PLAYER_LABEL[player],
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


def play_game(model, device: str, seed: int) -> dict[str, object]:
    env = GomokuEnv(opponent=RuleBasedBot(), seed=seed)
    obs, mask = env.reset()

    moves: list[dict[str, object]] = []
    agent_player = env.agent_player
    bot_player = -agent_player

    opening_action = detect_opening_action(env.board, bot_player)
    if opening_action is not None:
        record_move(moves, bot_player, opening_action, "bot")

    done = False
    result = None
    while not done:
        action = choose_action(model, obs, mask, device)
        result = env.step(action)
        record_move(moves, agent_player, action, "model")

        opponent_action = result.info.get("opponent_action")
        if opponent_action is not None:
            record_move(moves, bot_player, int(opponent_action), "bot")

        obs = result.observation
        mask = result.action_mask
        done = result.done

    assert result is not None
    return {
        "seed": seed,
        "agent_player": PLAYER_LABEL[agent_player],
        "bot_player": PLAYER_LABEL[bot_player],
        "result": result.info.get("agent_result"),
        "illegal_move": bool(result.info.get("illegal_move", False)),
        "winner": PLAYER_LABEL[result.info["winner"]] if "winner" in result.info else None,
        "total_reward": float(result.reward),
        "num_moves": len(moves),
        "moves": moves,
        "final_board_ascii": board_to_ascii(env.board),
    }


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
        for move in game["moves"]:
            sections.append(
                "  move {move_number:02d}: {player} ({source}) -> ({row}, {col}) action={action}".format(**move)
            )
        sections.append("  final board:")
        sections.append(str(game["final_board_ascii"]))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def build_visual_payload(game: dict[str, object]) -> dict[str, object]:
    moves = [
        {
            "row": move["row"],
            "col": move["col"],
            "color": move["player"],
        }
        for move in game["moves"]
    ]
    return {
        "moves": moves,
        "analysis": {
            "winRate": {},
            "evaluations": [],
            "suggestions": [],
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    export_visual = parse_bool(args.export_visual)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(args.filename).stem or "match_record"
    output_path = output_dir / f"{filename}.json"
    checkpoint_path, warning = resolve_checkpoint(args.checkpoint)
    model = load_model_from_checkpoint(checkpoint_path, device)

    games = [play_game(model, device, args.seed + idx) for idx in range(args.games)]
    payload = {
        "checkpoint": str(checkpoint_path),
        "device": device,
        "games": games,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    text_path = output_dir / f"{filename}.txt"
    text_path.write_text(render_text_report(games), encoding="utf-8")
    visual_paths: list[Path] = []
    if export_visual:
        if len(games) == 1:
            visual_path = output_dir / f"{filename}_visual.json"
            visual_path.write_text(json.dumps(build_visual_payload(games[0]), indent=2), encoding="utf-8")
            visual_paths.append(visual_path)
        else:
            for idx, game in enumerate(games, start=1):
                visual_path = output_dir / f"{filename}_game{idx:02d}_visual.json"
                visual_path.write_text(json.dumps(build_visual_payload(game), indent=2), encoding="utf-8")
                visual_paths.append(visual_path)

    if warning is not None:
        print(f"warning: {warning}")
    print(f"saved json: {output_path}")
    print(f"saved text: {text_path}")
    for visual_path in visual_paths:
        print(f"saved visual json: {visual_path}")
    print(f"checkpoint: {checkpoint_path}")
    for idx, game in enumerate(games, start=1):
        print(
            "game={idx} seed={seed} result={result} moves={moves} model={agent} bot={bot}".format(
                idx=idx,
                seed=game["seed"],
                result=game["result"],
                moves=game["num_moves"],
                agent=game["agent_player"],
                bot=game["bot_player"],
            )
        )


if __name__ == "__main__":
    main()
