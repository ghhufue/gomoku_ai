from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bots import available_bots, create_bot
from gomoku_ai.env import EnvConfig, GomokuEnv
from gomoku_ai.model import ActorCriticNet, model_config_from_checkpoint_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a workshop Gomoku checkpoint against one or more bots.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bots", nargs="+", default=["random", "classic_rule"], choices=available_bots())
    parser.add_argument("--output", type=Path, default=None)
    return parser


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_model(checkpoint_path: Path, device: str) -> ActorCriticNet:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ActorCriticNet(model_config_from_checkpoint_payload(payload)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def choose_action(model: ActorCriticNet, obs: np.ndarray, mask: np.ndarray, device: str) -> int:
    obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask[None, ...], dtype=torch.bool, device=device)
    with torch.no_grad():
        dist, _ = model.masked_distribution(obs_tensor, mask_tensor)
    return int(torch.argmax(dist.logits, dim=-1).item())


def play_game(model: ActorCriticNet, bot_name: str, seed: int, device: str) -> dict[str, object]:
    env = GomokuEnv(
        opponent=create_bot(name=bot_name),
        seed=seed,
        config=EnvConfig(reward_mode="terminal", shape_alpha=0.0, use_action_mask=True),
    )
    obs, mask = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    result = None
    while not done:
        result = env.step(choose_action(model, obs, mask, device))
        obs = result.observation
        mask = result.action_mask
        done = result.done
        total_reward += result.reward
        steps += 1

    assert result is not None
    return {
        "result": result.info.get("agent_result"),
        "illegal_move": bool(result.info.get("illegal_move", False)),
        "total_reward": total_reward,
        "steps": steps,
    }


def evaluate_bot(model: ActorCriticNet, bot_name: str, games: int, seed: int, device: str) -> dict[str, float]:
    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0
    lengths = []
    for index in range(games):
        game = play_game(model, bot_name, seed + index, device)
        if game["result"] == "win":
            wins += 1
        elif game["result"] == "loss":
            losses += 1
        else:
            draws += 1
        illegal_moves += int(game["illegal_move"])
        lengths.append(int(game["steps"]))
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "illegal_moves": float(illegal_moves),
        "win_rate": wins / max(1, games),
        "avg_steps": float(np.mean(lengths)) if lengths else 0.0,
    }


def workshop_score(results: dict[str, dict[str, float]]) -> float:
    weights = {
        "random": 20.0,
        "classic_rule": 40.0,
        "reward_driven_medium": 40.0,
    }
    total_weight = sum(weight for name, weight in weights.items() if name in results)
    if total_weight <= 0:
        return 0.0
    return sum(results[name]["win_rate"] * weight for name, weight in weights.items() if name in results) / total_weight


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    results = {
        bot_name: evaluate_bot(model, bot_name, args.games, args.seed + offset * 10000, device)
        for offset, bot_name in enumerate(args.bots)
    }
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": device,
        "games_per_bot": args.games,
        "results": results,
        "workshop_score": workshop_score(results),
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
