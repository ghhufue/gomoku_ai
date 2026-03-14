from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_ai.env import GomokuEnv
from gomoku_ai.model import ActorCriticNet
from gomoku_ai.rule_bot import RuleBasedBot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Gomoku PPO checkpoint against the rule bot.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-seeds", type=int, default=1)
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


def evaluate_model(
    model: ActorCriticNet,
    games: int,
    device: str,
    seed: int,
    verbose: bool = True,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    opponent = RuleBasedBot()
    env = GomokuEnv(opponent=opponent, seed=seed)

    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0
    episode_lengths: list[int] = []

    for game_idx in range(games):
        obs, mask = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            action = choose_action(model, obs, mask, device)
            result = env.step(action)
            obs = result.observation
            mask = result.action_mask
            done = result.done
            total_reward += result.reward
            steps += 1

        outcome = result.info.get("agent_result")
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            draws += 1

        if result.info.get("illegal_move"):
            illegal_moves += 1

        episode_lengths.append(steps)
        if verbose:
            print(f"game={game_idx + 1} result={outcome} steps={steps} total_reward={total_reward:.2f}")

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
    verbose: bool = True,
) -> dict[str, float]:
    runs: list[dict[str, float]] = []
    for seed in seeds:
        metrics = evaluate_model(model=model, games=games, device=device, seed=seed, verbose=verbose)
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


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> ActorCriticNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ActorCriticNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    print(f"using device: {device}")
    model = load_model_from_checkpoint(args.checkpoint, device)
    seeds = [args.seed + idx for idx in range(args.num_seeds)]
    metrics = evaluate_across_seeds(model=model, games=args.games, device=device, seeds=seeds, verbose=True)

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


if __name__ == "__main__":
    main()
