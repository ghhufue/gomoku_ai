from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bots.base import Bot
from gomoku_ai.model.network import (
    ActorCriticNet,
    model_config_from_checkpoint_payload,
)
from gomoku_ai.env import BOARD_SIZE


def _board_to_obs(board: np.ndarray, player: int) -> torch.Tensor:
    """Convert raw board to 3-channel observation tensor (1, 3, H, W)."""
    opponent = 3 - player
    ch0 = (board == player).astype(np.float32)
    ch1 = (board == opponent).astype(np.float32)
    ch2 = np.ones_like(ch0) if player == 1 else np.zeros_like(ch0)
    obs = np.stack([ch0, ch1, ch2], axis=0)  # (3, H, W)
    return torch.from_numpy(obs).unsqueeze(0)  # (1, 3, H, W)


class TrainedBot(Bot):
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        config = model_config_from_checkpoint_payload(payload)
        self.model = ActorCriticNet(config)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        obs = _board_to_obs(board, player)
        action_mask = torch.from_numpy((board == 0).flatten())

        with torch.no_grad():
            logits, _ = self.model(obs)
            masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
            action = int(masked_logits.argmax(dim=-1).item())  # greedy best move
        return action