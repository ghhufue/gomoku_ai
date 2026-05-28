from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gomoku_ai.env import BOARD_SIZE, EMPTY, neighboring_action_mask
from gomoku_ai.model import ActorCriticNet, model_config_from_checkpoint_payload


class CheckpointOpponent:
    """Greedy policy opponent loaded from a saved ActorCritic checkpoint."""

    def __init__(self, checkpoint_path: Path | str, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = device
        payload = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.model = ActorCriticNet(model_config_from_checkpoint_payload(payload)).to(device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def next_action(self, board: np.ndarray, player: int, rng: np.random.Generator) -> int:
        del rng
        obs = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (board == player).astype(np.float32)
        obs[1] = (board == -player).astype(np.float32)
        obs[2] = (board == EMPTY).astype(np.float32)
        mask = neighboring_action_mask(board, radius=2, opening_radius=1)
        obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(mask[None, ...], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            dist, _ = self.model.masked_distribution(obs_tensor, mask_tensor)
        return int(torch.argmax(dist.logits, dim=-1).item())
