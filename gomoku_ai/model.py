from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ActorCriticNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        trunk = [
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        trunk.extend(ResidualBlock(channels) for _ in range(blocks))
        self.trunk = nn.Sequential(*trunk)

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 15 * 15, 225),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(15 * 15, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def masked_distribution(self, obs: torch.Tensor, action_mask: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(obs)
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits), value

    def sample_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.masked_distribution(obs, action_mask)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.masked_distribution(obs, action_mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
