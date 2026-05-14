from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib

import torch
from torch import nn
from torch.distributions import Categorical


BOARD_SIZE = 15
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
OBS_CHANNELS = 4
PRESETS_PATH = Path(__file__).with_name("presets.toml")


def load_model_presets() -> dict[str, dict[str, int]]:
    payload = tomllib.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    presets = payload.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError(f"model presets missing or invalid: {PRESETS_PATH}")
    normalized: dict[str, dict[str, int]] = {}
    for name, config in presets.items():
        if not isinstance(config, dict):
            raise ValueError(f"model preset must be a table: {name}")
        normalized[name] = {key: int(value) for key, value in config.items()}
    return normalized


MODEL_PRESET_CONFIGS = load_model_presets()
CUSTOM_MODEL_PRESET = "custom"
MODEL_PRESETS = tuple((*MODEL_PRESET_CONFIGS.keys(), CUSTOM_MODEL_PRESET))


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 4
    channels: int = 64
    blocks: int = 4
    policy_channels: int = 2
    value_channels: int = 1
    value_hidden_dim: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ModelConfig":
        if payload is None:
            return cls()
        defaults = cls()
        values = defaults.to_dict()
        for key in values:
            if key in payload:
                values[key] = int(payload[key])
        return cls(**values)


def model_preset_config(name: str) -> ModelConfig:
    preset = name.strip().lower()
    if preset not in MODEL_PRESET_CONFIGS:
        raise ValueError(f"unsupported model preset: {name}")
    return ModelConfig.from_dict(MODEL_PRESET_CONFIGS[preset])


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
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        trunk = [
            nn.Conv2d(self.config.input_channels, self.config.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.config.channels),
            nn.ReLU(inplace=True),
        ]
        trunk.extend(ResidualBlock(self.config.channels) for _ in range(self.config.blocks))
        self.trunk = nn.Sequential(*trunk)

        self.policy_head = nn.Sequential(
            nn.Conv2d(self.config.channels, self.config.policy_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.policy_channels * BOARD_AREA, BOARD_AREA),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(self.config.channels, self.config.value_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.value_channels * BOARD_AREA, self.config.value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden_dim, 1),
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


def model_config_from_checkpoint_payload(payload: dict) -> ModelConfig:
    extra = payload.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("model_config"), dict):
        return ModelConfig.from_dict(extra["model_config"])

    top_level_model_config = payload.get("model_config")
    if isinstance(top_level_model_config, dict):
        return ModelConfig.from_dict(top_level_model_config)

    config = payload.get("config")
    if isinstance(config, dict):
        model_section = config.get("model")
        if isinstance(model_section, dict):
            return ModelConfig.from_dict(model_section)

    model_state = payload.get("model_state_dict")
    if isinstance(model_state, dict):
        return infer_model_config_from_state_dict(model_state)
    return ModelConfig()


def infer_model_config_from_state_dict(model_state: dict[str, torch.Tensor]) -> ModelConfig:
    first_conv_weight = model_state.get("trunk.0.weight")
    channels = int(first_conv_weight.shape[0]) if first_conv_weight is not None else ModelConfig().channels
    input_channels = int(first_conv_weight.shape[1]) if first_conv_weight is not None else ModelConfig().input_channels
    residual_indices = {
        int(name.split(".")[1])
        for name in model_state
        if name.startswith("trunk.") and name.endswith("net.0.weight")
    }
    blocks = len(residual_indices)
    policy_channels = int(model_state["policy_head.0.weight"].shape[0])
    value_channels = int(model_state["value_head.0.weight"].shape[0])
    value_hidden_dim = int(model_state["value_head.3.weight"].shape[0])
    return ModelConfig(
        input_channels=input_channels,
        channels=channels,
        blocks=blocks,
        policy_channels=policy_channels,
        value_channels=value_channels,
        value_hidden_dim=value_hidden_dim,
    )


def validate_checkpoint_channels(payload: dict, expected_input_channels: int = 4) -> None:
    """Validate checkpoint input channels compatibility.
    
    Raises ValueError with a clear message if the checkpoint uses an 
    incompatible number of input channels.
    """
    model_state = payload.get("model_state_dict")
    first_conv = None
    if isinstance(model_state, dict):
        first_conv = model_state.get("trunk.0.weight")
    
    ckpt_channels = None
    if first_conv is not None:
        ckpt_channels = int(first_conv.shape[1])
    
    if ckpt_channels is None:
        # Try to infer from model_config metadata
        extra = payload.get("extra") or {}
        model_cfg = extra.get("model_config", {}) if isinstance(extra, dict) else {}
        ckpt_channels = model_cfg.get("input_channels", None)
    
    if ckpt_channels is None:
        # Cannot determine, assume compatible
        return
    
    if ckpt_channels != expected_input_channels:
        raise ValueError(
            f"Checkpoint input_channels mismatch: "
            f"checkpoint has {ckpt_channels} channels, "
            f"but current model expects {expected_input_channels} channels. "
            f"Cannot load a {ckpt_channels}-channel checkpoint into a {expected_input_channels}-channel model. "
            f"Please use a checkpoint trained with the same input_channel configuration, "
            f"or start a new training run."
        )
