from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from gomoku_ai.model import (
    ActorCriticNet,
    MODEL_PRESETS,
    ModelConfig,
    infer_model_config_from_state_dict,
    model_config_from_checkpoint_payload,
    model_preset_config,
)
from scripts.train import build_model_config
from scripts.evaluate import load_model_from_checkpoint


def test_infer_model_config_from_state_dict_round_trips_custom_shape() -> None:
    config = ModelConfig(
        channels=96,
        blocks=6,
        policy_channels=4,
        value_channels=3,
        value_hidden_dim=256,
    )
    model = ActorCriticNet(config)

    inferred = infer_model_config_from_state_dict(model.state_dict())

    assert inferred == config


def test_model_config_from_checkpoint_payload_prefers_explicit_metadata() -> None:
    payload = {
        "model_config": {"channels": 32},
        "extra": {
            "model_config": {
                "channels": 128,
                "blocks": 8,
                "policy_channels": 4,
                "value_channels": 2,
                "value_hidden_dim": 192,
            }
        },
    }

    config = model_config_from_checkpoint_payload(payload)

    assert config == ModelConfig(
        channels=128,
        blocks=8,
        policy_channels=4,
        value_channels=2,
        value_hidden_dim=192,
    )


def test_load_model_from_checkpoint_supports_legacy_state_only_payload(tmp_path: Path) -> None:
    config = ModelConfig(channels=80, blocks=5, policy_channels=3, value_channels=2, value_hidden_dim=160)
    source_model = ActorCriticNet(config)
    checkpoint_path = tmp_path / "legacy_model.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    restored = load_model_from_checkpoint(checkpoint_path, device="cpu")

    assert restored.config == config
    assert restored.state_dict().keys() == source_model.state_dict().keys()


def test_model_preset_config_supports_named_sizes() -> None:
    assert "custom" in MODEL_PRESETS
    assert model_preset_config("small") == ModelConfig(
        channels=48,
        blocks=4,
        policy_channels=2,
        value_channels=1,
        value_hidden_dim=96,
    )
    assert model_preset_config("base") == ModelConfig(
        channels=64,
        blocks=6,
    )
    assert model_preset_config("large") == ModelConfig(
        channels=128,
        blocks=10,
        policy_channels=4,
        value_channels=2,
        value_hidden_dim=256,
    )


def test_build_model_config_uses_named_preset_without_toml_overrides() -> None:
    args = SimpleNamespace(
        model_preset=None,
        model_channels=160,
        model_blocks=None,
        policy_channels=None,
        value_channels=None,
        value_hidden_dim=None,
    )

    config = build_model_config({"preset": "large", "channels": 64, "value_hidden_dim": 320}, args)

    assert config == ModelConfig(
        channels=160,
        blocks=10,
        policy_channels=4,
        value_channels=2,
        value_hidden_dim=256,
    )


def test_build_model_config_uses_custom_preset_with_toml_values() -> None:
    args = SimpleNamespace(
        model_preset=None,
        model_channels=None,
        model_blocks=None,
        policy_channels=None,
        value_channels=None,
        value_hidden_dim=None,
    )

    config = build_model_config(
        {
            "preset": "custom",
            "channels": 160,
            "blocks": 7,
            "policy_channels": 5,
            "value_channels": 3,
            "value_hidden_dim": 320,
        },
        args,
    )

    assert config == ModelConfig(
        channels=160,
        blocks=7,
        policy_channels=5,
        value_channels=3,
        value_hidden_dim=320,
    )
