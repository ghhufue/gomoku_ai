from __future__ import annotations

import tomllib
from pathlib import Path

from bots.base import Bot
from bots.classic_rule_bot import ClassicRuleBot
from bots.random_bot import RandomBot
from bots.reward_driven_bot import RewardDrivenBot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOT_CONFIG_PATH = ROOT / "configs" / "bots.toml"

def load_bot_registry(config_path: Path = DEFAULT_BOT_CONFIG_PATH) -> dict:
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _build_alias_map(registry: dict) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for bot_name, payload in registry.get("bots", {}).items():
        alias_map[bot_name.lower()] = bot_name
        for alias in payload.get("aliases", []):
            alias_map[str(alias).strip().lower()] = bot_name
    return alias_map


def resolve_bot_name(name: str, config_path: Path = DEFAULT_BOT_CONFIG_PATH) -> str:
    registry = load_bot_registry(config_path)
    normalized = _build_alias_map(registry).get(name.strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported bot: {name}")
    return normalized


def resolve_bot_from_difficulty(difficulty: str, config_path: Path = DEFAULT_BOT_CONFIG_PATH) -> str:
    registry = load_bot_registry(config_path)
    payload = registry.get("difficulty", {}).get(difficulty.strip().lower())
    if payload is None:
        raise ValueError(f"unsupported bot difficulty: {difficulty}")
    return str(payload["bot"])


def create_bot(
    name: str | None = None,
    *,
    difficulty: str | None = None,
    config_path: Path = DEFAULT_BOT_CONFIG_PATH,
) -> Bot:
    registry = load_bot_registry(config_path)
    if difficulty:
        resolved_name = resolve_bot_from_difficulty(difficulty, config_path)
    elif name:
        resolved_name = resolve_bot_name(name, config_path)
    else:
        raise ValueError("either name or difficulty must be provided")

    payload = registry["bots"][resolved_name]
    kind = str(payload["kind"]).strip().lower()
    if kind == "reward_driven":
        return RewardDrivenBot(
            candidate_radius=int(payload.get("candidate_radius", 2)),
            top_k=int(payload.get("top_k", 1)),
        )
    if kind == "classic_rule":
        return ClassicRuleBot(candidate_radius=int(payload.get("candidate_radius", 2)))
    if kind == "random":
        return RandomBot()
    raise ValueError(f"unsupported bot kind: {kind}")


def available_bots(config_path: Path = DEFAULT_BOT_CONFIG_PATH) -> list[str]:
    registry = load_bot_registry(config_path)
    return sorted(registry.get("bots", {}).keys())


def available_difficulties(config_path: Path = DEFAULT_BOT_CONFIG_PATH) -> list[str]:
    registry = load_bot_registry(config_path)
    return sorted(registry.get("difficulty", {}).keys())


def bot_table(config_path: Path = DEFAULT_BOT_CONFIG_PATH) -> list[dict[str, object]]:
    registry = load_bot_registry(config_path)
    rows: list[dict[str, object]] = []
    for bot_name, payload in registry.get("bots", {}).items():
        rows.append(
            {
                "name": bot_name,
                "kind": payload.get("kind"),
                "difficulty": payload.get("difficulty"),
                "aliases": list(payload.get("aliases", [])),
                "capabilities": list(payload.get("capabilities", [])),
            }
        )
    return rows
