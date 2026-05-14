from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    SummaryWriter = Any

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


REWARD_COMPONENT_KEYS = (
    "total_reward",
    "terminal_reward",
    "raw_auxiliary_reward",
    "scaled_auxiliary_reward",
    "final_reward",
)


@dataclass
class PPOConfig:
    n_envs: int = 8
    n_steps: int = 128
    total_updates: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.05
    entropy_coef: float = 0.03
    max_grad_norm: float = 0.5
    batch_size: int = 256
    epochs: int = 4
    device: str = "cpu"
    show_progress: bool = True
    # Reward scaling config
    auxiliary_reward_scale: float = 0.001
    auxiliary_reward_clip: float = 1.0
    auxiliary_coef_start: float = 0.1
    auxiliary_coef_end: float = 0.0
    auxiliary_coef_anneal_updates: int = 5000


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    targets = np.asarray(y_true, dtype=np.float32)
    predictions = np.asarray(y_pred, dtype=np.float32)
    target_var = float(np.var(targets))
    if target_var <= 1e-8:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_var)


def init_reward_component_stats() -> dict[str, float]:
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


def accumulate_reward_component_stats(
    totals: dict[str, float],
    infos: list[dict],
    count: int,
) -> tuple[dict[str, float], int]:
    for info in infos:
        totals["terminal_reward"] += float(info.get("terminal_reward", 0.0))
        totals["raw_auxiliary_reward"] += float(info.get("raw_auxiliary_reward", 0.0))
        scaled_aux = float(info.get("scaled_auxiliary_reward", 0.0))
        totals["scaled_auxiliary_reward"] += scaled_aux
        totals["final_reward"] += float(info.get("final_reward", 0.0))
        count += 1
    return totals, count


def compute_auxiliary_coef(update: int, config: PPOConfig) -> float:
    """Compute the auxiliary reward coefficient for the given update number.
    
    Linear annealing from auxiliary_coef_start to auxiliary_coef_end.
    """
    denom = max(1, config.auxiliary_coef_anneal_updates)
    progress = min(float(update) / denom, 1.0)
    return config.auxiliary_coef_start + progress * (config.auxiliary_coef_end - config.auxiliary_coef_start)


def scale_auxiliary_reward(raw_aux: float, config: PPOConfig) -> float:
    """Scale and clip the raw auxiliary reward."""
    scaled = raw_aux * config.auxiliary_reward_scale
    return float(np.clip(scaled, -config.auxiliary_reward_clip, config.auxiliary_reward_clip))


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.reset()

    def reset(self) -> None:
        self.obs: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.obs.append(obs.copy())
        self.masks.append(action_mask.copy())
        self.actions.append(actions.copy())
        self.log_probs.append(log_probs.copy())
        self.rewards.append(rewards.copy())
        self.dones.append(dones.copy())
        self.values.append(values.copy())

    def compute_returns_advantages(self, last_values: np.ndarray, config: PPOConfig) -> dict[str, np.ndarray]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - dones[step]
            else:
                next_values = values[step + 1]
                next_non_terminal = 1.0 - dones[step]

            delta = rewards[step] + config.gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae

        returns = advantages + values
        return {
            "obs": np.asarray(self.obs, dtype=np.float32).reshape(-1, 4, 15, 15),
            "masks": np.asarray(self.masks, dtype=bool).reshape(-1, 225),
            "actions": np.asarray(self.actions, dtype=np.int64).reshape(-1),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32).reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": values.reshape(-1),
        }


class PPOTrainer:
    def __init__(self, model: nn.Module, env, config: PPOConfig, writer: SummaryWriter | None = None):
        self.model = model.to(config.device)
        self.env = env
        self.config = config
        self.writer = writer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def collect_rollout(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        update: int,
    ) -> tuple[RolloutBuffer, np.ndarray, np.ndarray, dict]:
        buffer = RolloutBuffer(self.config.n_steps, self.config.n_envs)
        episode_rewards: list[float] = []
        finished_rewards = np.zeros(self.config.n_envs, dtype=np.float32)
        finished_lengths = np.zeros(self.config.n_envs, dtype=np.int32)
        stats = {"wins": 0, "losses": 0, "draws": 0}
        # Per-bot statistics
        bot_stats: dict[str, dict[str, float]] = {}
        reward_component_totals = init_reward_component_stats()
        reward_component_count = 0
        
        # Compute fixed auxiliary_coef for this rollout
        auxiliary_coef = compute_auxiliary_coef(update, self.config)
        stats["auxiliary_coef"] = auxiliary_coef
        
        iterator = range(self.config.n_steps)
        if self.config.show_progress:
            iterator = tqdm(
                iterator,
                total=self.config.n_steps,
                desc=f"rollout u{update}",
                leave=False,
                position=1,
            )

        for _ in iterator:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.config.device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.config.device)
            with torch.no_grad():
                actions, log_probs, values = self.model.sample_action(obs_tensor, mask_tensor)

            next_obs, next_masks, rewards, dones, infos = self.env.step(actions.cpu().numpy())
            
            # Apply reward scaling: final_reward = terminal + auxiliary_coef * scaled_aux
            scaled_rewards = np.zeros_like(rewards)
            for idx, (r, info) in enumerate(zip(rewards, infos)):
                terminal = float(info.get("terminal_reward", 0.0))
                raw_aux = float(info.get("raw_auxiliary_reward", r - terminal))
                scaled_aux = scale_auxiliary_reward(raw_aux, self.config)
                final_reward = terminal + auxiliary_coef * scaled_aux
                scaled_rewards[idx] = final_reward
                info["scaled_auxiliary_reward"] = scaled_aux
                info["final_reward"] = final_reward
            
            reward_component_totals, reward_component_count = accumulate_reward_component_stats(
                reward_component_totals,
                infos,
                reward_component_count,
            )
            buffer.add(
                obs,
                masks,
                actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                scaled_rewards.astype(np.float32),
                dones,
                values.cpu().numpy(),
            )

            finished_rewards += scaled_rewards
            finished_lengths += 1
            for idx, done in enumerate(dones):
                if done:
                    episode_rewards.append(float(finished_rewards[idx]))
                    finished_rewards[idx] = 0.0
                    finished_lengths[idx] = 0
                    result = infos[idx].get("agent_result")
                    bot_name = str(infos[idx].get("opponent_bot", "unknown"))
                    if bot_name not in bot_stats:
                        bot_stats[bot_name] = {"games": 0, "wins": 0, "losses": 0, "draws": 0, 
                                                "total_reward": 0.0, "total_length": 0.0}
                    bot_stats[bot_name]["games"] += 1
                    bot_stats[bot_name]["total_reward"] += float(finished_rewards[idx])
                    bot_stats[bot_name]["total_length"] += float(finished_lengths[idx])
                    if result == "win":
                        stats["wins"] += 1
                        bot_stats[bot_name]["wins"] += 1
                    elif result == "loss":
                        stats["losses"] += 1
                        bot_stats[bot_name]["losses"] += 1
                    elif result == "draw" or infos[idx].get("draw"):
                        stats["draws"] += 1
                        bot_stats[bot_name]["draws"] += 1

            obs, masks = next_obs, next_masks

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.config.device)
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.config.device)
        with torch.no_grad():
            _, last_values = self.model.masked_distribution(obs_tensor, mask_tensor)

        data = buffer.compute_returns_advantages(last_values.cpu().numpy(), self.config)
        stats["ep_rew_mean"] = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        stats["episodes"] = len(episode_rewards)
        for key, total in reward_component_totals.items():
            stats[key] = total / max(1, reward_component_count)
        # Compute per-bot summary stats
        for bot_name, bstats in bot_stats.items():
            g = bstats["games"]
            if g > 0:
                stats[f"bot_{bot_name}_games"] = g
                stats[f"bot_{bot_name}_win_rate"] = bstats["wins"] / g
                stats[f"bot_{bot_name}_draw_rate"] = bstats["draws"] / g
                stats[f"bot_{bot_name}_loss_rate"] = bstats["losses"] / g
                stats[f"bot_{bot_name}_avg_reward"] = bstats["total_reward"] / g
                stats[f"bot_{bot_name}_avg_length"] = bstats["total_length"] / g
        return data, obs, masks, stats

    def update(self, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        advantages = rollout["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rollout["advantages"] = advantages

        total_items = rollout["actions"].shape[0]
        indices = np.arange(total_items)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "value_loss_weighted": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        batches = 0

        for _ in range(self.config.epochs):
            np.random.shuffle(indices)
            for start in range(0, total_items, self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                obs = torch.as_tensor(rollout["obs"][batch_idx], dtype=torch.float32, device=self.config.device)
                masks = torch.as_tensor(rollout["masks"][batch_idx], dtype=torch.bool, device=self.config.device)
                actions = torch.as_tensor(rollout["actions"][batch_idx], dtype=torch.int64, device=self.config.device)
                old_log_probs = torch.as_tensor(
                    rollout["log_probs"][batch_idx], dtype=torch.float32, device=self.config.device
                )
                returns = torch.as_tensor(rollout["returns"][batch_idx], dtype=torch.float32, device=self.config.device)
                batch_advantages = torch.as_tensor(
                    rollout["advantages"][batch_idx], dtype=torch.float32, device=self.config.device
                )

                log_probs, entropy, values = self.model.evaluate_actions(obs, masks, actions)
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                entropy_loss = entropy.mean()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > self.config.clip_range).float().mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["value_loss_weighted"] += float((self.config.value_coef * value_loss).item())
                metrics["entropy"] += float(entropy_loss.item())
                metrics["approx_kl"] += float(approx_kl.item())
                metrics["clip_fraction"] += float(clip_fraction.item())
                batches += 1

        for key in metrics:
            metrics[key] /= max(1, batches)
        metrics["explained_variance"] = explained_variance(rollout["values"], rollout["returns"])
        return metrics

    def train(
        self,
        start_update: int = 0,
        on_update_end: Callable[[int, dict[str, float], nn.Module], None] | None = None,
    ) -> list[dict[str, float]]:
        obs, masks = self.env.reset()
        history: list[dict[str, float]] = []
        iterator = range(1, self.config.total_updates + 1)
        if self.config.show_progress:
            iterator = tqdm(
                iterator,
                total=self.config.total_updates,
                desc="training",
                leave=True,
                position=0,
            )

        for local_update in iterator:
            update = start_update + local_update
            rollout_start = perf_counter()
            rollout, obs, masks, rollout_stats = self.collect_rollout(obs, masks, update)
            rollout_time = perf_counter() - rollout_start
            optimize_start = perf_counter()
            train_stats = self.update(rollout)
            optimize_time = perf_counter() - optimize_start
            total_time = rollout_time + optimize_time
            samples = self.config.n_envs * self.config.n_steps
            stats = {
                "update": float(update),
                "rollout_time_s": rollout_time,
                "optimize_time_s": optimize_time,
                "total_time_s": total_time,
                "samples_per_sec": float(samples / max(total_time, 1e-6)),
                **rollout_stats,
                **train_stats,
            }
            history.append(stats)
            self.log_stats(update, stats)
            message = (
                "update={update} episodes={episodes} ep_rew_mean={ep_rew_mean:.4f} "
                "wins={wins} losses={losses} draws={draws} "
                "terminal={terminal_reward:.4f} raw_aux={raw_auxiliary_reward:.2f} "
                "scaled_aux={scaled_auxiliary_reward:.4f} coef={auxiliary_coef:.4f} "
                "policy_loss={policy_loss:.4f} "
                "value_loss={value_loss_weighted:.4f} entropy={entropy:.4f} "
                "rollout={rollout_time_s:.1f}s optimize={optimize_time_s:.1f}s fps={samples_per_sec:.1f}"
            ).format(**stats)
            if self.config.show_progress:
                tqdm.write(message)
                iterator.set_postfix(
                    ep_rew=f"{stats['ep_rew_mean']:.1f}",
                    ploss=f"{stats['policy_loss']:.1f}",
                    vloss=f"{stats['value_loss_weighted']:.1f}",
                    fps=f"{stats['samples_per_sec']:.1f}",
                )
            else:
                print(message)
            if on_update_end is not None:
                on_update_end(update, stats, self.model)

        return history

    def log_stats(self, update: int, stats: dict[str, float]) -> None:
        if self.writer is None:
            return

        self.writer.add_scalar("train/ep_rew_mean", stats["ep_rew_mean"], update)
        self.writer.add_scalar("train/episodes", stats["episodes"], update)
        self.writer.add_scalar("train/wins", stats["wins"], update)
        self.writer.add_scalar("train/losses", stats["losses"], update)
        self.writer.add_scalar("train/draws", stats["draws"], update)
        self.writer.add_scalar("loss/policy", stats["policy_loss"], update)
        self.writer.add_scalar("loss/value", stats["value_loss"], update)
        self.writer.add_scalar("value/explained_variance", stats["explained_variance"], update)
        self.writer.add_scalar("policy/entropy", stats["entropy"], update)
        self.writer.add_scalar("policy/approx_kl", stats["approx_kl"], update)
        self.writer.add_scalar("policy/clip_fraction", stats["clip_fraction"], update)
        self.writer.add_scalar("reward/total_reward", stats.get("total_reward", 0), update)
        self.writer.add_scalar("reward/terminal_reward", stats.get("terminal_reward", 0), update)
        self.writer.add_scalar("reward/raw_auxiliary_reward", stats.get("raw_auxiliary_reward", 0), update)
        self.writer.add_scalar("reward/scaled_auxiliary_reward", stats.get("scaled_auxiliary_reward", 0), update)
        self.writer.add_scalar("reward/final_reward", stats.get("final_reward", 0), update)
        self.writer.add_scalar("reward/auxiliary_coef", stats.get("auxiliary_coef", 0), update)
        self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], update)
        self.writer.add_scalar("perf/rollout_time_s", stats["rollout_time_s"], update)
        self.writer.add_scalar("perf/optimize_time_s", stats["optimize_time_s"], update)
        self.writer.add_scalar("perf/samples_per_sec", stats["samples_per_sec"], update)
        self.writer.flush()
