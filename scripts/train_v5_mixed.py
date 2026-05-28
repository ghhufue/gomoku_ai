from __future__ import annotations

from train_common import AlphaSchedule, StageConfig, run_stage


if __name__ == "__main__":
    run_stage(
        StageConfig(
            stage_name="v5_mixed",
            description="V5: PPO against a mixed opponent pool for better policy generalization.",
            reward_mode="shaped",
            shape_alpha=0.10,
            use_action_mask=True,
            bot_pool=("random", "classic_rule", "reward_driven_medium"),
            bot_weights=(0.4, 0.4, 0.2),
            alpha_schedule=AlphaSchedule(start=0.10, end=0.02),
        )
    )
