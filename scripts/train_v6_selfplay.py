from __future__ import annotations

from train_common import AlphaSchedule, StageConfig, run_stage


if __name__ == "__main__":
    run_stage(
        StageConfig(
            stage_name="v6_selfplay",
            description="V6: PPO with mixed bots plus optional historical checkpoint opponents.",
            reward_mode="shaped",
            shape_alpha=0.05,
            use_action_mask=True,
            bot_pool=("classic_rule", "reward_driven_medium"),
            bot_weights=(0.5, 0.3),
            checkpoint_weight=0.2,
            alpha_schedule=AlphaSchedule(start=0.05, end=0.01),
        )
    )
