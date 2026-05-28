from __future__ import annotations

from train_common import StageConfig, run_stage


if __name__ == "__main__":
    run_stage(
        StageConfig(
            stage_name="v3_shaped",
            description="V3: PPO with action masking and C++ pattern-based reward shaping.",
            reward_mode="shaped",
            shape_alpha=0.05,
            use_action_mask=True,
            bot_pool=("classic_rule",),
            bot_weights=(1.0,),
        )
    )
