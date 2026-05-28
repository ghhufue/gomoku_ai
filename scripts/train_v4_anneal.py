from __future__ import annotations

from train_common import AlphaSchedule, StageConfig, run_stage


if __name__ == "__main__":
    run_stage(
        StageConfig(
            stage_name="v4_anneal",
            description="V4: PPO with shaped reward annealed back toward the terminal objective.",
            reward_mode="shaped",
            shape_alpha=0.20,
            use_action_mask=True,
            bot_pool=("classic_rule",),
            bot_weights=(1.0,),
            alpha_schedule=AlphaSchedule(start=0.20, end=0.02),
        )
    )
