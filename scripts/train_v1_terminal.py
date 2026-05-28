from __future__ import annotations

from train_common import StageConfig, run_stage


if __name__ == "__main__":
    run_stage(
        StageConfig(
            stage_name="v1_terminal",
            description="V1: PPO with terminal win/loss rewards against a random bot.",
            reward_mode="terminal",
            shape_alpha=0.0,
            use_action_mask=False,
            bot_pool=("random",),
            bot_weights=(1.0,),
        )
    )
