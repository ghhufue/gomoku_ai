# `tests/test_env.py` 测试意图说明

下面按序号列出 `tests/test_env.py` 中每个测试的目标。可通过 `gmkt test --case-id <序号>` 或 `python tests/test_env.py --case-id <序号>` 运行单个测试。

> 说明：
> - `debug` 列表示该测试支持额外打印棋盘、落子位置、reward 和 reward 构成。
> - 没有 `debug` 的测试仍可单独运行，但只会输出通过信息。

| 序号 | 测试名 | 测试意图 | debug |
|---|---|---|---|
| 1 | `test_classify_move_detects_five_in_row` | 验证成五识别正确。 | `6` |
| 2 | `test_classify_move_detects_live_four` | 验证活四识别正确。 |  |
| 3 | `test_classify_move_detects_live_three` | 验证活三识别正确。 | `2` |
| 4 | `test_classify_move_detects_rush_four` | 验证冲四识别正确。 |  |
| 5 | `test_classify_move_detects_sleep_three` | 验证眠三识别正确。 |  |
| 6 | `test_classify_move_detects_double_live_three` | 验证双活三识别正确。 |  |
| 7 | `test_immediate_winning_actions_finds_both_ends` | 验证立即制胜点搜索能找到两端。 |  |
| 8 | `test_threat_summary_counts_winning_actions` | 验证 `threat_summary` 对立即制胜点计数正确。 |  |
| 9 | `test_threat_summary_open_three_counts_live_four_actions_without_rush_four_overlap` | 验证活三局面不会把 `live_four` 与 `rush_four` 重叠计数。 |  |
| 10 | `test_evaluate_reward_rewards_blocking_opponent_win` | 验证成功挡住对手立即取胜点会获得高奖励。 | `1` |
| 11 | `test_evaluate_reward_rewards_live_three_creation` | 验证形成自己的活三会获得正奖励。 | `2` |
| 12 | `test_evaluate_reward_rewards_double_live_three_more_than_single_live_three` | 验证双活三奖励高于单活三。 |  |
| 13 | `test_evaluate_reward_partial_block_does_not_get_critical_bonus` | 验证只挡住一部分必胜点时不会拿到完整关键奖励。 |  |
| 14 | `test_evaluate_reward_penalizes_ignoring_opponent_winning_threat` | 验证无视对手立即取胜威胁会被重罚。 |  |
| 15 | `test_evaluate_reward_penalizes_ignoring_opponent_live_four` | 验证无视对手四威胁会被惩罚。 | `3` |
| 16 | `test_evaluate_reward_penalizes_ignoring_opponent_live_three` | 验证无视对手活三前驱威胁会被惩罚。 | `4` |
| 17 | `test_evaluate_reward_blocking_opponent_live_three_avoids_live_three_penalty` | 验证去挡对手活三会优于完全不挡。 | `5` |
| 18 | `test_evaluate_reward_live_two_stays_small` | 验证活二奖励保持较小。 |  |
| 19 | `test_evaluate_reward_reward_is_reasonably_bounded_for_non_terminal_move` | 验证非终局单步奖励有合理上界。 |  |
| 20 | `test_evaluate_reward_returns_terminal_reward_for_win` | 验证成五时直接返回终局奖励。 | `6` |
| 21 | `test_evaluate_reward_uses_custom_reward_config` | 验证 reward 配置可被自定义参数覆盖。 | `6` |
| 22 | `test_env_illegal_move_ends_episode` | 验证环境中非法落子会直接判负终局。 |  |
| 23 | `test_env_reports_win_before_opponent_turn` | 验证我方成五时环境会在对手行动前结束。 |  |
| 24 | `test_cpp_backend_matches_python_reference_on_hot_path` | 验证 C++ 后端与 Python 参考实现一致。 |  |

## 调用示例

- 跑全部测试：
  - `gmkt test`
  - `python tests/test_env.py`

- 查看序号列表：
  - `gmkt test --list-cases`
  - `python tests/test_env.py --list-cases`

- 跑单个测试：
  - `gmkt test --case-id 17`
  - `python tests/test_env.py --case-id 17`

- 直接按调试案例名打印 reward 细节：
  - `gmkt test --case block_opponent_live_three`
  - `python tests/test_env.py --case block_opponent_live_three`
