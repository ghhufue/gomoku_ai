# C++ Backend Status

更新时间：`2026-03-18`

## 目标

把棋盘热点分析、reward 评估和候选落子打分从 Python 下沉到 C++，并接回现有 Python 环境、Bot 和训练脚本。

## 当前已落地

核心文件：

- [cpp/RewardEvaluator.cpp](/d:/code/gomoku_ai/cpp/RewardEvaluator.cpp)
- [cpp/GameStateStore.cpp](/d:/code/gomoku_ai/cpp/GameStateStore.cpp)
- [cpp/RewardConfigStore.cpp](/d:/code/gomoku_ai/cpp/RewardConfigStore.cpp)
- [cpp/StateValueRegistry.cpp](/d:/code/gomoku_ai/cpp/StateValueRegistry.cpp)
- [cpp/direction_encoding.cpp](/d:/code/gomoku_ai/cpp/direction_encoding.cpp)
- [cpp/utils/direction_pattern_lookup.cpp](/d:/code/gomoku_ai/cpp/utils/direction_pattern_lookup.cpp)
- [cpp/precompute/DirectionDeltaTable.cpp](/d:/code/gomoku_ai/cpp/precompute/DirectionDeltaTable.cpp)
- [gomoku_ai/cpp_backend.py](/d:/code/gomoku_ai/gomoku_ai/cpp_backend.py)

当前 C++ 已提供：

- 单步 reward 评估：
  - `evaluate_reward`
  - `evaluate_env_reward`
- 候选动作打分：
  - `score_classic_candidates`
  - `score_reward_candidates`
- 环境状态缓存：
  - `reset_env_state`
  - `apply_env_move`
  - `clear_env_state`
  - `env_done`
  - `env_winner`
- 调试/辅助接口：
  - `list_state_values`
  - `decode_reward_events`
  - 方向编码与查表 debug 接口

Python 侧接线情况：

- [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py) 的 `GomokuEnv.step()` 已经依赖 C++ reward 主路径。
- 环境 reset/step 会同步维护 C++ 侧 `GameStateStore`。
- Python 中保留的 `classify_move_counts / threat_summary / immediate_winning_actions` 目前仍是参考实现，不是环境 reward 主路径。

## 当前 reward 返回结构

目前 `evaluate_reward(...)` / `evaluate_env_reward(...)` 返回：

```python
{
    "reward": float,
    "offense_score": float,
    "defense_score": float,
    "special_rewards": {...},
    "events": [(event_id, count), ...],
}
```

其中：

- `reward` 是最终标量奖励。
- `offense_score` / `defense_score` 是状态增量乘权重后的分数。
- `special_rewards` 记录额外奖励/惩罚项拆分。
- `events` 是压缩后的状态变化摘要。

这说明 C++ reward 已经具备实际可用性，但公开接口仍偏“调试友好”，还没完全收敛到 [cpp/REWARD_INTERFACE.md](/d:/code/gomoku_ai/cpp/REWARD_INTERFACE.md) 里描述的最小契约。

## 设计现状

当前 reward 机制更准确的描述是：

- 基于方向查表和状态值增量。
- 对当前玩家计算 `self_delta`。
- 对对手受影响状态计算 `opp_delta`。
- 再叠加特殊规则：
  - `double_live_three_bonus`
  - `block_winning_bonus`
  - `block_live_four_bonus`
  - `block_live_three_bonus`
  - `unresolved_*_penalty`
  - `missed_immediate_win_penalty`
  - `step_penalty`

这已经不是最早的 Python `threat_summary` 总表差分模式；但也还没有完全变成一套纯事件枚举、极简输出、完全由 C++ 独占配置的最终形态。

## 构建方式

主扩展构建入口：

- [setup.py](/d:/code/gomoku_ai/setup.py)

Windows 辅助脚本：

- [scripts/build_cpp_tools.ps1](/d:/code/gomoku_ai/scripts/build_cpp_tools.ps1)

常用构建命令：

```powershell
python setup.py build_ext --inplace
```

如果只需要构建 C++ 辅助测试工具：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_cpp_tools.ps1
```

当前 `setup.py` 使用：

- `pybind11`
- C++17
- Windows 下 `/std:c++17 /O2 /utf-8`
- 构建产物集中到 `outputs/build/`

## 当前文档与实现的差异

旧文档里有几处已经不准确：

1. 不应再写成“仅把 hot path 接回 Python，reward 语义未改”。
   现在 reward 语义本身已经主要由 C++ 决定。
2. 不应再把核心能力描述成 `classify_move_counts / threat_summary / affected_actions`。
   这些接口现在不是对外主角；真正关键的是 `evaluate_env_reward`、候选打分和状态缓存。
3. 不应再沿用旧的 MinGW 专用构建说明作为唯一构建方式。
   当前 [setup.py](/d:/code/gomoku_ai/setup.py) 明显按 MSVC/Windows 常规路径组织构建。

## 当前未完成项

1. reward 公共接口还没有完全收敛到最小四字段。
2. `special_rewards` 是否保留为默认返回仍未定。
3. 还缺一轮围绕新接口的 Python 工具和测试收敛。
4. 旧测试和文档里仍有一部分基于旧默认值、旧签名、旧模型预设的假设。

## 建议下一步

1. 先同步测试，恢复一套可信的 C++ reward 回归基线。
2. 决定 `special_rewards` 是保留在主接口，还是改成 debug-only 接口。
3. 如果继续推进 reward 重构，再统一：
   - C++ 配置所有权
   - Python info payload
   - replay/debug 工具字段
