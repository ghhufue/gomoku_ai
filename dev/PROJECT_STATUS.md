# 项目状态记录

更新时间：`2026-03-18`

## 当前目标

这个仓库仍然是一个面向 PPO 教学和工程演化的五子棋项目，而不是追求最强棋力的最终版本。当前主线目标是：

1. 保持训练闭环可运行。
2. 让训练、评估、回放、run 管理和 profiling 数据链路完整。
3. 持续把热点棋盘分析逻辑下沉到 C++。
4. 把 reward 逻辑从旧的 threat-summary 思路进一步收敛到更稳定的 C++ 事件/状态增量路径。

## 当前实现快照

### 环境与对局

核心文件：

- [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py)
- [bots/factory.py](/d:/code/gomoku_ai/bots/factory.py)
- [configs/bots.toml](/d:/code/gomoku_ai/configs/bots.toml)

当前已实现：

- `15x15` 五子棋环境。
- 单智能体对内置 Bot 对战。
- 非法落子直接判负。
- 随机先后手；若智能体执白，环境会先执行 Bot 开局一步。
- 观测为 `3 x 15 x 15`：
  - 当前玩家棋子
  - 对手棋子
  - 空位
- 动作掩码为 `225` 维合法落子布尔数组。
- 同时提供：
  - `VectorEnv`
  - `SubprocVectorEnv`

当前内置对手：

- `random`
- `classic_rule`
- `reward_driven_medium`
- `reward_driven_hard`

难度映射来自 [configs/bots.toml](/d:/code/gomoku_ai/configs/bots.toml)：

- `easy -> random`
- `medium -> classic_rule`
- `hard -> reward_driven_hard`

### Reward 与 C++ backend

核心文件：

- [cpp/RewardEvaluator.cpp](/d:/code/gomoku_ai/cpp/RewardEvaluator.cpp)
- [cpp/GameStateStore.cpp](/d:/code/gomoku_ai/cpp/GameStateStore.cpp)
- [cpp/RewardConfigStore.cpp](/d:/code/gomoku_ai/cpp/RewardConfigStore.cpp)
- [gomoku_ai/cpp_backend.py](/d:/code/gomoku_ai/gomoku_ai/cpp_backend.py)

当前真实状态：

- 环境 step 的 reward 主路径已经走 C++，不是旧文档里的 Python reward 主算路。
- `GomokuEnv.step()` 会调用 `cpp_evaluate_env_reward(...)`，并配合 `GameStateStore` 维护每个 env 的棋盘与计数缓存。
- C++ 当前对 Python 暴露的核心能力包括：
  - `evaluate_reward`
  - `evaluate_env_reward`
  - `score_classic_candidates`
  - `score_reward_candidates`
  - `reset_env_state / apply_env_move / env_done / env_winner`
  - 若干 debug / decode 接口
- 当前 reward 返回给 Python 的主字段已经包括：
  - `reward`
  - `offense_score`
  - `defense_score`
  - `events`
  - `special_rewards`

这意味着 [cpp/REWARD_INTERFACE.md](/d:/code/gomoku_ai/cpp/REWARD_INTERFACE.md) 里的“目标接口”已经部分落地，但还没有完全收敛到“只返回最小 4 字段”的最终形式，因为目前仍附带 `special_rewards` 调试信息。

### 方向编码与状态值表

核心文件：

- [cpp/direction_encoding.cpp](/d:/code/gomoku_ai/cpp/direction_encoding.cpp)
- [cpp/utils/direction_pattern_lookup.cpp](/d:/code/gomoku_ai/cpp/utils/direction_pattern_lookup.cpp)
- [cpp/StateValueRegistry.cpp](/d:/code/gomoku_ai/cpp/StateValueRegistry.cpp)
- [cpp/precompute/DirectionDeltaTable.cpp](/d:/code/gomoku_ai/cpp/precompute/DirectionDeltaTable.cpp)

当前已实现：

- 四个方向的局部侧翼编码。
- 基于查表的状态增量计算。
- 追踪状态值的注册表与描述信息。
- reward event 的 decode/debug 接口。

当前未完成：

- 还没有把 reward 进一步压缩成纯净、稳定、不带附加调试字段的最终公共接口。
- 旧文档中提到的“event-driven reward 完整替代旧路径”还没有完全结束；现在更准确的说法是：
  - C++ 已经在做状态增量驱动的 reward 计算；
  - 但接口清理、配置所有权收敛、周边工具适配仍在进行中。

### 模型与 PPO

核心文件：

- [gomoku_ai/model/network.py](/d:/code/gomoku_ai/gomoku_ai/model/network.py)
- [gomoku_ai/model/presets.toml](/d:/code/gomoku_ai/gomoku_ai/model/presets.toml)
- [gomoku_ai/ppo.py](/d:/code/gomoku_ai/gomoku_ai/ppo.py)

当前已实现：

- Residual Actor-Critic 网络。
- `225` 维策略输出。
- 标量 value head。
- masked categorical 动作分布。
- PPO + GAE 训练。
- checkpoint 中保存模型结构元数据；恢复时支持从 checkpoint 反推模型配置。

当前模型预设以 [gomoku_ai/model/presets.toml](/d:/code/gomoku_ai/gomoku_ai/model/presets.toml) 为准：

- `small`: `48 channels / 3 blocks`
- `base`: `64 channels / 4 blocks`
- `large`: `128 channels / 10 blocks`
- `custom`: 从训练参数显式覆盖

### 训练、评估与 run 管理

核心文件：

- [scripts/train.py](/d:/code/gomoku_ai/scripts/train.py)
- [scripts/evaluate.py](/d:/code/gomoku_ai/scripts/evaluate.py)
- [scripts/profile_env.py](/d:/code/gomoku_ai/scripts/profile_env.py)
- [scripts/list_runs.py](/d:/code/gomoku_ai/scripts/list_runs.py)
- [gomoku_ai/run_registry.py](/d:/code/gomoku_ai/gomoku_ai/run_registry.py)

当前已实现：

- 基于 TOML 的训练配置。
- 训练时打印“实际生效策略摘要”。
- checkpoint / best model / final model / resume。
- run manifest 和 `runs/index.json`。
- 每个 run 的 `history/` 持续记录：
  - `updates.log`
  - `updates.jsonl`
  - `updates.csv`
  - `evals.jsonl`
  - `training_strategy.json`
- 支持多 seed 聚合评估。
- 评估支持：
  - model vs bot
  - bot vs bot
- 评估导出支持：
  - JSON record
  - text report
  - visual JSON
- 提供环境 benchmark / cProfile / training-profile 三类 profiling。

需要注意的真实实现细节：

- `scripts/train.py` 当前固定把训练环境组织成 `8` 个 worker、每个 worker `2` 个 env，并最终覆盖 `n_envs = 16`。
- 因此命令行 `--n-envs` 当前不会像旧文档写的那样真正决定最终并行环境数。
- 当前训练主环境是 [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py) 中的 `SubprocVectorEnv`。

### CLI

核心文件：

- [tools/cli.py](/d:/code/gomoku_ai/tools/cli.py)

当前 `gmkt` 子命令包括：

- `help`
- `cleanup-latest-run`
- `cleanup-artifacts`
- `test`
- `reward-table`
- `evaluate`
- `replay`

## 当前验证状态

### 测试

`2026-03-18` 在本地执行：

```powershell
python -m pytest -q
```

结果不是全绿，而是：

- `34 passed`
- `4 failed`

当前失败项：

1. [tests/test_cli_evaluate.py](/d:/code/gomoku_ai/tests/test_cli_evaluate.py)
   现有实现默认 bot 是 `reward_driven_hard`，测试仍在断言旧值 `rule`。
2. [tests/test_evaluate_exports.py](/d:/code/gomoku_ai/tests/test_evaluate_exports.py)
   `export_match_outputs()` 签名已经调整，测试仍按旧接口传 `checkpoint_path` 和 `device`。
3. [tests/test_model_config.py](/d:/code/gomoku_ai/tests/test_model_config.py)
   测试断言 `large.blocks == 8`，但当前预设文件实际是 `10`。
4. [tests/test_model_config.py](/d:/code/gomoku_ai/tests/test_model_config.py)
   同一处旧断言导致 `build_model_config(...)` 相关 case 也失败。

结论：

- 代码主线已经演化；
- 测试里还有一部分旧假设没有同步；
- 当前状态不应再写成“测试全部通过”。

## 已知问题与风险

### 高优先级

1. 测试基线与当前实现不同步。
2. reward 接口还没有完全收敛到最终极简契约。
3. `scripts/train.py` 中 `--n-envs` 的实际语义与文档直觉不一致，容易误导使用者。

### 中优先级

4. profiling 脚本仍主要使用 `VectorEnv`，与训练默认使用的 `SubprocVectorEnv` 不完全一致。
5. 评估、回放、reward 可视化等周边工具已经能用，但部分文档仍沿用旧 bot 命名和旧 reward 叙述。

### 低优先级

6. 旧的 Python 参考函数仍保留在 [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py) 中，容易让新接手者误判主路径。

## 建议的下一步

如果现在继续维护这个仓库，优先顺序建议是：

1. 先把失败测试修到与当前实现一致，恢复 CI/本地回归基线。
2. 明确 `train.py` 中并行环境参数的真实配置来源，决定是暴露可配还是固定写死。
3. 把 reward 对外接口收敛到最终契约，评估是否移除默认返回中的 `special_rewards`。
4. 再决定是否继续推进更深层的 C++ 下沉或 reward 事件精化。

## 建议接手顺序

建议优先阅读：

1. [README.md](/d:/code/gomoku_ai/README.md)
2. [dev/PROJECT_STATUS.md](/d:/code/gomoku_ai/dev/PROJECT_STATUS.md)
3. [dev/COMMON_COMMANDS.md](/d:/code/gomoku_ai/dev/COMMON_COMMANDS.md)
4. [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py)
5. [cpp/RewardEvaluator.cpp](/d:/code/gomoku_ai/cpp/RewardEvaluator.cpp)
6. [scripts/train.py](/d:/code/gomoku_ai/scripts/train.py)
7. [scripts/evaluate.py](/d:/code/gomoku_ai/scripts/evaluate.py)
