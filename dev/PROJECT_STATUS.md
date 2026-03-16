# 项目进度记录

## 项目目标

这个仓库当前的目标不是直接做出最强五子棋 AI，而是把基于 PPO 的训练闭环稳定下来，并为后续迭代保留可分析、可扩展的工程结构。

当前主线目标：

1. 环境逻辑正确
2. 奖励设计可训练
3. PPO 训练流程可持续运行
4. run / checkpoint / 评估 / 历史数据链路完整
5. 模型结构、配置和工具链便于后续继续迭代

当前阶段仍然是 Python-first 原型，不是最终高性能版本。

## 当前已实现

### 环境与规则

核心文件：

- [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py)
- [gomoku_ai/rule_bot.py](/d:/code/gomoku_ai/gomoku_ai/rule_bot.py)

已实现：

- `15x15` 五子棋环境
- 单智能体对战内置 Bot
- 合法动作掩码
- 非法落子直接判负
- 五连、活四、冲四、活三、眠三、活二等棋形识别
- 基于落子前后 threat 变化的 reward 计算
- `VectorEnv` 批量采样接口

说明：

- 训练环境目前仍是单进程串行 step，不是真正并行
- 当前训练瓶颈主要还是环境与 reward 计算

### 模型与 PPO

核心文件：

- [gomoku_ai/model/network.py](/d:/code/gomoku_ai/gomoku_ai/model/network.py)
- [gomoku_ai/model/presets.toml](/d:/code/gomoku_ai/gomoku_ai/model/presets.toml)
- [gomoku_ai/ppo.py](/d:/code/gomoku_ai/gomoku_ai/ppo.py)

已实现：

- Residual Actor-Critic 网络
- 策略头输出 `225` 维动作 logits
- 价值头输出标量 `V(s)`
- masked categorical 动作分布
- 同步 PPO 更新
- GAE 优势估计
- TensorBoard 指标记录
- rollout / optimize / fps 统计

模型配置能力：

- 模型代码已迁移到 `gomoku_ai/model/` 包
- 预设模型放在 [gomoku_ai/model/presets.toml](/d:/code/gomoku_ai/gomoku_ai/model/presets.toml)
- 支持 `small / base / large / custom`
- `custom` 时才读取 TOML 里的显式模型参数
- checkpoint 可保存并恢复模型结构
- 旧 checkpoint 无结构元数据时可从 `state_dict` 推断

### 训练与评估

核心文件：

- [scripts/train.py](/d:/code/gomoku_ai/scripts/train.py)
- [scripts/evaluate.py](/d:/code/gomoku_ai/scripts/evaluate.py)

已实现：

- 基于 TOML 的训练配置
- 命令行覆盖关键超参
- 启动前打印实际生效的训练策略摘要
- 多 seed 聚合评估
- 周期 checkpoint
- `best_model`
- `final_model`
- `resume-from` 恢复训练

训练历史记录：

- 每个 run 下会写出 `history/`
- `updates.log`：控制台摘要行
- `updates.jsonl`：每次 update 的结构化训练数据
- `updates.csv`：便于表格和脚本分析
- `evals.jsonl`：每次评估结果
- `training_strategy.json`：本次 run 的生效配置快照

### run 管理与工具

核心文件：

- [gomoku_ai/run_registry.py](/d:/code/gomoku_ai/gomoku_ai/run_registry.py)
- [scripts/list_runs.py](/d:/code/gomoku_ai/scripts/list_runs.py)
- [tools/cli.py](/d:/code/gomoku_ai/tools/cli.py)
- [setup_env.ps1](/d:/code/gomoku_ai/setup_env.ps1)

已实现：

- 所有训练产物统一落到 `runs/<run_name>/`
- 每个 run 自动生成：
  - `manifest.json`
  - `latest_eval.json`
  - `checkpoints/`
  - `history/`
  - `tensorboard/`
  - `final_model.pt`
- `runs/index.json` 统一维护 run 索引
- `gmkt` 项目工具入口

当前 CLI 命令包括：

- `cleanup-latest-run`
- `cleanup-artifacts`
- `evaluate`
- `test`
- `reward-table`

### 配置外置

核心文件：

- [configs/train.toml](/d:/code/gomoku_ai/configs/train.toml)
- [configs/custom/train_large.toml](/d:/code/gomoku_ai/configs/custom/train_large.toml)
- [configs/reward.toml](/d:/code/gomoku_ai/configs/reward.toml)

已实现：

- PPO 训练超参外置
- 评估策略外置
- checkpoint 策略外置
- run 产物目录策略外置
- reward 参数外置
- 模型 preset 与 custom 参数外置

### 测试

核心目录：

- [tests](/d:/code/gomoku_ai/tests)

当前已覆盖：

- 环境与棋形识别
- CLI evaluate 参数
- evaluate 导出格式
- replay record 兼容性
- run layout
- 模型配置推断与 checkpoint 兼容
- cleanup-artifacts 行为

说明：

- 仓库内已有 `pytest` 测试文件
- 但本地当前会话环境未必已经安装 `pytest` / `tensorboard`

## 当前训练现状

目前模型可以稳定学会击败 `random` bot，但对 `rule` bot 仍然偏弱。

当前观察：

- 大模型配置已能正确生效，不再误回落到小模型
- 训练日志和历史数据已经足够支持离线分析
- 一些 run 会出现对 `random` 胜率高、对 `rule` 胜率长期为 `0` 的情况

这说明当前主要问题不是“模型根本不会下棋”，而是：

- 训练时长不够
- 对手分布偏强
- reward 虽然能学到基础落子，但还不足以稳定学出对抗 rule bot 的策略

## 已知瓶颈

### 高优先级

1. 训练质量分析不足自动化

- 现在 history 已经落盘，但还缺少专门的分析脚本
- 需要更系统地对 `win_rate / entropy / value_loss / reward components` 做回顾

2. 训练对手过于单一

- 当前主要仍围绕 rule bot
- 建议后续增加 curriculum 或混合对手

3. 环境吞吐仍偏低

- rollout 时间显著高于 optimize 时间
- 当前瓶颈主要仍在环境与规则逻辑，而不是网络反向传播

### 中优先级

4. 更丰富的评估基线

- 当前主要评估对象是 `rule` 和 `random`
- 还缺中间难度对手

5. 训练监控继续增强

- 当前已经有 history
- 后续可增加更多统计摘要和自动诊断

### 低优先级

6. C++ 下沉更多环境热路径

- 现有 C++ backend 已存在
- 但是否继续下沉，最好先基于 profiling 再做决定

## 当前建议的下一步

如果新的 AI 接手，这一轮最值得优先做的是：

1. 基于 `runs/<run>/history/` 做训练结果分析脚本
2. 对比 `small / base / large / custom` 的实际学习效率
3. 引入混合对手或 curriculum 训练
4. 再决定 reward 调整还是环境性能优化优先

不建议直接跳到：

- 全量 C++ 重写环境
- 直接引入 MCTS
- 在训练主链路里加入复杂搜索

## 接手建议

建议优先阅读：

1. [README.md](/d:/code/gomoku_ai/README.md)
2. [dev/PROJECT_STATUS.md](/d:/code/gomoku_ai/dev/PROJECT_STATUS.md)
3. [dev/COMMON_COMMANDS.md](/d:/code/gomoku_ai/dev/COMMON_COMMANDS.md)
4. [configs/train.toml](/d:/code/gomoku_ai/configs/train.toml)
5. [gomoku_ai/model/network.py](/d:/code/gomoku_ai/gomoku_ai/model/network.py)
6. [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py)
7. [gomoku_ai/ppo.py](/d:/code/gomoku_ai/gomoku_ai/ppo.py)

建议优先执行：

```powershell
python -m pytest -q
```

```powershell
python scripts/train.py --config configs/train.toml
```

```powershell
python scripts/list_runs.py --limit 20
```

只有在 run 历史和 profiling 结果都比较明确后，再决定下一步做：

- reward 调整
- 对手策略分层
- 环境热路径优化
- C++ 继续下沉
