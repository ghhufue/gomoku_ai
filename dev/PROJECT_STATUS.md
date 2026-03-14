# 项目进度记录

## 项目目标

本项目正在实现一个基于 PPO 的五子棋 AI 原型。当前主线目标不是追求最终最强棋力，而是先把下面这些能力做稳定：

1. 环境逻辑正确
2. 奖励机制可训练
3. PPO 训练闭环可运行
4. checkpoint / run / 评估链路完整
5. 后续能平滑过渡到更高性能实现

当前阶段仍然是 Python-first 原型阶段，不是最终高性能版本。

## 当前已实现

### 环境与规则

核心文件：
- [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py)
- [gomoku_ai/rule_bot.py](/d:/code/gomoku_ai/gomoku_ai/rule_bot.py)

已实现：
- `15x15` 五子棋环境
- 单智能体对战规则 Bot
- 合法动作掩码
- 非法落子直接判负
- 胜负判断
- 棋型识别：
  - 五连
  - 活四
  - 冲四
  - 活三
  - 眠三
  - 活二
- 基于“落子前后 threat 变化”的奖励逻辑

说明：
- 规则逻辑已明显强于最初版本，但 reward 和 threat 计算仍然偏重，现阶段是性能瓶颈候选。

### 模型与 PPO

核心文件：
- [gomoku_ai/model.py](/d:/code/gomoku_ai/gomoku_ai/model.py)
- [gomoku_ai/ppo.py](/d:/code/gomoku_ai/gomoku_ai/ppo.py)

已实现：
- 轻量 ResNet Actor-Critic
- 动作掩码策略分布
- 同步 PPO 训练器
- TensorBoard 指标记录
- `tqdm` 训练进度显示
- rollout / optimize 耗时和吞吐输出

近期重要变更：
- 已去掉 value head 最后的 `Tanh`
原因：
  当前 reward 尺度到 `±1000`，critic 输出限制在 `[-1, 1]` 不合理，会导致异常巨大的 `value_loss`

### 训练与评估脚本

核心文件：
- [scripts/train.py](/d:/code/gomoku_ai/scripts/train.py)
- [scripts/evaluate.py](/d:/code/gomoku_ai/scripts/evaluate.py)

已实现：
- 基于配置文件启动训练
- 支持 CLI 局部覆盖
- 聚合评估
- 多 seed 评估
- 训练中周期 checkpoint
- `best_model`
- `final_model`
- `resume-from` 恢复训练

### run 管理与工具

核心文件：
- [gomoku_ai/run_registry.py](/d:/code/gomoku_ai/gomoku_ai/run_registry.py)
- [scripts/list_runs.py](/d:/code/gomoku_ai/scripts/list_runs.py)
- [tools/cli.py](/d:/code/gomoku_ai/tools/cli.py)
- [setup_env.ps1](/d:/code/gomoku_ai/setup_env.ps1)

已实现：
- 训练产物统一落在 `runs/<run_name>/`
- 每个 run 自动生成：
  - `manifest.json`
  - `latest_eval.json`
  - `checkpoints/`
  - `tensorboard/`
  - `final_model.pt`
- `runs/index.json` 统一维护 run 索引
- `gmkt` 当前会话工具入口
- 已有工具命令：
  - `cleanup-latest-run`

### 配置外置

核心文件：
- [configs/train.toml](/d:/code/gomoku_ai/configs/train.toml)
- [configs/train_exp01.toml](/d:/code/gomoku_ai/configs/train_exp01.toml)

已实现：
- 训练超参外置
- checkpoint 保留策略外置
- 输出路径策略外置
- 配置注释已改成中文

### 测试

核心文件：
- [tests/test_env.py](/d:/code/gomoku_ai/tests/test_env.py)

当前测试已覆盖：
- 五连识别
- 活四识别
- 冲四识别
- 活三识别
- 眠三识别
- 双活三识别
- 立即取胜点检测
- threat summary
- 关键防守奖励
- 活二奖励边界
- 非终局奖励上界
- 非法落子
- 终局行为

当前状态：
- `17 passed`

## 当前训练现状

已经开始尝试正式训练配置：
- [configs/train_exp01.toml](/d:/code/gomoku_ai/configs/train_exp01.toml)

目前明确观察到：
- `optimize` 时间很短
- `rollout` 时间极长
- CPU 基本单核忙
- GPU 利用率低

这说明当前训练瓶颈不是网络训练，而是环境采样。

已知现象：
- rollout 可能达到几十秒
- optimize 可能只有不到 1 秒

这基本确认当前主要瓶颈在：
- `env.step()`
- reward 计算
- Rule Bot 决策
- 当前 `VectorEnv` 只是串行批处理，不是真并行

## 新增的 profiling 能力

新增文件：
- [scripts/profile_env.py](/d:/code/gomoku_ai/scripts/profile_env.py)

用途：
- benchmark 模式：
  测环境步进整体耗时和 `steps_per_sec`
- cProfile 模式：
  打印 Python 调用层的累计耗时热点，并可保存 `.prof`

推荐命令：

```powershell
.\.venv\Scripts\python.exe scripts\profile_env.py --mode benchmark --steps 200
```

```powershell
.\.venv\Scripts\python.exe scripts\profile_env.py --mode cprofile --steps 200 --top-k 30 --profile-out artifacts\env_profile.prof
```

当前还没有基于 profiling 结果做热点重构，下一位 AI 应优先跑这份脚本，拿到定量结论。

## 当前最重要的未完成事项

### 高优先级

1. 环境性能 profiling
目标：
  确认真正最慢的是哪几个函数，而不是只凭体感判断

2. 环境热点优化
可能候选：
  - `threat_summary(...)`
  - `immediate_winning_actions(...)`
  - `classify_move(...)`
  - `evaluate_shape_reward(...)`
  - `RuleBasedBot.select_action(...)`

3. 训练吞吐优化
当前 `n_envs` 是串行管理，不是真并行
后续需要评估：
  - 多进程环境
  - Python 层减重
  - C++ 下沉热点

### 中优先级

4. Rule Bot 难度分级
当前 Bot 偏基础，只适合早期训练与基线评估

5. 更丰富的评估基线
目前主要评估对象仍是 Rule Bot

6. 更完整的训练监控
后续可以补：
  - value prediction 统计
  - return 统计
  - reward 分项统计

### 低优先级

7. C++ 环境内核
现在已经可以开始设计，但不建议直接整套重写
更合理路线：
  - 先 profiling
  - 再选热点
  - 再下沉 reward / pattern / state core

## 当前建议的下一步

如果新的 AI 接手，这一轮最推荐做的事情是：

1. 跑环境 profiling
2. 用结果确认热点
3. 优先优化最重的 Python 逻辑
4. 再决定是否开始 C++ 下沉

不建议下一位 AI 一上来就直接：
- 全量 C++ 重写环境
- 引入 MCTS
- 引入 Alpha-Beta 到训练主链路

## 给下一个 AI 的接手说明

建议优先阅读：

1. [dev/PROJECT_STATUS.md](/d:/code/gomoku_ai/dev/PROJECT_STATUS.md)
2. [README.md](/d:/code/gomoku_ai/README.md)
3. [configs/train.toml](/d:/code/gomoku_ai/configs/train.toml)
4. [configs/train_exp01.toml](/d:/code/gomoku_ai/configs/train_exp01.toml)
5. [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py)
6. [gomoku_ai/ppo.py](/d:/code/gomoku_ai/gomoku_ai/ppo.py)
7. [tests/test_env.py](/d:/code/gomoku_ai/tests/test_env.py)

建议优先执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
.\.venv\Scripts\python.exe scripts\profile_env.py --mode benchmark --steps 200
```

```powershell
.\.venv\Scripts\python.exe scripts\profile_env.py --mode cprofile --steps 200 --top-k 30
```

只有在 profiling 结果明确后，再开始决定：
- 先做 Python 热点优化
- 还是开始 C++ 环境核心下沉
