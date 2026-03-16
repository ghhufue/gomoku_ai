# Gomoku AI

这个项目主要用于 workshop 教学，帮助学习者通过一个可运行、可观察、可分析的案例，熟悉强化学习的基本流程，理解 PPO 算法的训练闭环，并练习奖惩机制的设计与迭代。

它的目标不是训练出最强的五子棋 AI。恰恰相反，五子棋在这里更像一个直观的实验载体: 棋局状态清晰、动作空间明确、胜负反馈直接，学习者可以比较容易地看到模型从“不会下”到“有一定策略”的变化过程。

也需要明确一点: 从算法选择上看，PPO 本身并不是解决五子棋这类棋盘博弈问题的最佳方案。对于这类任务，搜索、专家先验、甚至更贴近自博弈体系的方法通常更合适。这里选择 PPO，不是因为它最强，而是因为它足够经典、结构清晰，适合作为教学场景中理解策略优化、价值估计、优势函数和 reward shaping 的入口。

## 主要组件

- [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py): 五子棋环境、棋形识别、reward 计算、`VectorEnv`
- [bots/reward_driven_bot.py](/d:/code/gomoku_ai/bots/reward_driven_bot.py): reward 驱动规则对手
- [configs/bots.toml](/d:/code/gomoku_ai/configs/bots.toml): bot 注册表、难度映射与能力说明
- [gomoku_ai/model/network.py](/d:/code/gomoku_ai/gomoku_ai/model/network.py): Actor-Critic 网络与模型配置恢复
- [gomoku_ai/model/presets.toml](/d:/code/gomoku_ai/gomoku_ai/model/presets.toml): `small / base / large` 模型预设
- [gomoku_ai/ppo.py](/d:/code/gomoku_ai/gomoku_ai/ppo.py): PPO 与 GAE 实现
- [scripts/train.py](/d:/code/gomoku_ai/scripts/train.py): 训练入口
- [scripts/evaluate.py](/d:/code/gomoku_ai/scripts/evaluate.py): 评估与对局导出
- [tools/cli.py](/d:/code/gomoku_ai/tools/cli.py): `gmkt` 命令入口

## 模型配置

模型支持四种 preset：

- `small`
- `base`
- `large`
- `custom`

预设参数定义在 [gomoku_ai/model/presets.toml](/d:/code/gomoku_ai/gomoku_ai/model/presets.toml)。

训练时的规则是：

- `small / base / large` 直接使用预设参数
- 只有 `custom` 才会读取 `train.toml` 里的 `channels / blocks / policy_channels / value_channels / value_hidden_dim`

当前预设：

```toml
[presets.small]
channels = 48
blocks = 3
policy_channels = 2
value_channels = 1
value_hidden_dim = 96

[presets.base]
channels = 64
blocks = 4
policy_channels = 2
value_channels = 1
value_hidden_dim = 128

[presets.large]
channels = 128
blocks = 10
policy_channels = 4
value_channels = 2
value_hidden_dim = 256
```

## 训练

默认训练：

```bash
python scripts/train.py --config configs/train.toml
```

使用模型预设：

```bash
python scripts/train.py --config configs/train.toml --model-preset small
python scripts/train.py --config configs/train.toml --model-preset base
python scripts/train.py --config configs/train.toml --model-preset large
```

使用 custom 模型：

```bash
python scripts/train.py --config configs/train.toml --model-preset custom --model-channels 160 --model-blocks 7 --policy-channels 5 --value-channels 3 --value-hidden-dim 320
```

更大的训练配置示例：

```bash
python scripts/train.py --config configs/custom/train_large.toml
```

其他常见覆盖参数：

```bash
python scripts/train.py --config configs/train.toml --device cuda
python scripts/train.py --config configs/train.toml --updates 1000
python scripts/train.py --config configs/train.toml --n-envs 8
python scripts/train.py --config configs/train.toml --resume-from runs/20260314_213500/checkpoints/checkpoint_update_0010.pt
```

训练开始前，脚本会先打印一次“实际生效”的训练策略摘要，明确输出：

- 设备、seed、run 目录
- 实际模型 preset 和结构参数
- PPO 关键超参
- 评估策略
- checkpoint 策略
- 产物路径
- resume 信息

## run 产物

每次训练都会在 `runs/` 下创建独立目录，例如：

```text
runs/20260314_213500/
  manifest.json
  final_model.pt
  latest_eval.json
  checkpoints/
  history/
  tensorboard/
runs/index.json
```

其中：

- `manifest.json`: 当前 run 的完整配置、路径和状态
- `final_model.pt`: 最终模型
- `latest_eval.json`: 最近一次评估结果
- `checkpoints/`: 周期 checkpoint 与 `best_model.pt`
- `tensorboard/`: TensorBoard 日志
- `history/`: 训练中间统计与分析数据

`history/` 当前包含：

- `updates.log`: 控制台摘要行
- `updates.jsonl`: 每个 update 的结构化训练数据
- `updates.csv`: 便于脚本和表格分析
- `evals.jsonl`: 每次评估结果
- `training_strategy.json`: 本次 run 的生效配置快照

## 评估

直接评估 checkpoint：

```bash
python scripts/evaluate.py --checkpoint runs/20260314_213500/final_model.pt --games 20 --num-seeds 3
```

可选对手：

```bash
python scripts/evaluate.py --checkpoint runs/20260314_213500/final_model.pt --bot rule
python scripts/evaluate.py --checkpoint runs/20260314_213500/final_model.pt --bot random
```

默认评估会导出 replay 兼容的对局记录到 `outputs/evaluation/`。

## 回放与导出

回放评估导出的对局：

```bash
python tools/replay_record.py --record outputs/evaluation/<export_dir>/match_record.json --game 1
python tools/replay_record.py --record outputs/evaluation/<export_dir>/match_record.json --game 50
```

回放内置样例：

```bash
python tools/replay_record.py --name block_live_three
python tools/replay_record.py --list
```

查看内置棋形 reward：

```bash
python tools/replay_shapes.py
python tools/replay_shapes.py --reward configs/reward.toml
python tools/replay_shapes.py --list
```

## `gmkt` CLI

如果已经通过 [setup_env.ps1](/d:/code/gomoku_ai/setup_env.ps1) 配好本地命令入口，可以直接使用：

```powershell
.\setup_env.ps1
gmkt help
```

当前常用命令：

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt
gmkt replay --record outputs\evaluation\<export_dir>\match_record.json --game 1
gmkt replay --name block_live_three
gmkt reward-table --reward configs/reward.toml
gmkt cleanup-latest-run --yes
gmkt cleanup-artifacts --yes
gmkt test
```

## TensorBoard

启动 TensorBoard：

```bash
tensorboard --logdir runs/20260314_213500/tensorboard
```

## 测试

运行全部测试：

```bash
python -m pytest -q
```

只跑环境测试：

```bash
python -m pytest tests/test_env.py -q
```

也可以通过 CLI：

```powershell
gmkt test
gmkt test --list-cases
gmkt test --case-id 16
```

## 配置文件

核心配置文件：

- [configs/train.toml](/d:/code/gomoku_ai/configs/train.toml): 默认训练配置
- [configs/custom/train_large.toml](/d:/code/gomoku_ai/configs/custom/train_large.toml): 更大训练配置示例
- [configs/reward.toml](/d:/code/gomoku_ai/configs/reward.toml): reward 配置

`train.toml` 当前包含：

- `runtime`
- `training`
- `model`
- `evaluation`
- `artifacts`
- `checkpoint`

## 当前已知情况

- 模型已经可以稳定学会击败 `random` bot
- 对 `rule` bot 的学习仍然偏难，往往需要更长训练、更合理的对手分布和更细致的训练分析
- 当前训练瓶颈主要还是 rollout 和环境逻辑，不是网络反向传播
- 训练历史已经可以完整落盘，适合后续做 AI/脚本分析

## 说明

- 当前仍是 Python-first 原型
- 环境 API 刻意保持简洁，便于后续继续下沉到 C++ 或替换实现
- 如果要快速了解当前工程状态，优先看：
  - [dev/PROJECT_STATUS.md](/d:/code/gomoku_ai/dev/PROJECT_STATUS.md)
  - [dev/COMMON_COMMANDS.md](/d:/code/gomoku_ai/dev/COMMON_COMMANDS.md)
