# 常用命令

更新时间：`2026-03-19`

这份文档只记录当前仓库里已经存在、并且命令参数与实现一致的常用操作。

## 环境

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

不激活时直接调用解释器：

```powershell
.\.venv\Scripts\python.exe <script>
```

## 训练

默认训练：

```powershell
python scripts/train.py --config configs/train.toml
```

指定模型预设：

```powershell
python scripts/train.py --config configs/train.toml --model-preset small
python scripts/train.py --config configs/train.toml --model-preset base
python scripts/train.py --config configs/train.toml --model-preset large
```

自定义模型结构：

```powershell
python scripts/train.py --config configs/train.toml --model-preset custom --model-channels 160 --model-blocks 7 --policy-channels 5 --value-channels 3 --value-hidden-dim 320
```

覆盖常见训练参数：

```powershell
python scripts/train.py --config configs/train.toml --device cuda
python scripts/train.py --config configs/train.toml --updates 1000
python scripts/train.py --config configs/train.toml --batch-size 256 --epochs 4 --lr 1e-4
python scripts/train.py --config configs/train.toml --bot classic_rule
python scripts/train.py --config configs/train.toml --bot-difficulty medium
```

按概率从预制状态开始训练：

```powershell
python scripts/train.py --config configs/train.toml --reset-state-path outputs\build\precompute\rush_four_group.json --reset-state-prob 0.3
python scripts/train.py --config configs/train.toml --reset-state-prob 0
```

从 checkpoint 恢复：

```powershell
python scripts/train.py --config configs/train.toml --resume-from runs\<run_name>\checkpoints\checkpoint_update_0200.pt
python scripts/train.py --config configs/train.toml --resume-from runs\<run_name>\final_model.pt
```

说明：

- 训练入口当前默认使用 `SubprocVectorEnv`。
- `scripts/train.py` 目前固定为 `8` 个 worker、每个 worker `2` 个 env，因此 `--n-envs` 不是完全按字面生效的用户开关。

## 列出 Run

```powershell
python scripts/list_runs.py --limit 20
```

## 评估

评估 checkpoint：

```powershell
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --games 20 --num-seeds 3
```

指定对手：

```powershell
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --bot random
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --bot classic_rule
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --bot reward_driven_hard
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --bot-difficulty medium
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --bot-difficulty hard
```

关闭导出：

```powershell
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --no-export-visual
python scripts/evaluate.py --checkpoint runs\<run_name>\final_model.pt --no-export-record --no-export-text --no-export-visual
```

Bot 对 Bot：

```powershell
python scripts/evaluate.py --bot-a classic_rule --bot-b reward_driven_hard
python scripts/evaluate.py --bot-a classic_rule --bot-b reward_driven_hard --games 10 --num-seeds 3
```

说明：

- 未指定 `--checkpoint` 时，普通评估模式会回退到 `runs/` 下最新的 `final_model.pt`。
- 默认导出目录是 `outputs/evaluation/<timestamp>_<label>/`。

## CLI

如果已经配置好本地入口：

```powershell
.\setup_env.ps1
gmkt help
```

常用命令：

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt --bot reward_driven_hard
gmkt replay --record outputs\evaluation\<export_dir>\match_record.json --game 1
gmkt reward-table --reward configs/reward.toml
gmkt cleanup-latest-run --yes
gmkt cleanup-artifacts --yes
gmkt test
```

查看帮助：

```powershell
gmkt help
gmkt evaluate --help
gmkt replay --help
gmkt test --help
```

## 回放

回放评估导出的对局：

```powershell
python tools/replay_record.py --record outputs\evaluation\<export_dir>\match_record.json --game 1
python tools/replay_record.py --record outputs\evaluation\<export_dir>\match_record.json --game 50
```

回放内置样例：

```powershell
python tools/replay_record.py --name block_live_three
python tools/replay_record.py --list
```

## Reward 可视化

把 reward TOML 生成成 Markdown 表格：

```powershell
python tools/cli.py reward-table --reward configs/reward.toml
gmkt reward-table --reward configs/reward.toml
```

输出位置默认是：

```text
outputs/visual/reward_table.md
```

## Profiling

训练 rollout / update profiling：

```powershell
python scripts/profile_env.py
python scripts/profile_env.py --presets small,base --updates 2
python scripts/profile_env.py --bot classic_rule
python scripts/profile_env.py --bot-difficulty medium
```

更小规模的 profiling：

```powershell
python scripts/profile_env.py --presets small --updates 1 --n-envs 1 --n-steps 4 --epochs 1 --batch-size 4 --device cpu
```

环境吞吐 benchmark：

```powershell
python scripts/profile_env.py --mode benchmark --steps 200
```

Python 调用级 cProfile：

```powershell
python scripts/profile_env.py --mode cprofile --steps 200 --top-k 30
python scripts/profile_env.py --mode cprofile --steps 200 --profile-out outputs/profile/env_step.prof
```

## 测试

运行全部测试：

```powershell
python -m pytest -q
```

运行环境测试：

```powershell
python -m pytest tests/test_env.py -q
```

运行 C++ 相关测试：

```powershell
python -m pytest tests/test_direction_encoding.py -q
python -m pytest tests/test_cpp_reward_events.py -q
python -m pytest tests/test_cpp_direction_pattern_lookup.py -q
```

通过 CLI 跑测试或调试样例：

```powershell
gmkt test
gmkt test --list-cases
gmkt test --case-id 16
gmkt test --case block_opponent_live_three
```

说明：

- 截至 `2026-03-18`，`python -m pytest -q` 的真实结果是 `34 passed, 4 failed`。
- 当前失败项主要集中在旧默认值、旧函数签名和旧模型预设断言没有同步。

## C++ 构建

构建 Python 扩展：

```powershell
python setup.py build_ext --inplace
```

构建 C++ 辅助工具：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_cpp_tools.ps1
```
