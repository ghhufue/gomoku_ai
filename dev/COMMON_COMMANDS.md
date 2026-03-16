# 常用命令

这份文件记录当前项目里最常用的训练、评估、模拟、回放和测试命令，方便重复使用。

## 环境

如果还没激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果不想激活，也可以直接用：

```powershell
.\.venv\Scripts\python.exe <脚本>
```

## 训练

使用默认训练配置启动训练：

```powershell
python scripts/train.py --config configs/train.toml
```

指定设备：

```powershell
python scripts/train.py --config configs/train.toml --device cuda
```

覆盖训练轮数：

```powershell
python scripts/train.py --config configs/train.toml --updates 1000
```

覆盖并行环境数：

```powershell
python scripts/train.py --config configs/train.toml --n-envs 8
```

## 恢复训练

从周期 checkpoint 恢复：

```powershell
python scripts/train.py --config configs/train.toml --resume-from runs\<run_name>\checkpoints\checkpoint_update_0200.pt
```

从最终模型恢复：

```powershell
python scripts/train.py --config configs/train.toml --resume-from runs\<run_name>\final_model.pt
```

说明：

- `updates` 表示在当前恢复点基础上再训练多少个 update
- 恢复训练会继续写回原来的 `runs\<run_name>\` 目录

## 评估

使用 `gmkt evaluate` 评估指定权重：

```powershell
python tools/cli.py evaluate --checkpoint runs\<run_name>\final_model.pt
```

如果已经把 `gmkt` 配好了，也可以直接：

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt
```

指定局数和 seed 数：

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt --games 50 --num-seeds 3
```

指定对手：

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt --bot rule
```

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt --bot random
```

不导出可视化文件：

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt --no-export-visual
```

不导出任何对局文件，只看胜率：

```powershell
gmkt evaluate --checkpoint runs\<run_name>\final_model.pt --no-export-record --no-export-text --no-export-visual
```

说明：

- 默认会导出到 `outputs/evaluation/<时间戳_模型名>/`
- 默认会导出：
  - `match_record.json`
  - `match_record.txt`
  - `match_record_visual.json` 或 `match_record_gameXX_visual.json`

## 模拟 / 对局导出

`play_match.py` 现在是 `evaluate.py` 的兼容包装器，默认会导出对局记录：

```powershell
python scripts/play_match.py --checkpoint runs\<run_name>\final_model.pt --games 1
```

多局导出：

```powershell
python scripts/play_match.py --checkpoint runs\<run_name>\final_model.pt --games 10
```

## 对局回放

回放评估导出的第 1 局：

```powershell
python tools/replay_record.py --record outputs\evaluation\<export_dir>\match_record.json --game 1
```

回放第 50 局：

```powershell
python tools/replay_record.py --record outputs\evaluation\<export_dir>\match_record.json --game 50
```

回放内置样例：

```powershell
python tools/replay_record.py --name block_live_three
```

列出内置样例：

```powershell
python tools/replay_record.py --list
```

## 棋形浏览

浏览内置棋形 reward：

```powershell
python tools/replay_shapes.py
```

指定 reward 配置：

```powershell
python tools/replay_shapes.py --reward configs/reward.toml
```

列出棋形案例：

```powershell
python tools/replay_shapes.py --list
```

## 测试

运行全部环境测试：

```powershell
python -m pytest tests/test_env.py -q
```

运行全部测试：

```powershell
python -m pytest -q
```

通过 CLI 运行测试：

```powershell
gmkt test
```

列出测试 case：

```powershell
gmkt test --list-cases
```

运行单个编号测试：

```powershell
gmkt test --case-id 16
```

## Reward 配置表

把 reward TOML 转成 Markdown 表格：

```powershell
gmkt reward-table --reward configs/reward.toml
```

输出会落到：

```text
outputs/visual/reward_table.md
```

## run 管理

清理最近一次 run：

```powershell
gmkt cleanup-latest-run --yes
```

查看可用命令帮助：

```powershell
gmkt help
```

```powershell
gmkt evaluate --help
```

```powershell
gmkt test --help
```
