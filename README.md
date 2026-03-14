# Gomoku AI Initial Prototype

This repository contains the first runnable prototype for a Gomoku reinforcement learning agent.

## Included

- `gomoku_ai/env.py`: single-agent environment against a rule-based bot
- `gomoku_ai/rule_bot.py`: baseline opponent
- `gomoku_ai/model.py`: lightweight residual actor-critic network
- `gomoku_ai/ppo.py`: synchronous PPO trainer
- `scripts/train.py`: local training entrypoint

## Run

```bash
python scripts/train.py --config configs/train.toml
```

By default, each training run writes all outputs into its own directory under `runs/`, for example:

```text
runs/20260314_213500/
  manifest.json
  final_model.pt
  latest_eval.json
  checkpoints/
  tensorboard/
runs/index.json
```

The training behavior is controlled by [configs/train.toml](/d:/code/gomoku_ai/configs/train.toml). The file uses comments to explain every variable, including:
- PPO hyperparameters
- Evaluation cadence
- Run output locations
- Whether to keep `best_model.pt`
- Whether to keep `final_model.pt`
- How many recent periodic checkpoints to retain
- Whether milestone checkpoints are never deleted

TensorBoard:

```bash
tensorboard --logdir runs/20260314_213500/tensorboard
```

Evaluation:

```bash
python scripts/evaluate.py --checkpoint runs/20260314_213500/final_model.pt --games 20 --num-seeds 3
```

Training with periodic checkpointing and aggregated evaluation:

```bash
python scripts/train.py --config configs/train.toml --run-name exp001
```

Resume training from a checkpoint:

```bash
python scripts/train.py --config configs/train.toml --resume-from runs/20260314_213500/checkpoints/checkpoint_update_0010.pt
```

List tracked runs:

```bash
python scripts/list_runs.py --limit 20
```

Cleanup the latest run directory:

```bash
python tools/cli.py cleanup-latest-run --yes
```

Project-local `tools` entrypoint:

```powershell
.\setup_env.ps1
gmkt help
gmkt cleanup-latest-run --yes
```

## Notes

- This is a Python-first prototype intended to validate training flow.
- The environment API is intentionally simple so it can be replaced by a C++ backend later.
