# Gomoku AI Workshop

This branch is organized as a teaching-oriented reinforcement learning workshop.
The same environment and PPO core are reused across stages, while each training
stage has its own entry script.

## Stages

```text
V0  scripts/demo_v0_env.py        environment reset/step/action-mask demo
V1  scripts/train_v1_terminal.py  terminal win/loss reward, no action mask
V2  scripts/train_v2_masked.py    terminal reward with action mask
V3  scripts/train_v3_shaped.py    action mask plus C++ pattern reward shaping
V4  scripts/train_v4_anneal.py    shaped reward with alpha annealing
V5  scripts/train_v5_mixed.py     mixed opponent pool
V6  scripts/train_v6_selfplay.py  mixed bots plus optional historical checkpoint opponents
```

V3 and later use the C++ backend and stronger bots:

```powershell
python -m pip install -r requirements.txt
```

When this repository is checked out as part of Workshop, use the Workshop root
virtual environment and run the command above from the Workshop root. Do not
create a separate virtual environment only for this submodule.

## Smoke Run

```powershell
python gomoku_ai/scripts/demo_v0_env.py
python gomoku_ai/scripts/train_v1_terminal.py --n-envs 1 --n-steps 4 --updates 1 --batch-size 4 --epochs 1 --no-progress
python gomoku_ai/scripts/evaluate.py --checkpoint runs/v1_terminal/<run>/final_model.pt --games 10 --bots random
```

Historical checkpoint opponents can be added in V6:

```powershell
python gomoku_ai/scripts/train_v6_selfplay.py --opponent-checkpoint runs/v5_mixed/<run>/final_model.pt
```

## Policy Viewer

Use the GUI policy viewer to inspect a checkpoint interactively:

```powershell
python gomoku_ai/tools/policy_viewer.py --checkpoint runs/v3_shaped/<run>/final_model.pt --bot classic_rule
```

The viewer shows model policy probabilities as a heatmap on the board and lets
you step through model-vs-bot games.

## Tests

```powershell
pytest -q
```
