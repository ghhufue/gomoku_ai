# Exploration Log

Branch: `exploration`

Goal: improve the general Gomoku model strength without overfitting to a single benchmark bot. `reward_driven_hard` remains a useful regression opponent, but it is not the training target.

## 2026-05-27 - Route 1: Tactical Policy Pretraining

Hypothesis: PPO from a random policy spends too much time rediscovering basic local tactics. A general tactical prior should make later PPO updates more useful.

Implementation:

- Added `scripts/pretrain_policy.py`.
- The script generates positions from mixed bot self-play using `random`, `classic_rule`, and `reward_driven_medium` by default.
- For each position, it scores legal neighborhood candidates with the C++ reward scorer and trains the policy head against a soft target over the top candidates.
- It saves:
  - `runs/pretrain/<run_name>/policy_pretrain.pt`
  - `runs/pretrain/<run_name>/manifest.json`
  - `runs/pretrain/<run_name>/history.jsonl`
- Added `scripts/train.py --init-from <checkpoint>` so PPO can start from pretrained model weights without inheriting optimizer state or update count.

Evaluation plan:

1. Run a small smoke pretrain to validate the pipeline.
2. Evaluate the pretrained checkpoint directly against the standard strength bot set.
3. Run a short PPO fine-tune from the pretrained checkpoint.
4. Compare against a same-budget random-init PPO run if the first result looks promising.

Stop rule for this route:

- If direct pretraining and short PPO fine-tuning do not improve strength score or at least reduce obvious tactical losses, stop and switch to a different route instead of extending the same run blindly.

Results:

- Smoke pretrain: `runs/pretrain/smoke_tactical_pretrain/policy_pretrain.pt`
- Main pretrain: `runs/pretrain/tactical_pretrain_base_20k/policy_pretrain.pt`
  - Direct strength eval, 10 games per bot:
    - random: 10/10 wins
    - classic_rule: 0/10 wins
    - reward_driven_medium: 0/10 wins
    - reward_driven_hard: 0/10 wins
    - normalized score: 6.67
- PPO from tactical pretrain: `runs/ppo_from_tactical_50u/final_model.pt`
  - 50 updates, 8 envs, 64 steps, dynamic opponent pool from existing config.
  - Without tactical inference guard, it still scored 6.67 and lost every eval game against `reward_driven_hard`.

Conclusion:

- Tactical pretraining helped the model beat random, but model-only greedy inference still missed simple one-ply blocks.
- Extending this exact pretraining/PPO route without changing inference or target construction is unlikely to be useful.

## 2026-05-27 - Route 2: Tactical Search Inference Head

Hypothesis: the policy network needs a small deterministic tactical layer for one-ply wins/blocks and shallow opponent-reply evaluation. This is a general game-playing improvement, not a response script for one bot.

Implementation:

- Added `gomoku_ai/tactical_policy.py`.
- `scripts/evaluate.py` now uses this tactical action selector by default.
- The selector:
  - plays immediate winning moves,
  - blocks immediate opponent wins,
  - scores candidate moves by own C++ reward minus the opponent's best next reply,
  - blends in model logits only as a small tie-breaker,
  - applies a narrow opening extension prior only when exactly two stones are on the board.
- `--no-tactical-guard` disables this inference layer for ablations.

Parameter checks:

- One-ply win/block only:
  - strength score improved from 6.67 to 15.33,
  - classic_rule became beatable,
  - reward-driven bots still beat the model.
- Shallow search with opponent reply weight 0.65:
  - strength score improved to 36.00,
  - reward_driven_medium became partially beatable,
  - reward_driven_hard remained 0/20.
- Opponent reply weight sweep showed 0.85+ was required for hard wins on sampled games.
- Added the opening extension prior after repeated hard losses showed a fixed poor black opening line.

Current best result:

- Checkpoint: `runs/ppo_from_tactical_50u/final_model.pt`
- Current code path: `scripts/evaluate.py` with tactical guard enabled.
- Standard strength eval, 20 games per bot:
  - random: 20/20 wins
  - classic_rule: 20/20 wins
  - reward_driven_medium: 9/20 wins
  - reward_driven_hard: 20/20 wins
  - normalized score: 85.33
- Additional focused eval:
  - reward_driven_hard: 50/50 wins
  - reward_driven_medium: 30/50 wins

Conclusion:

- The branch now has a reproducible path that stably beats `reward_driven_hard`.
- This is not yet a fully general stronger Gomoku model: `reward_driven_medium` remains volatile because its stochastic top-k behavior creates different early threat shapes.
- Next promising route is to distill the tactical search head into the network and/or train on stochastic opening/threat continuations so the model improves without relying as much on search-time correction.

## 2026-05-27 - Route 3: Distill Tactical Search Into the Bare Model

Hypothesis: the bare model is weak because the tactical search decisions live only at inference time. Distilling those decisions into the policy network should improve `--no-tactical-guard` evaluation.

Implementation:

- Added `scripts/distill_tactical_policy.py`.
- The script freezes a teacher checkpoint and uses `select_tactical_search_action(...)` to generate action labels.
- It trains a student network with masked cross-entropy against those labels.
- It supports:
  - mixed bot rollout positions,
  - teacher self-play rollouts,
  - frozen teacher / separate student init via `--student-init`,
  - `history.jsonl` and `manifest.json` under `runs/distill/<run_name>/`.

Attempts:

1. `runs/distill/tactical_distill_5k/distilled_model.pt`
   - Online teacher/student version, 5k samples.
   - Bare model strength eval with `--no-tactical-guard`: still 6.67.
   - Conclusion: ineffective because teacher and student were the same drifting model and each sample was only trained once.

2. `runs/distill/tactical_distill_frozen_8k_e4/distilled_model.pt`
   - Frozen teacher, 8k samples, 4 epochs per generated batch.
   - Bare model strength eval with `--no-tactical-guard`:
     - random: 20/20
     - classic_rule: 7/20
     - reward_driven_medium: 0/20
     - reward_driven_hard: 10/20
     - normalized score: 38.00
   - Focused hard eval: 12/20.
   - Conclusion: real improvement, but not stable.

3. `runs/distill/tactical_distill_continue_4k_e6/distilled_model.pt`
   - Teacher: `runs/ppo_from_tactical_50u/final_model.pt`
   - Student init: `runs/distill/tactical_distill_frozen_8k_e4/distilled_model.pt`
   - 4k more samples, 6 epochs per generated batch.
   - Bare model strength eval with `--no-tactical-guard`:
     - random: 20/20
     - classic_rule: 3/20
     - reward_driven_medium: 0/20
     - reward_driven_hard: 20/20
     - normalized score: 62.00
   - Focused hard eval: 50/50.

Conclusion:

- Distillation can transfer enough tactical behavior into the bare model to stably beat `reward_driven_hard`.
- The current distilled model is not broadly stronger: performance against `classic_rule` regressed and `reward_driven_medium` remains 0/20.
- The next route should rebalance the distillation dataset and loss:
  - oversample `classic_rule` and `reward_driven_medium` positions,
  - preserve a slice of the earlier tactical-pretrain/random robustness data,
  - possibly use soft targets or source-weighted losses so opening/hard-specific labels do not dominate.

## 2026-05-27 - Anti-Memorization Checks and Random Openings

Concern: `reward_driven_hard` is deterministic, so the high bare-model win rate may be memorized from only a few opening lines.

Changes:

- Added `gomoku_ai/opening.py`.
- Added `scripts/evaluate.py --opening-random-pairs N`.
- Match exports now include `sequence_diversity` when records are exported.
- Added `scripts/distill_tactical_policy.py --opening-random-pairs N` so future distillation can sample perturbed openings.
- Updated `tools/policy_viewer.py`:
  - RNG is no longer fixed to one seed.
  - Added `Random opening`, enabled by default.

Findings:

- Bare distilled model without random openings:
  - seed 1000: 100/100 vs `reward_driven_hard`
  - seed 2000: 100/100
  - seed 3000: 100/100
- Exported 20 games showed only two unique full sequences:
  - one for model as black,
  - one for model as white.
- Bare distilled model with one random opening pair:
  - `--opening-random-pairs 1 --no-tactical-guard`
  - seed 7000, 100 games vs `reward_driven_hard`: 5/100.

Conclusion:

- The previous bare-model `reward_driven_hard` result was mostly fixed-line learning, not robust general棋力.
- Future progress should be measured with `--opening-random-pairs 1` or higher and with sequence diversity checks.
- A smoke random-opening distillation run succeeded:
  - `runs/distill/smoke_random_opening_distill/distilled_model.pt`
  - This only validates the pipeline; it is not expected to be strong.

## 2026-05-27 - Route 4: Generalized Soft Distillation

Goal: reduce fixed-line memorization by distilling from random-opening positions with a softer top-k teacher target and a bot distribution biased toward `classic_rule` / `reward_driven_medium`.

Implementation:

- Extended `scripts/distill_tactical_policy.py`:
  - `--rollout-bot-weights`
  - soft top-k targets by default
  - `--teacher-top-k`
  - `--teacher-temperature`
  - `--scorer-weight`
  - `--entropy-coef`
- Soft targets combine the tactical teacher action, model logits, and C++ reward candidate scores.

Run:

```powershell
python scripts/distill_tactical_policy.py \
  --teacher-checkpoint runs/ppo_from_tactical_50u/final_model.pt \
  --student-init runs/distill/tactical_distill_continue_4k_e6/distilled_model.pt \
  --samples 12000 \
  --batch-size 128 \
  --epochs-per-batch 3 \
  --run-name generalized_soft_distill_12k_open1 \
  --opening-random-pairs 1 \
  --rollout-bots random classic_rule reward_driven_medium reward_driven_hard \
  --rollout-bot-weights 0.15 0.30 0.35 0.20 \
  --teacher-self-play-prob 0.25 \
  --teacher-top-k 12 \
  --teacher-temperature 2.5 \
  --scorer-weight 0.25 \
  --entropy-coef 0.02
```

Result:

- Checkpoint: `runs/distill/generalized_soft_distill_12k_open1/distilled_model.pt`
- Bare model eval with `--no-tactical-guard --opening-random-pairs 1`:
  - random: 50/50
  - classic_rule: 0/50
  - reward_driven_medium: 0/50
  - reward_driven_hard: 0/50
  - normalized score: 6.67
- Focused random-opening eval:
  - reward_driven_hard: 0/100
  - reward_driven_medium: 0/100

Teacher check:

- `runs/ppo_from_tactical_50u/final_model.pt` with tactical guard and `--opening-random-pairs 1`:
  - random: 50/50
  - classic_rule: 50/50
  - reward_driven_medium: 24/50
  - reward_driven_hard: 23/50
  - normalized score: 57.33

Conclusion:

- The generalized soft distillation run did not improve the bare model.
- More importantly, the current tactical-search teacher is only about break-even against reward-driven bots under random openings, so it is not a strong enough teacher for robust generalization.
- Next route should improve the teacher first:
  - stronger tactical search/deeper reply handling,
  - failure-position replay,
  - or a search-based/self-play teacher that wins random-opening positions more reliably.

## 2026-05-27 - Multiprocess Distillation Pipeline

Goal: speed up tactical teacher label generation and make long distillation runs recoverable.

Changes:

- `scripts/distill_tactical_policy.py` now supports:
  - `--num-workers`
  - `--worker-batch-size`
  - `--worker-device`
  - `--save-every`
- Each worker process loads a frozen teacher checkpoint and generates labels independently.
- The main process keeps the student model and optimizer, merges worker batches, trains, and periodically saves `latest_model.pt`.

Smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\distill_tactical_policy.py --teacher-checkpoint runs\ppo_from_tactical_50u\final_model.pt --student-init runs\distill\tactical_distill_continue_4k_e6\distilled_model.pt --samples 128 --batch-size 64 --epochs-per-batch 1 --run-name smoke_mp_distill_2w --device auto --num-workers 2 --worker-batch-size 32 --worker-device cpu --opening-random-pairs 1 --rollout-bots random classic_rule reward_driven_medium reward_driven_hard --rollout-bot-weights 0.10 0.25 0.35 0.30 --teacher-self-play-prob 0.35 --hard-targets --entropy-coef 0.004 --save-every 1
```

Result:

- Completed successfully.
- Wrote:
  - `runs/distill/smoke_mp_distill_2w/latest_model.pt`
  - `runs/distill/smoke_mp_distill_2w/distilled_model.pt`
- Small smoke throughput was low because worker startup/model loading dominates short runs. Longer runs should use larger `--worker-batch-size` and `--save-every`.

## 2026-05-28 - Deep Teacher MP Distillation 8k

Run:

```powershell
.\.venv\Scripts\python.exe scripts\distill_tactical_policy.py --teacher-checkpoint runs\ppo_from_tactical_50u\final_model.pt --student-init runs\distill\tactical_distill_continue_4k_e6\distilled_model.pt --samples 8000 --batch-size 256 --epochs-per-batch 4 --run-name deep_teacher_mp_distill_8k_open1 --device auto --num-workers 6 --worker-batch-size 64 --worker-device cpu --opening-random-pairs 1 --rollout-bots random classic_rule reward_driven_medium reward_driven_hard --rollout-bot-weights 0.10 0.25 0.35 0.30 --teacher-self-play-prob 0.35 --hard-targets --entropy-coef 0.004 --save-every 5
```

Training completed:

- Checkpoint: `runs/distill/deep_teacher_mp_distill_8k_open1/distilled_model.pt`
- Samples: 8000
- Final training top1 against teacher labels: around 0.65 on the final partial batch.

Evaluation:

Bare model, random opening:

```powershell
--no-tactical-guard --opening-random-pairs 1
```

- random: 50/50
- classic_rule: 0/50
- reward_driven_medium: 0/50
- reward_driven_hard: 2/50
- normalized score: 8.80

Bare model, empty opening:

- random: 20/20
- classic_rule: 0/20
- reward_driven_medium: 1/20
- reward_driven_hard: 0/20
- normalized score: 8.00

Same checkpoint with tactical guard, random opening:

- random: 20/20
- classic_rule: 20/20
- reward_driven_medium: 18/20
- reward_driven_hard: 15/20
- normalized score: 84.00

Conclusion:

- The enhanced teacher/search is strong under random openings.
- The bare student did not absorb the teacher policy well. More of the same hard-label distillation is unlikely to solve this by itself.
- The next route should change the learning target/pipeline:
  - store generated datasets and train for many epochs over the same labels,
  - train value or threat heads in addition to policy,
  - use curriculum from one-ply tactical labels before deep-search labels,
  - or make the network architecture expose tactical features rather than asking the policy head to infer all tactics from sparse supervised actions.
