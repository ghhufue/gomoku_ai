请你基于当前五子棋 PPO 项目进行一次训练逻辑改造。目标不是重写项目，而是在现有训练闭环、环境、模型、reward、评估脚本基础上做兼容性修改。请先阅读项目结构和现有实现，再根据实际代码位置修改。所有代码注释请使用英文。

## 总体目标

本次修改包含四个核心方向：

1. 修改模型输入通道，从当前 3 通道改为推荐的 4 通道。
2. 调整 reward 数值尺度，使终局胜负奖励成为主导，辅助棋形 reward 缩小。
3. 训练时引入不同强度的对手 bot，而不是只和单一 bot 对弈。
4. 在验证/评估脚本中增加一个“棋力评分”机制，用固定对手池和固定比例计算当前模型强度，输出一个可参考的数值。

请尽量保持现有 CLI、配置、checkpoint、run 管理和日志系统兼容。如果涉及破坏性变更，请提供迁移处理或明确错误提示。

---

## 一、修改输入通道设计

当前观测通道为：

- obs[0]：当前玩家棋子
- obs[1]：对手棋子
- obs[2]：空位

请将其改为 4 通道：

- obs[0]：当前要落子的玩家的棋子位置，当前玩家棋子为 1.0，其他为 0.0
- obs[1]：对手棋子位置，对手棋子为 1.0，其他为 0.0
- obs[2]：上一手落点 one-hot，上一手位置为 1.0，其他为 0.0；如果当前局面没有上一手，则全 0
- obs[3]：当前玩家是否为黑棋的常数平面；如果当前玩家是黑棋，整张 15x15 为 1.0；如果当前玩家是白棋，整张 15x15 为 0.0

注意事项：

1. obs[0] 和 obs[1] 必须始终采用“当前玩家视角”，不要固定成黑棋/白棋。
2. 空位通道不再作为模型输入，合法动作仍然通过 action mask 处理。
3. action mask 的逻辑保持不变，仍然用于屏蔽非法落子。
4. 需要同步修改模型 input_channels 配置、默认 preset、checkpoint metadata、恢复 checkpoint 的逻辑。
5. 如果旧 checkpoint 的 input_channels 为 3，加载到新模型时应给出清晰提示，不要静默加载错误结构。
6. 所有训练、评估、回放、demo 中使用 observation 的地方都要同步检查。
7. 添加必要的单元测试或最小验证，确保：
   - 第一手时 last_move 通道全 0；
   - bot 或 agent 下完一步后，下一次 observation 的 last_move 通道正确；
   - 当前玩家切换时 obs[0]/obs[1] 仍然保持当前玩家视角；
   - obs[3] 能正确反映当前玩家是否为黑棋。

---

## 二、调整 reward 数值尺度

当前项目中的棋形 reward 数值偏大。请将训练时真正进入 PPO 的 reward 改为：

reward = terminal_reward + auxiliary_coef * scaled_auxiliary_reward

其中：

- terminal_reward：
  - 当前 agent 赢棋：+1.0
  - 当前 agent 输棋：-1.0
  - 平局或非终局：0.0
  - 非法落子可视为失败，建议 -1.0 或沿用现有非法落子惩罚逻辑，但最终尺度不要过大

- scaled_auxiliary_reward：
  - 来自现有 C++ reward evaluator 或 Python reward 逻辑的辅助棋形 reward
  - 不要直接使用原始大数值
  - 使用线性缩放，例如：
    scaled_auxiliary_reward = raw_auxiliary_reward * auxiliary_reward_scale
  - auxiliary_reward_scale 作为配置项，例如默认 0.001 或根据当前 reward 原始范围设置
  - 建议增加 clip，避免极端 reward 破坏 PPO：
    scaled_auxiliary_reward = clamp(scaled_auxiliary_reward, -auxiliary_reward_clip, auxiliary_reward_clip)
  - auxiliary_reward_clip 也作为配置项，例如默认 1.0 或 0.5

示例配置含义：

- auxiliary_reward_scale = 0.001
- auxiliary_reward_clip = 1.0
- auxiliary_coef_start = 0.1
- auxiliary_coef_end = 0.0
- auxiliary_coef_anneal_updates = 5000

也就是说，最终进入 PPO 的辅助 reward 大多数时候应该远小于终局奖励。终局胜负 +1/-1 应该成为主导信号。

---

## 三、辅助 reward 系数动态衰减

请加入动态辅助 reward 系数 auxiliary_coef，使训练前期可以利用棋形 reward 加速学习，后期逐渐回到以终局胜负为主。

推荐线性衰减：

progress = min(current_update / auxiliary_coef_anneal_updates, 1.0)

auxiliary_coef = auxiliary_coef_start + progress * (auxiliary_coef_end - auxiliary_coef_start)

最终：

reward = terminal_reward + auxiliary_coef * scaled_auxiliary_reward

要求：

1. auxiliary_coef_start、auxiliary_coef_end、auxiliary_coef_anneal_updates 都应放入训练配置。
2. 每个 PPO update 或每个 rollout 开始时确定当前 auxiliary_coef。
3. 同一个 rollout 内尽量固定 auxiliary_coef，避免一批数据内部 reward 函数不一致。
4. 日志中记录当前 auxiliary_coef。
5. 日志中最好同时记录：
   - terminal_reward_mean / sum
   - raw_auxiliary_reward_mean / sum
   - scaled_auxiliary_reward_mean / sum
   - final_reward_mean / sum
   - auxiliary_coef
6. 如果当前项目已有 reward events、offense_score、defense_score、special_rewards 等字段，请尽量保留 debug 信息，但 PPO 使用的 final reward 必须是缩放后的结果。

---

## 四、训练时使用不同对手 bot

当前训练不应只和单一 bot 对弈。请实现 configurable opponent pool。

要求：

1. 在训练配置中加入 opponent_pool，例如：

[training.opponents]
names = ["random", "classic_rule", "reward_driven_medium", "reward_driven_hard"]
weights = [0.2, 0.4, 0.3, 0.1]

2. 每局开始时，根据 weights 随机选择一个对手 bot。
3. 选择的 bot 名称应写入 episode info 或训练日志，便于统计不同 bot 下的胜率。
4. 如果当前环境初始化时只支持固定 bot，请改造为：
   - 每个 env reset 时可以指定或采样 bot；
   - 或在 wrapper/vector env 层实现按 episode 切换 bot。
5. 训练日志中增加按 bot 统计：
   - games
   - win_rate
   - draw_rate
   - loss_rate
   - average_episode_length
   - average_reward
6. 保持现有 bot factory 机制，不要硬编码 bot 类。
7. 如果某些 bot 名称不存在，应在启动时给出清晰错误。

---

## 五、增加棋力评估脚本/验证逻辑

请在验证脚本中增加一个固定的“棋力评分”评估模式，用于每次验证时输出一个单一数值，方便观察模型强度变化。

核心思路：

模型分别和不同难度 bot 对弈，按照固定对局数量和固定权重计算 score。

建议新增脚本或在现有 evaluate.py 中增加参数，例如：

python scripts/evaluate_strength.py --checkpoint path/to/model.pt

或者：

python scripts/evaluate.py --mode strength --checkpoint path/to/model.pt

评估对手池建议：

- random：低难度
- classic_rule：中等难度
- reward_driven_medium：较高难度
- reward_driven_hard：高难度

每个 bot 使用固定局数，例如：

- random：20 局
- classic_rule：20 局
- reward_driven_medium：20 局
- reward_driven_hard：20 局

评分规则示例：

每局基础结果分：

- win = 1.0
- draw = 0.5
- loss = 0.0

不同 bot 的难度权重：

- random：1.0
- classic_rule：2.0
- reward_driven_medium：4.0
- reward_driven_hard：8.0

对某个 bot 的得分：

bot_score = result_rate * difficulty_weight

其中：

result_rate = (wins + 0.5 * draws) / total_games

总棋力评分：

strength_score = sum(bot_score for all bots)

也可以归一化为 0 到 100：

strength_score_100 = 100 * sum(result_rate_i * weight_i) / sum(weight_i)

请输出两种形式：

1. raw_strength_score
2. normalized_strength_score_0_100

输出报告中包含：

- checkpoint path
- total games
- overall win/draw/loss
- 每个 bot 的：
  - games
  - wins
  - draws
  - losses
  - win_rate
  - draw_rate
  - loss_rate
  - result_rate
  - difficulty_weight
  - contribution_score
- raw_strength_score
- normalized_strength_score_0_100

请同时支持 JSON 输出，便于 run history 记录，例如：

runs/<run_id>/history/strength_evals.jsonl

如果当前项目已有 evals.jsonl，可以把 strength score 也写进去，字段名清晰即可。

---

## 六、训练期间自动验证棋力

如果当前训练脚本已有定期 evaluate，请将新的 strength evaluation 接入训练流程。

要求：

1. 每隔 eval_interval updates，运行一次 strength evaluation。
2. 使用固定随机种子或可配置 seed，保证不同训练阶段的分数可比较。
3. 验证时不要使用训练时的随机 opponent weights，而是使用固定评估配置。
4. 评估结果写入日志和 run history。
5. 如果 normalized_strength_score_0_100 超过历史最好，则可以保存 best_strength_model.pt。
6. 不要只用训练 reward 判断 best model，增加按 strength score 保存模型的选项。

---

## 七、配置建议

请新增或调整配置项，示例：

[model]
input_channels = 4

[reward]
terminal_win = 1.0
terminal_loss = -1.0
terminal_draw = 0.0
auxiliary_reward_scale = 0.001
auxiliary_reward_clip = 1.0
auxiliary_coef_start = 0.1
auxiliary_coef_end = 0.0
auxiliary_coef_anneal_updates = 5000

[training.opponents]
names = ["random", "classic_rule", "reward_driven_medium", "reward_driven_hard"]
weights = [0.2, 0.4, 0.3, 0.1]

[evaluation.strength]
enabled = true
bots = ["random", "classic_rule", "reward_driven_medium", "reward_driven_hard"]
games_per_bot = [20, 20, 20, 20]
difficulty_weights = [1.0, 2.0, 4.0, 8.0]
seed = 12345
save_best_by_strength = true

请根据项目现有配置格式实现，不一定必须完全照搬上述 TOML 结构，但语义要保持一致。

---

## 八、兼容性和测试要求

请完成以下检查：

1. 训练脚本可以正常启动。
2. 环境 reset/step 后 observation shape 为 4 x 15 x 15。
3. action mask 仍然正确。
4. PPO rollout 中 reward 使用缩放后的 final reward。
5. 日志中能看到 auxiliary_coef 随 update 变化。
6. 训练时 opponent bot 会按配置切换。
7. evaluate strength 脚本可以独立运行。
8. strength evaluation 能输出 JSON 和可读文本报告。
9. 旧的 3 通道 checkpoint 加载时有清晰报错。
10. 如果项目有 pytest，请添加或更新相关测试，确保核心逻辑不被破坏。

---

## 九、实现注意事项

1. 不要把 reward 原始大数值直接传入 PPO。
2. 不要删除 C++ reward evaluator 的 debug 能力。
3. 不要把空位通道重新塞回模型输入。
4. 不要把 bot 名称硬编码在训练循环里，应复用 bot factory。
5. 不要让评估用训练时的随机 opponent pool，评估应该固定、可复现。
6. 不要只看 average reward 保存 best model，新增按 strength score 保存 best model 的能力。
7. 如果需要改动 checkpoint metadata，请确保保存 input_channels、model preset、reward config、opponent config 等关键信息。
8. 如果某些旧测试依赖 3 通道 observation，需要同步更新到 4 通道。

---

## 最终期望结果

修改完成后，我希望能够：

1. 使用 4 通道输入训练模型：
   - 当前玩家棋子
   - 对手棋子
   - 上一手落点
   - 当前玩家是否黑棋

2. PPO 实际使用的 reward 是：
   - 终局胜负 +1/-1 为主
   - 辅助棋形 reward 被线性缩小
   - 辅助 reward 系数随训练动态降低

3. 训练时模型会和多个不同难度 bot 对弈，而不是只对一个固定 bot。

4. 每次验证时可以得到一个清晰的棋力评分，例如：
   - raw_strength_score
   - normalized_strength_score_0_100

5. 我可以通过这个 strength score 判断模型是否真的变强，而不是只看训练 reward。