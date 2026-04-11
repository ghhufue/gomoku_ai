### 1. 启动 Python Bot 服务（Uvicorn）
在项目根目录执行：

```bash
python -m uvicorn serve_bot:app --host 127.0.0.1 --port 8000 --reload
```

服务启动后默认地址：
- `http://127.0.0.1:8000`
- 主要接口：`POST /bot_move`
- 可选接口：`GET /list_runs`

---

### 2. Godot 端功能说明

当前主场景为 **3D Title**（`Node3D`），包含以下入口：

1. **VS PCbot**  
   与规则机器人对战（经典策略 bot）。

2. **VS Trainedbot**  
   与训练模型机器人对战（使用已选择的 `.pt` 模型路径）。

3. **PCbot vs Trainedbot**  
   两个 bot 对战（步进模式）。

右下角：
- **Trainedbot 路径选择按钮**（手动选择 `.pt` 文件）。

---

### 3. 进入 Game 场景后的交互

- 左上角：**Back to Title**（立即结束并返回标题）。
- 右上角：**Next Step**（仅在 `PCbot vs Trainedbot` 模式显示）。
- 其余区域：棋盘落子交互。

模式行为：
- `VS PCbot` / `VS Trainedbot`：玩家执先手，AI 自动应手。
- `PCbot vs Trainedbot`：点击 `Next Step` 执行下一手。

---

### 4. 已实现的前端表现（Godot）

- 棋子落子动画（缩放弹出效果）。
- 落子音效播放。
- 相机交互：
  - 滚轮缩放（zoom in/out）
  - 右键拖拽旋转（spin）
- 可调 AI 落子间隔（人机与 bot 对战可分开设置）。
- Trainedbot 模型路径可在 Title 手动选择并在 Game 内显示当前 bot 信息。

---

### 5. 使用注意

- 启动 Godot 前，请先启动 Uvicorn 服务。
- 若修改了 `serve_bot.py`，请确认服务已重启或使用 `--reload`。
- Trainedbot 请选择有效的 `.pt` 模型文件。