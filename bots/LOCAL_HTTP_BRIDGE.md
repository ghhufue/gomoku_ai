# Local Bot HTTP Bridge

This bridge is for Godot requests that should be handled by existing `gomoku_ai.bots` implementations.

It is Bot-only:

- Use this bridge for rule/heuristic/random bot opponents.
- Use `local_inference_service` MoveEngine handling for neural models, external logic processes, or generic stdin/stdout engines.

## Endpoints

```text
GET  /health
GET  /bots
POST /wake
POST /bot_move
```

`POST /wake` lazily creates a bot instance:

```json
{
  "bot_name": "random"
}
```

`POST /bot_move` matches Godot's current local provider payload:

```json
{
  "board": [[0]],
  "current_player": 1,
  "bot_name": "random"
}
```

Response:

```json
{
  "row": 7,
  "col": 7,
  "x": 7,
  "y": 7,
  "bot_name": "random",
  "debug": {
    "bridge": "gomoku_ai.bots",
    "action": 112
  }
}
```

## Run

From `gomoku_ai/`:

```bash
uvicorn bots.local_http_api:app --host 127.0.0.1 --port 8000
```

Godot's `LocalHttpMoveProvider` can then call:

```text
http://127.0.0.1:8000/bot_move
```
