from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bots import create_bot
from gomoku_ai.env import BLACK, EMPTY, EnvConfig, GomokuEnv, action_to_coord


def board_text(env: GomokuEnv) -> str:
    rows = []
    for row in range(env.board.shape[0]):
        chars = []
        for col in range(env.board.shape[1]):
            value = int(env.board[row, col])
            if value == BLACK:
                chars.append("X")
            elif value == -BLACK:
                chars.append("O")
            else:
                chars.append(".")
        rows.append(" ".join(chars))
    return "\n".join(rows)


def first_legal_action(mask) -> int:
    for action, legal in enumerate(mask):
        if legal:
            return action
    raise RuntimeError("no legal action available")


def main() -> None:
    env = GomokuEnv(
        opponent=create_bot(name="random"),
        seed=2026,
        config=EnvConfig(reward_mode="terminal", use_action_mask=True),
    )
    obs, mask = env.reset()
    print("V0 environment demo")
    print(f"observation_shape={obs.shape}")
    print(f"action_mask_shape={mask.shape}")
    print(f"legal_actions={int(mask.sum())}")
    print(f"agent_player={'black' if env.agent_player == BLACK else 'white'}")
    print(board_text(env))

    action = first_legal_action(mask)
    row, col = action_to_coord(action)
    result = env.step(action)
    print()
    print(f"step_action={action} coord=({row}, {col})")
    print(f"reward={result.reward:.3f} done={result.done} info={result.info}")
    print(f"next_observation_shape={result.observation.shape}")
    print(f"next_legal_actions={int(result.action_mask.sum())}")
    print(board_text(env))


if __name__ == "__main__":
    main()
