from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - fallback for minimal Python installs.
    Image = None
    ImageTk = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bots import available_bots, create_bot
from gomoku_ai.env import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    action_to_coord,
    call_bot_action,
    coord_to_action,
    empty_actions,
    neighboring_action_mask,
)
from gomoku_ai.model import ActorCriticNet, model_config_from_checkpoint_payload
from gomoku_ai.opening import apply_random_opening_pairs_to_board
from gomoku_ai.tactical_policy import select_tactical_search_action


APP_TITLE = "Gomoku Policy Studio"
CELL = 42
BOARD_PADDING = 42
BOARD_PIXELS = CELL * (BOARD_SIZE - 1) + BOARD_PADDING * 2
HEATMAP_MARGIN = CELL // 2

WINDOW_BG = "#0b0f14"
SURFACE = "#111820"
SURFACE_2 = "#17212b"
SURFACE_3 = "#1e2a36"
TEXT = "#edf2f7"
MUTED = "#91a0ad"
ACCENT = "#65e4b1"
ACCENT_2 = "#f7c85f"
BOARD_BASE = "#d7aa63"
BOARD_DARK = "#7f5b2c"
BLACK_STONE = "#151515"
WHITE_STONE = "#f5f0e6"

HEAT_STOPS = (
    (0.00, (27, 42, 72)),
    (0.20, (40, 112, 173)),
    (0.45, (52, 205, 180)),
    (0.70, (246, 205, 88)),
    (1.00, (246, 91, 85)),
)


@dataclass
class PolicySnapshot:
    probabilities: np.ndarray
    legal_mask: np.ndarray
    action: int
    value: float
    source: str = "model"


def player_name(player: int) -> str:
    return "Black" if player == BLACK else "White"


def stone_color(player: int) -> str:
    return BLACK_STONE if player == BLACK else WHITE_STONE


def find_latest_checkpoint() -> Path | None:
    candidates = list((ROOT / "runs").glob("*/final_model.pt"))
    candidates.extend((ROOT / "runs").glob("*/checkpoints/best_model.pt"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_model(checkpoint_path: Path, device: str) -> ActorCriticNet:
    payload = torch.load(checkpoint_path, map_location=device)
    model = ActorCriticNet(model_config_from_checkpoint_payload(payload)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def build_observation(board: np.ndarray, current_player: int, last_move: int | None) -> np.ndarray:
    obs = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    obs[0] = (board == current_player).astype(np.float32)
    obs[1] = (board == -current_player).astype(np.float32)
    if last_move is not None:
        row, col = action_to_coord(last_move)
        obs[2, row, col] = 1.0
    if current_player == BLACK:
        obs[3] = 1.0
    return obs


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def is_full(board: np.ndarray) -> bool:
    return not np.any(board == EMPTY)


def winner_after_move(board: np.ndarray, row: int, col: int, player: int) -> bool:
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            rr = row + dr * sign
            cc = col + dc * sign
            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr, cc] == player:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= 5:
            return True
    return False


def mix_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] * (1.0 - t) + b[i] * t)) for i in range(3))


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def heat_color(ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))
    for index in range(1, len(HEAT_STOPS)):
        left_pos, left_rgb = HEAT_STOPS[index - 1]
        right_pos, right_rgb = HEAT_STOPS[index]
        if ratio <= right_pos:
            span = max(1e-9, right_pos - left_pos)
            local_t = (ratio - left_pos) / span
            return rgb_hex(mix_rgb(left_rgb, right_rgb, local_t))
    return rgb_hex(HEAT_STOPS[-1][1])


def normalized_policy_strength(values: np.ndarray) -> np.ndarray:
    """Spread policy probabilities into visible color strength.

    PPO policies can be either very peaky or nearly flat. A log transform plus
    percentile floor makes weak moves visibly cool and preferred moves hot.
    """
    if values.size == 0:
        return values.astype(np.float64)
    clipped = np.maximum(values.astype(np.float64), 1e-12)
    log_values = np.log10(clipped)
    low = float(np.percentile(log_values, 5))
    high = float(log_values.max())
    if high <= low + 1e-9:
        return np.ones_like(log_values, dtype=np.float64)
    ratios = (log_values - low) / (high - low)
    return np.clip(ratios, 0.0, 1.0)


class PolicyViewer(tk.Tk):
    def __init__(self, checkpoint: Path | None, bot_name: str, device: str):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x860")
        self.minsize(1160, 780)
        self.configure(bg=WINDOW_BG)

        self.device = resolve_device(device)
        self.checkpoint_path: Path | None = None
        self.model: ActorCriticNet | None = None
        self.bot_name = bot_name
        self.bot = create_bot(name=bot_name)
        self.rng = np.random.default_rng()

        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.model_player = BLACK
        self.current_player = BLACK
        self.last_move: int | None = None
        self.move_number = 0
        self.done = False
        self.status = tk.StringVar(value="Load a model to begin")
        self.checkpoint_var = tk.StringVar(value="No checkpoint loaded")
        self.bot_var = tk.StringVar(value=bot_name)
        self.side_var = tk.StringVar(value="black")
        self.tactical_guard_var = tk.BooleanVar(value=True)
        self.random_opening_var = tk.BooleanVar(value=True)
        self.policy_snapshot: PolicySnapshot | None = None
        self.heatmap_image: object | None = None
        self.move_log: list[str] = []

        self._build_style()
        self._build_layout()
        self._bind_shortcuts()

        if checkpoint is None:
            checkpoint = find_latest_checkpoint()
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self.reset_game()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=WINDOW_BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=WINDOW_BG)
        style.configure("Panel.TFrame", background=SURFACE)
        style.configure("Card.TFrame", background=SURFACE_2)
        style.configure("TLabel", background=WINDOW_BG, foreground=TEXT)
        style.configure("Panel.TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Card.TLabel", background=SURFACE_2, foreground=TEXT)
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED)
        style.configure("CardMuted.TLabel", background=SURFACE_2, foreground=MUTED)
        style.configure("Title.TLabel", background=WINDOW_BG, foreground=TEXT, font=("Segoe UI Semibold", 24))
        style.configure("Subtitle.TLabel", background=WINDOW_BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=SURFACE, foreground=ACCENT, font=("Segoe UI Semibold", 13))
        style.configure("Hero.TLabel", background=SURFACE_2, foreground=TEXT, font=("Segoe UI Semibold", 16))
        style.configure("TButton", background=SURFACE_3, foreground=TEXT, borderwidth=0, padding=(14, 10))
        style.map("TButton", background=[("active", "#273646")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#06110d", font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#77edc0")])
        style.configure("TRadiobutton", background=SURFACE, foreground=TEXT)
        style.map("TRadiobutton", background=[("active", SURFACE)], foreground=[("active", TEXT)])
        style.configure("TCombobox", fieldbackground=SURFACE_3, background=SURFACE_3, foreground=TEXT, arrowcolor=TEXT)

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=20)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Gomoku Policy Studio", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Interactive probability heatmap for model move selection",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.status_label = tk.Label(
            header,
            textvariable=self.status,
            bg="#17211b",
            fg=ACCENT_2,
            padx=16,
            pady=9,
            font=("Segoe UI Semibold", 10),
        )
        self.status_label.grid(row=0, column=1, rowspan=2, sticky="e")

        board_panel = ttk.Frame(root, style="Panel.TFrame", padding=18)
        board_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
        board_panel.rowconfigure(0, weight=1)
        board_panel.columnconfigure(0, weight=1)
        self.board_canvas = tk.Canvas(
            board_panel,
            width=BOARD_PIXELS,
            height=BOARD_PIXELS,
            bg=BOARD_BASE,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        self.board_canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(root, style="Panel.TFrame", padding=18)
        side.grid(row=1, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        self._build_controls(side)
        self._build_snapshot_panel(side)
        self._build_policy_panel(side)
        self._build_log_panel(side)

    def _build_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent, style="Panel.TFrame")
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Checkpoint", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.checkpoint_var, style="Panel.TLabel", wraplength=380).grid(
            row=1, column=0, sticky="ew", pady=(4, 10)
        )
        ttk.Button(controls, text="Choose checkpoint", command=self.choose_checkpoint).grid(
            row=2, column=0, sticky="ew", pady=(0, 10)
        )

        ttk.Label(controls, text="Opponent bot", style="Muted.TLabel").grid(row=3, column=0, sticky="w")
        self.bot_combo = ttk.Combobox(controls, textvariable=self.bot_var, values=available_bots(), state="readonly")
        self.bot_combo.grid(row=4, column=0, sticky="ew", pady=(4, 10))
        self.bot_combo.bind("<<ComboboxSelected>>", lambda _event: self.change_bot())

        ttk.Label(controls, text="Model side", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        side_row = ttk.Frame(controls, style="Panel.TFrame")
        side_row.grid(row=6, column=0, sticky="ew", pady=(4, 10))
        ttk.Radiobutton(side_row, text="Black", variable=self.side_var, value="black", command=self.reset_game).pack(side=tk.LEFT)
        ttk.Radiobutton(side_row, text="White", variable=self.side_var, value="white", command=self.reset_game).pack(
            side=tk.LEFT,
            padx=(16, 0),
        )

        ttk.Checkbutton(
            controls,
            text="Tactical guard",
            variable=self.tactical_guard_var,
            command=self.render,
        ).grid(row=7, column=0, sticky="w", pady=(0, 10))
        ttk.Checkbutton(
            controls,
            text="Random opening",
            variable=self.random_opening_var,
            command=self.reset_game,
        ).grid(row=8, column=0, sticky="w", pady=(0, 10))

        ttk.Button(controls, text="Next move", style="Accent.TButton", command=self.next_step).grid(
            row=9, column=0, sticky="ew", pady=(12, 8)
        )
        ttk.Button(controls, text="New game", command=self.reset_game).grid(row=10, column=0, sticky="ew")

    def _build_snapshot_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Card.TFrame", padding=14)
        panel.grid(row=1, column=0, sticky="ew", pady=(18, 0))
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="Current Decision", style="Hero.TLabel").grid(row=0, column=0, sticky="w")
        self.decision_label = ttk.Label(
            panel,
            text="No model move yet",
            style="CardMuted.TLabel",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.decision_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_policy_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame")
        panel.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Top Policy", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.metric_text = tk.Text(
            panel,
            height=11,
            width=44,
            bg=SURFACE_2,
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.metric_text.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.metric_text.configure(state=tk.DISABLED)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame")
        panel.grid(row=3, column=0, sticky="nsew", pady=(18, 0))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Game Log", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(
            panel,
            height=12,
            width=44,
            bg="#0d1218",
            fg="#d6dde8",
            insertbackground=TEXT,
            relief=tk.FLAT,
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log_text.configure(state=tk.DISABLED)

    def _bind_shortcuts(self) -> None:
        self.bind("<space>", lambda _event: self.next_step())
        self.bind("<Control-o>", lambda _event: self.choose_checkpoint())
        self.bind("<Control-r>", lambda _event: self.reset_game())

    def choose_checkpoint(self) -> None:
        initial = ROOT / "runs"
        path = filedialog.askopenfilename(
            title="Choose checkpoint",
            initialdir=str(initial if initial.exists() else ROOT),
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.load_checkpoint(Path(path))
            self.reset_game()

    def load_checkpoint(self, path: Path) -> None:
        try:
            self.model = load_model(path, self.device)
        except Exception as exc:
            messagebox.showerror("Model load failed", str(exc))
            return
        self.checkpoint_path = path.resolve()
        display = self.checkpoint_path
        if self.checkpoint_path.is_relative_to(ROOT):
            display = self.checkpoint_path.relative_to(ROOT)
        self.checkpoint_var.set(str(display))
        self.status.set(f"Model loaded on {self.device}")

    def change_bot(self) -> None:
        try:
            self.bot_name = self.bot_var.get()
            self.bot = create_bot(name=self.bot_name)
        except Exception as exc:
            messagebox.showerror("Bot load failed", str(exc))
            return
        self.reset_game()

    def reset_game(self) -> None:
        self.board.fill(EMPTY)
        self.model_player = BLACK if self.side_var.get() == "black" else -BLACK
        self.current_player = BLACK
        self.last_move = None
        self.move_number = 0
        self.done = False
        self.policy_snapshot = None
        self.move_log = []
        if self.model_player != BLACK:
            self._play_bot_move()
        if self.random_opening_var.get():
            self._apply_random_opening_pair()
        self.status.set(f"{player_name(self.current_player)} to move")
        self.render()

    def _apply_random_opening_pair(self) -> None:
        first_player = self.current_player
        before_last = self.last_move
        last_move, next_player = apply_random_opening_pairs_to_board(
            self.board,
            1,
            self.rng,
            first_player=first_player,
        )
        if last_move is None:
            self.last_move = before_last
            return

        player = first_player
        # Reconstruct the two random moves from board state is not reliable, so log
        # the opening as a compact marker instead of exact coordinates.
        self.move_number += 2
        self.move_log.append(f"{self.move_number - 1:02d}-{self.move_number:02d}. Random opening pair")
        self.current_player = next_player
        self.last_move = last_move

    def next_step(self) -> None:
        if self.done:
            self.status.set("Game finished. Start a new game.")
            return
        if self.model is None:
            messagebox.showinfo("Model required", "Choose a checkpoint first.")
            return
        try:
            if self.current_player == self.model_player:
                snapshot = self.compute_policy()
                self.policy_snapshot = snapshot
                self._apply_move(snapshot.action, self.current_player, "model")
            else:
                self._play_bot_move()
        except Exception as exc:
            messagebox.showerror("Move failed", str(exc))
            return
        self.render()

    def compute_policy(self) -> PolicySnapshot:
        assert self.model is not None
        mask = neighboring_action_mask(self.board, radius=2, opening_radius=1)
        obs = build_observation(self.board, self.current_player, self.last_move)
        obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(mask[None, ...], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            dist, value = self.model.masked_distribution(obs_tensor, mask_tensor)
            probs = dist.probs.squeeze(0).detach().cpu().numpy().astype(np.float64)
            logits = dist.logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
            action = int(np.argmax(probs))
        source = "model"
        if self.tactical_guard_var.get():
            action, source, _ = select_tactical_search_action(self.board, self.current_player, logits)
        return PolicySnapshot(
            probabilities=probs,
            legal_mask=mask.astype(bool),
            action=action,
            value=float(value.item()),
            source=source,
        )

    def _play_bot_move(self) -> None:
        action = call_bot_action(self.bot, self.board.copy(), self.current_player, self.rng)
        row, col = action_to_coord(action)
        if self.board[row, col] != EMPTY:
            empties = empty_actions(self.board)
            action = int(self.rng.choice(empties))
        self._apply_move(action, self.current_player, self.bot_name)

    def _apply_move(self, action: int, player: int, source: str) -> None:
        row, col = action_to_coord(action)
        if self.board[row, col] != EMPTY:
            raise ValueError(f"illegal move at ({row}, {col})")
        self.board[row, col] = player
        self.last_move = action
        self.move_number += 1
        label = "Model" if source == "model" else f"Bot:{source}"
        self.move_log.append(f"{self.move_number:02d}. {label:<22s} {player_name(player):<5s} -> ({row:02d}, {col:02d})")
        if winner_after_move(self.board, row, col, player):
            self.done = True
            self.status.set(f"{label} wins")
            return
        if is_full(self.board):
            self.done = True
            self.status.set("Draw")
            return
        self.current_player = -player
        self.status.set(f"{player_name(self.current_player)} to move")

    def render(self) -> None:
        self.render_board()
        self.render_metrics()
        self.render_log()

    def board_xy(self, row: int, col: int) -> tuple[int, int]:
        return BOARD_PADDING + col * CELL, BOARD_PADDING + row * CELL

    def render_board(self) -> None:
        canvas = self.board_canvas
        canvas.delete("all")
        self._draw_board_background(canvas)
        self._draw_policy_heat(canvas)
        self._draw_grid(canvas)
        self._draw_stones(canvas)
        self._draw_board_labels(canvas)
        self._draw_legend(canvas)

    def _draw_board_background(self, canvas: tk.Canvas) -> None:
        canvas.create_rectangle(0, 0, BOARD_PIXELS, BOARD_PIXELS, fill=BOARD_BASE, outline="")
        for i in range(10):
            inset = 12 + i * 2
            color = "#c9914b" if i % 2 == 0 else "#e0b872"
            canvas.create_rectangle(inset, inset, BOARD_PIXELS - inset, BOARD_PIXELS - inset, outline=color)
        canvas.create_rectangle(18, 18, BOARD_PIXELS - 18, BOARD_PIXELS - 18, outline=BOARD_DARK, width=2)

    def _draw_policy_heat(self, canvas: tk.Canvas) -> None:
        snapshot = self.policy_snapshot
        if snapshot is None:
            self.heatmap_image = None
            return
        probs = snapshot.probabilities.reshape(BOARD_SIZE, BOARD_SIZE)
        legal = snapshot.legal_mask.reshape(BOARD_SIZE, BOARD_SIZE)
        legal_values = probs[legal]
        top = float(legal_values.max()) if legal_values.size else 0.0
        if top <= 0:
            self.heatmap_image = None
            return
        strengths = normalized_policy_strength(legal_values)
        strength_grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
        strength_grid[legal] = strengths

        self.heatmap_image = self._build_heatmap_image(strength_grid)
        x0 = BOARD_PADDING - HEATMAP_MARGIN
        y0 = BOARD_PADDING - HEATMAP_MARGIN
        canvas.create_image(x0, y0, image=self.heatmap_image, anchor=tk.NW)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if not legal[row, col]:
                    continue
                probability = float(probs[row, col])
                ratio = float(strength_grid[row, col])
                if probability <= 0 or ratio <= 0.32:
                    continue
                x, y = self.board_xy(row, col)
                fg = "#111820" if ratio > 0.58 else "#edf6ff"
                canvas.create_text(x, y, text=f"{probability * 100:.0f}", fill=fg, font=("Segoe UI Semibold", 8))

        best_row, best_col = action_to_coord(snapshot.action)
        x, y = self.board_xy(best_row, best_col)
        canvas.create_rectangle(x - 19, y - 19, x + 19, y + 19, outline="#fff1a8", width=3)
        canvas.create_rectangle(x - 23, y - 23, x + 23, y + 23, outline="#0b0f14", width=1)

    def _build_heatmap_image(self, strength_grid: np.ndarray) -> tk.PhotoImage:
        size = CELL * (BOARD_SIZE - 1) + HEATMAP_MARGIN * 2 + 1
        if Image is not None and ImageTk is not None:
            values = self._interpolated_heat_values(strength_grid, size)
            rgb = self._heat_rgb_array(values)
            board = np.asarray([215, 170, 99], dtype=np.float64)
            alpha = (0.30 + values[..., None] * 0.66).clip(0.0, 1.0)
            blended = (board * (1.0 - alpha) + rgb.astype(np.float64) * alpha).clip(0, 255).astype(np.uint8)
            return ImageTk.PhotoImage(Image.fromarray(blended, mode="RGB"))

        pixels: list[str] = []
        for y in range(size):
            board_y = (y - HEATMAP_MARGIN) / CELL
            row0 = int(np.floor(board_y))
            row1 = row0 + 1
            ty = board_y - row0
            row0 = max(0, min(BOARD_SIZE - 1, row0))
            row1 = max(0, min(BOARD_SIZE - 1, row1))
            row_colors: list[str] = []
            for x in range(size):
                board_x = (x - HEATMAP_MARGIN) / CELL
                col0 = int(np.floor(board_x))
                col1 = col0 + 1
                tx = board_x - col0
                col0 = max(0, min(BOARD_SIZE - 1, col0))
                col1 = max(0, min(BOARD_SIZE - 1, col1))

                top_value = strength_grid[row0, col0] * (1.0 - tx) + strength_grid[row0, col1] * tx
                bottom_value = strength_grid[row1, col0] * (1.0 - tx) + strength_grid[row1, col1] * tx
                value = top_value * (1.0 - ty) + bottom_value * ty
                value = max(0.0, min(1.0, float(value)))

                heat = tuple(int(heat_color(value)[i : i + 2], 16) for i in (1, 3, 5))
                board = (215, 170, 99)
                alpha = 0.30 + value * 0.66
                color = rgb_hex(mix_rgb(board, heat, alpha))
                row_colors.append(color)
            pixels.append("{" + " ".join(row_colors) + "}")

        image = tk.PhotoImage(width=size, height=size)
        image.put(" ".join(pixels), to=(0, 0))
        return image

    def _interpolated_heat_values(self, strength_grid: np.ndarray, size: int) -> np.ndarray:
        axis = (np.arange(size, dtype=np.float64) - HEATMAP_MARGIN) / CELL
        base = np.floor(axis).astype(np.int64)
        frac = axis - base
        base = np.clip(base, 0, BOARD_SIZE - 1)
        next_index = np.clip(base + 1, 0, BOARD_SIZE - 1)

        left = strength_grid[:, base] * (1.0 - frac) + strength_grid[:, next_index] * frac
        top = left[base, :] * (1.0 - frac[:, None])
        bottom = left[next_index, :] * frac[:, None]
        return np.clip(top + bottom, 0.0, 1.0)

    def _heat_rgb_array(self, values: np.ndarray) -> np.ndarray:
        rgb = np.zeros((*values.shape, 3), dtype=np.float64)
        remaining = np.ones(values.shape, dtype=bool)
        for index in range(1, len(HEAT_STOPS)):
            left_pos, left_rgb = HEAT_STOPS[index - 1]
            right_pos, right_rgb = HEAT_STOPS[index]
            segment = remaining & (values <= right_pos)
            if not np.any(segment):
                continue
            span = max(1e-9, right_pos - left_pos)
            t = ((values[segment] - left_pos) / span).clip(0.0, 1.0)
            left = np.asarray(left_rgb, dtype=np.float64)
            right = np.asarray(right_rgb, dtype=np.float64)
            rgb[segment] = left * (1.0 - t[:, None]) + right * t[:, None]
            remaining[segment] = False
        if np.any(remaining):
            rgb[remaining] = np.asarray(HEAT_STOPS[-1][1], dtype=np.float64)
        return rgb.astype(np.uint8)

    def _draw_grid(self, canvas: tk.Canvas) -> None:
        for index in range(BOARD_SIZE):
            x0, y0 = self.board_xy(index, 0)
            x1, y1 = self.board_xy(index, BOARD_SIZE - 1)
            canvas.create_line(x0, y0, x1, y1, fill="#65471f", width=1)
            x0, y0 = self.board_xy(0, index)
            x1, y1 = self.board_xy(BOARD_SIZE - 1, index)
            canvas.create_line(x0, y0, x1, y1, fill="#65471f", width=1)
        for row, col in ((3, 3), (3, 11), (7, 7), (11, 3), (11, 11)):
            x, y = self.board_xy(row, col)
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#4d3518", outline="")

    def _draw_stones(self, canvas: tk.Canvas) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                value = int(self.board[row, col])
                if value == EMPTY:
                    continue
                x, y = self.board_xy(row, col)
                radius = 17
                fill = stone_color(value)
                outline = "#050505" if value == BLACK else "#b8ac96"
                canvas.create_oval(x - radius - 2, y - radius + 2, x + radius - 2, y + radius + 2, fill="#6e4b24", outline="")
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline, width=2)
                shine = "#383838" if value == BLACK else "#ffffff"
                canvas.create_oval(x - 9, y - 10, x - 2, y - 3, fill=shine, outline="")
        if self.last_move is not None:
            row, col = action_to_coord(self.last_move)
            x, y = self.board_xy(row, col)
            canvas.create_rectangle(x - 7, y - 7, x + 7, y + 7, outline="#fff1a8", width=2)

    def _draw_board_labels(self, canvas: tk.Canvas) -> None:
        for i in range(BOARD_SIZE):
            x, _ = self.board_xy(0, i)
            _, y = self.board_xy(i, 0)
            canvas.create_text(x, 22, text=str(i), fill="#3e2a13", font=("Segoe UI Semibold", 9))
            canvas.create_text(22, y, text=str(i), fill="#3e2a13", font=("Segoe UI Semibold", 9))

    def _draw_legend(self, canvas: tk.Canvas) -> None:
        x0 = BOARD_PADDING
        y0 = BOARD_PIXELS - 24
        width = 210
        steps = 42
        for i in range(steps):
            ratio = i / max(1, steps - 1)
            color = heat_color(ratio)
            x = x0 + int(width * i / steps)
            canvas.create_rectangle(x, y0, x + int(width / steps) + 2, y0 + 8, fill=color, outline="")
        canvas.create_text(x0, y0 - 8, text="low", fill="#3e2a13", anchor="w", font=("Segoe UI", 8))
        canvas.create_text(x0 + width, y0 - 8, text="high probability", fill="#3e2a13", anchor="e", font=("Segoe UI", 8))

    def render_metrics(self) -> None:
        self.metric_text.configure(state=tk.NORMAL)
        self.metric_text.delete("1.0", tk.END)
        snapshot = self.policy_snapshot
        if snapshot is None:
            self.decision_label.configure(text="No model move yet. Press Next move to render the probability field.")
            text = "The board will color every legal point by policy probability.\n\nBlue: low probability\nGreen: medium probability\nYellow/red: model preference"
        else:
            legal_probs = snapshot.probabilities.copy()
            legal_probs[~snapshot.legal_mask] = -1.0
            top_actions = np.argsort(legal_probs)[::-1][:10]
            legal_distribution = snapshot.probabilities[snapshot.legal_mask]
            entropy = -float(np.sum(legal_distribution * np.log(legal_distribution + 1e-12)))
            best_row, best_col = action_to_coord(snapshot.action)
            best_prob = snapshot.probabilities[snapshot.action] * 100.0
            self.decision_label.configure(
                text=f"Move ({best_row:02d}, {best_col:02d}) with {best_prob:.2f}% probability. "
                f"Value estimate {snapshot.value:+.4f}; entropy {entropy:.4f}; source {snapshot.source}."
            )
            lines = [
                f"value   {snapshot.value:+.4f}",
                f"entropy {entropy:.4f}",
                f"best    ({best_row:02d}, {best_col:02d})  {best_prob:6.2f}%",
                f"source  {snapshot.source}",
                "scale   blue < cyan < yellow < red",
                "",
            ]
            for rank, action in enumerate(top_actions, start=1):
                row, col = action_to_coord(int(action))
                prob = snapshot.probabilities[action] * 100.0
                bar = "#" * max(1, int(round(prob / max(0.1, best_prob) * 16)))
                lines.append(f"{rank:02d} ({row:02d},{col:02d}) {prob:6.2f}%  {bar}")
            text = "\n".join(lines)
        self.metric_text.insert("1.0", text)
        self.metric_text.configure(state=tk.DISABLED)

    def render_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", "\n".join(self.move_log[-26:]))
        self.log_text.configure(state=tk.DISABLED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive GUI for visualizing Gomoku policy probabilities.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to latest checkpoint under runs/.")
    parser.add_argument("--bot", type=str, default="reward_driven_hard", choices=available_bots())
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = PolicyViewer(checkpoint=args.checkpoint, bot_name=args.bot, device=args.device)
    app.mainloop()


if __name__ == "__main__":
    main()
