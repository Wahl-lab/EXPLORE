"""Active-learning frame review dialog.

Shows only the frames the model is uncertain about (probability ≈ 0.5).
The user clicks "Exploration" or "Not exploration" for each.  After reviewing
all presented frames, the classifier head is refitted automatically.

Workflow
--------
1. ``ReviewDialog`` is created with a list of frame images and their predicted
   probabilities.
2. The user reviews each frame and presses a key or clicks a button.
3. On close the ``corrections`` dict ``{frame_index: label}`` is available for
   callers to pass to ``ActiveLearner.update()``.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


class ReviewDialog:
    """Frame-by-frame review dialog for active learning corrections.

    Parameters
    ----------
    parent:
        Parent Tk window.
    frames:
        List of BGR numpy arrays — the uncertain frames to review.
    frame_indices:
        Original frame indices in the full video (used as keys in ``corrections``).
    probas:
        CLIP probability for each frame (displayed as a progress bar).
    on_done:
        Callback invoked with the ``corrections`` dict when the user finishes.
    """

    _MAX_DISPLAY = (640, 480)

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        frames: list[np.ndarray],
        frame_indices: list[int],
        probas: list[float],
        on_done: callable | None = None,  # type: ignore[type-arg]
    ) -> None:
        if not frames:
            raise ValueError("frames must not be empty")

        self.frames = frames
        self.frame_indices = frame_indices
        self.probas = probas
        self.on_done = on_done

        self.corrections: dict[int, int] = {}
        self._current = 0
        self._photo: ImageTk.PhotoImage | None = None

        self.window = tk.Toplevel(parent)
        self.window.title("EXPLORE 2.0 — Review Uncertain Frames")
        self.window.resizable(True, True)
        self.window.bind("<Key-e>", lambda _: self._label(1))
        self.window.bind("<Key-n>", lambda _: self._label(0))
        self.window.bind("<Key-s>", lambda _: self._skip())
        self._build_ui()
        self._show_frame(0)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Info bar ----
        info = ttk.Frame(self.window, padding=6)
        info.pack(fill="x")

        self._progress_var = tk.StringVar()
        ttk.Label(info, textvariable=self._progress_var, font=("Helvetica", 11)).pack(
            side="left"
        )

        self._prob_var = tk.StringVar()
        ttk.Label(info, textvariable=self._prob_var, foreground="grey").pack(
            side="right"
        )

        # ---- Canvas ----
        self._canvas = tk.Label(self.window)
        self._canvas.pack(pady=4)

        # ---- Confidence bar ----
        bar_frame = ttk.Frame(self.window, padding=(10, 0))
        bar_frame.pack(fill="x")
        ttk.Label(bar_frame, text="CLIP confidence:").pack(side="left")
        self._conf_bar = ttk.Progressbar(bar_frame, length=300, maximum=100)
        self._conf_bar.pack(side="left", padx=8)

        # ---- Keyboard hint ----
        hint = ttk.Label(
            self.window,
            text="Keyboard:  [E] Exploration    [N] Not exploration    [S] Skip",
            foreground="grey",
        )
        hint.pack(pady=(2, 6))

        # ---- Buttons ----
        btn_frame = ttk.Frame(self.window, padding=8)
        btn_frame.pack()

        ttk.Button(
            btn_frame,
            text="✓ Exploration  (E)",
            command=lambda: self._label(1),
        ).pack(side="left", padx=8)

        ttk.Button(
            btn_frame,
            text="✗ Not exploration  (N)",
            command=lambda: self._label(0),
        ).pack(side="left", padx=8)

        ttk.Button(
            btn_frame,
            text="→ Skip  (S)",
            command=self._skip,
        ).pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _show_frame(self, idx: int) -> None:
        self._current = idx
        total = len(self.frames)

        self._progress_var.set(f"Frame {idx + 1} of {total}")
        prob = self.probas[idx]
        self._prob_var.set(f"P(exploration) = {prob:.2f}")
        self._conf_bar["value"] = prob * 100

        frame = self.frames[idx]
        rgb = frame[:, :, ::-1]  # BGR → RGB
        img = Image.fromarray(rgb)

        max_w, max_h = self._MAX_DISPLAY
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.configure(image=self._photo)

    def _label(self, label: int) -> None:
        orig_idx = self.frame_indices[self._current]
        self.corrections[orig_idx] = label
        self._advance()

    def _skip(self) -> None:
        self._advance()

    def _advance(self) -> None:
        nxt = self._current + 1
        if nxt >= len(self.frames):
            self._finish()
        else:
            self._show_frame(nxt)

    def _finish(self) -> None:
        self.window.destroy()
        if self.on_done:
            self.on_done(self.corrections)
