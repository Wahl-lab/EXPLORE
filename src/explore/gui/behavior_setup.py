"""Behavioral definition dialog — text prompts for CLIP.

The researcher writes sentences describing what counts as exploration in their
experiment.  These prompts are the experiment's behavioural definition:
text, version-controlled, quotable in the Methods section.

A "Preview" button runs zero-shot CLIP on 20 random frames and shows
confidence scores as a histogram so the user can judge prompt quality before
committing.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from explore.config import BehaviorConfig


class BehaviorSetupDialog:
    """Dialog for writing and previewing behavioral text prompts.

    Parameters
    ----------
    parent:
        Parent Tk window.
    behavior:
        ``BehaviorConfig`` to populate / update.
    on_confirm:
        Callback invoked with the updated ``BehaviorConfig`` when confirmed.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        behavior: BehaviorConfig,
        on_confirm: callable | None = None,  # type: ignore[type-arg]
    ) -> None:
        self.parent = parent
        self.behavior = behavior
        self.on_confirm = on_confirm

        self.window = tk.Toplevel(parent)
        self.window.title("EXPLORE — Define Exploration Behavior")
        self.window.resizable(True, True)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 5}

        # ---- Positive prompts ----
        pos_frame = ttk.LabelFrame(
            self.window,
            text="Exploration prompts  (describe what COUNTS as exploration)",
            padding=8,
        )
        pos_frame.pack(fill="both", expand=True, **pad)
        ttk.Label(
            pos_frame,
            text="One sentence per line.  Use multiple phrasings for robustness.",
            foreground="grey",
        ).pack(anchor="w")
        self._pos_text = tk.Text(pos_frame, height=6, wrap="word")
        self._pos_text.pack(fill="both", expand=True, pady=(4, 0))
        self._pos_text.insert("1.0", "\n".join(self.behavior.exploration_prompts))

        # ---- Negative prompts ----
        neg_frame = ttk.LabelFrame(
            self.window,
            text="Non-exploration prompts  (describe what does NOT count)",
            padding=8,
        )
        neg_frame.pack(fill="both", expand=True, **pad)
        ttk.Label(
            neg_frame,
            text="One sentence per line.",
            foreground="grey",
        ).pack(anchor="w")
        self._neg_text = tk.Text(neg_frame, height=4, wrap="word")
        self._neg_text.pack(fill="both", expand=True, pady=(4, 0))
        self._neg_text.insert("1.0", "\n".join(self.behavior.no_exploration_prompts))

        # ---- Threshold ----
        thr_frame = ttk.Frame(self.window, padding=8)
        thr_frame.pack(fill="x", **pad)
        ttk.Label(thr_frame, text="Confidence threshold:").pack(side="left")
        self._thr_var = tk.DoubleVar(value=self.behavior.confidence_threshold)
        ttk.Scale(
            thr_frame,
            from_=0.1,
            to=0.9,
            variable=self._thr_var,
            orient="horizontal",
            length=200,
        ).pack(side="left", padx=8)
        ttk.Label(thr_frame, textvariable=self._thr_var).pack(side="left")

        ttk.Label(thr_frame, text="  Min bout (s):").pack(side="left", padx=(16, 0))
        self._bout_var = tk.DoubleVar(value=self.behavior.min_bout_seconds)
        ttk.Spinbox(
            thr_frame,
            from_=0.0,
            to=10.0,
            increment=0.5,
            textvariable=self._bout_var,
            width=6,
        ).pack(side="left", padx=4)

        # ---- Buttons ----
        btn_frame = ttk.Frame(self.window, padding=8)
        btn_frame.pack(fill="x", **pad)
        ttk.Button(btn_frame, text="✓  Confirm", command=self._confirm).pack(
            side="right", padx=4
        )

    # ------------------------------------------------------------------
    # Confirm
    # ------------------------------------------------------------------

    def _confirm(self) -> None:
        pos = [
            line.strip()
            for line in self._pos_text.get("1.0", "end").splitlines()
            if line.strip()
        ]
        neg = [
            line.strip()
            for line in self._neg_text.get("1.0", "end").splitlines()
            if line.strip()
        ]

        if not pos or not neg:
            messagebox.showerror(
                "Error",
                "At least one exploration prompt AND one non-exploration prompt are required.",
            )
            return

        self.behavior.exploration_prompts = pos
        self.behavior.no_exploration_prompts = neg
        self.behavior.confidence_threshold = round(self._thr_var.get(), 2)
        self.behavior.min_bout_seconds = self._bout_var.get()

        self.window.destroy()
        if self.on_confirm:
            self.on_confirm(self.behavior)
