"""Object description and bounding-box confirmation dialog.

Users type natural-language descriptions of their objects.  When they click
"Detect", Grounding DINO runs and draws the found bounding boxes on the
reference frame.  Users can drag-adjust boxes before confirming.

The dialog is fully self-contained: it takes a list of ObjectConfig objects,
populates them with detected bounding boxes, and returns control to the caller.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from explore.config import ObjectConfig


class ObjectSetupDialog:
    """Dialog for describing and confirming object bounding boxes.

    Parameters
    ----------
    parent:
        Parent Tk window.
    reference_frame:
        BGR image used for display and detection.
    objects:
        List of ObjectConfig instances to populate.  Descriptions may already
        be filled in; bounding_box fields will be set after detection.
    on_confirm:
        Callback invoked with the updated ``objects`` list when the user
        clicks Confirm.
    """

    _COLORS = ["#00FF00", "#FF00FF", "#00FFFF", "#FF8000"]
    _DISPLAY_MAX_W = 800
    _DISPLAY_MAX_H = 600

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        reference_frame: np.ndarray,
        objects: list[ObjectConfig],
        on_confirm: callable | None = None,  # type: ignore[type-arg]
    ) -> None:
        self.parent = parent
        self.reference_frame = reference_frame
        self.objects = objects
        self.on_confirm = on_confirm

        self._scale = 1.0
        self._photo: ImageTk.PhotoImage | None = None
        self._drag_state: dict | None = None  # type: ignore[type-arg]

        self.window = tk.Toplevel(parent)
        self.window.title("EXPLORE 2.0 — Define Objects")
        self.window.resizable(True, True)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Left panel: object descriptions ----
        left = ttk.Frame(self.window, padding=10)
        left.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            left,
            text="Describe each object in plain English.\nGrounding DINO will find them automatically.",
            justify="left",
        ).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        self._desc_vars: list[tk.StringVar] = []
        self._name_vars: list[tk.StringVar] = []

        for i, obj in enumerate(self.objects):
            name_var = tk.StringVar(value=obj.name)
            desc_var = tk.StringVar(value=obj.description)
            self._name_vars.append(name_var)
            self._desc_vars.append(desc_var)

            color = self._COLORS[i % len(self._COLORS)]
            ttk.Label(left, text=f"Object {i + 1} name:").grid(
                row=2 * i + 1, column=0, sticky="e", padx=4
            )
            name_entry = ttk.Entry(left, textvariable=name_var, width=15)
            name_entry.grid(row=2 * i + 1, column=1, sticky="w", padx=4, pady=2)

            ttk.Label(left, text="Description:").grid(
                row=2 * i + 2, column=0, sticky="e", padx=4
            )
            desc_entry = ttk.Entry(left, textvariable=desc_var, width=40)
            desc_entry.grid(row=2 * i + 2, column=1, sticky="w", padx=4, pady=2)

            # Color swatch
            swatch = tk.Label(left, bg=color, width=2, relief="raised")
            swatch.grid(row=2 * i + 1, column=2, rowspan=2, padx=6)

        btn_row = len(self.objects) * 2 + 2
        detect_btn = ttk.Button(
            left, text="🔍  Detect Objects", command=self._run_detection
        )
        detect_btn.grid(row=btn_row, column=0, columnspan=3, pady=(12, 4), sticky="ew")

        confirm_btn = ttk.Button(left, text="✓  Confirm", command=self._confirm)
        confirm_btn.grid(row=btn_row + 1, column=0, columnspan=3, pady=4, sticky="ew")

        # ---- Right panel: canvas ----
        right = ttk.Frame(self.window, padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        h, w = self.reference_frame.shape[:2]
        self._scale = min(
            self._DISPLAY_MAX_W / w,
            self._DISPLAY_MAX_H / h,
            1.0,
        )
        disp_w = int(w * self._scale)
        disp_h = int(h * self._scale)

        self._canvas = tk.Canvas(right, width=disp_w, height=disp_h, cursor="crosshair")
        self._canvas.pack()
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        self._redraw()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _run_detection(self) -> None:
        # Update object names/descriptions from entry fields
        for i, obj in enumerate(self.objects):
            obj.name = self._name_vars[i].get().strip()
            obj.description = self._desc_vars[i].get().strip()

        descriptions = [o.description for o in self.objects]
        if not all(descriptions):
            messagebox.showerror("Error", "All object descriptions must be filled in.")
            return

        try:
            from explore.detection.object_detector import ObjectDetector

            detector = ObjectDetector()
            results = detector.detect(self.reference_frame, descriptions)
        except Exception as exc:
            messagebox.showerror("Detection failed", str(exc))
            return

        for obj, result in zip(self.objects, results, strict=False):
            obj.bounding_box = result.box

        self._redraw()

    # ------------------------------------------------------------------
    # Canvas drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        h, w = self.reference_frame.shape[:2]
        disp_w = int(w * self._scale)
        disp_h = int(h * self._scale)

        img = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img).resize((disp_w, disp_h), Image.LANCZOS)
        draw = ImageDraw.Draw(pil)

        for i, obj in enumerate(self.objects):
            if obj.bounding_box is None:
                continue
            x1, y1, x2, y2 = obj.bounding_box
            x1s, y1s = int(x1 * self._scale), int(y1 * self._scale)
            x2s, y2s = int(x2 * self._scale), int(y2 * self._scale)
            color = self._COLORS[i % len(self._COLORS)]
            draw.rectangle([x1s, y1s, x2s, y2s], outline=color, width=2)
            draw.text((x1s + 4, y1s + 2), obj.name, fill=color)

        self._photo = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ------------------------------------------------------------------
    # Drag-to-adjust bounding boxes
    # ------------------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._drag_state = {"x0": event.x, "y0": event.y, "obj_idx": None}
        # Find which object to adjust: the one whose box the user clicked inside
        for i, obj in enumerate(self.objects):
            if obj.bounding_box is None:
                continue
            x1, y1, x2, y2 = obj.bounding_box
            sx1, sy1 = int(x1 * self._scale), int(y1 * self._scale)
            sx2, sy2 = int(x2 * self._scale), int(y2 * self._scale)
            if sx1 <= event.x <= sx2 and sy1 <= event.y <= sy2:
                self._drag_state["obj_idx"] = i
                self._drag_state["orig_box"] = obj.bounding_box
                break

    def _on_drag_move(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._drag_state is None or self._drag_state["obj_idx"] is None:
            return
        idx = self._drag_state["obj_idx"]
        orig = self._drag_state["orig_box"]
        dx = int((event.x - self._drag_state["x0"]) / self._scale)
        dy = int((event.y - self._drag_state["y0"]) / self._scale)
        x1, y1, x2, y2 = orig
        self.objects[idx].bounding_box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        self._redraw()

    def _on_drag_end(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._drag_state = None

    # ------------------------------------------------------------------
    # Confirm
    # ------------------------------------------------------------------

    def _confirm(self) -> None:
        for i, obj in enumerate(self.objects):
            obj.name = self._name_vars[i].get().strip()
            obj.description = self._desc_vars[i].get().strip()

        if any(
            o.bounding_box is None for o in self.objects
        ) and not messagebox.askokcancel(
            "Missing bounding boxes",
            "Some objects have no bounding box detected. Continue anyway?",
        ):
            return

        self.window.destroy()
        if self.on_confirm:
            self.on_confirm(self.objects)
