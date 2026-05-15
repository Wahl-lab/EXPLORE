"""NiceGUI browser-based application for EXPLORE 2.0.

Workflow
--------
1. **Project** — name, output directory, video files, duration.
2. **Label Objects** — draw bounding boxes on a random reference frame;
   assign a name to each object.
3. **Verify Boxes** — ORB re-localization runs per video; annotated sample
   frames let the user confirm boxes look correct.
4. **Analyze** — configure CLIP prompts and run the full pipeline.
5. **Results** — view the results table; label uncertain frames for active
   learning and re-run.

Launch
------
>>> from explore.gui.nicegui_app import launch
>>> launch()
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from nicegui import run, ui

from explore.config import (
    AnalysisConfig,
    BehaviorConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectConfig,
)
from explore.detection.box_localizer import BoxLocalizer
from explore.pipeline.prediction import ExplorationPipeline

_LOGO_PATH = Path(__file__).parents[1] / "assets" / "explore_logo.png"


def _load_logo_data_url(height_px: int = 80) -> str:
    """Load the logo, auto-crop transparent margins, resize, return a PNG data-URL."""
    if not _LOGO_PATH.exists():
        return ""
    try:
        import numpy as _np
        from PIL import Image as _PILImage

        img = _PILImage.open(_LOGO_PATH).convert("RGBA")
        arr = _np.array(img)
        alpha = arr[:, :, 3]
        rows = _np.any(alpha > 10, axis=1)
        cols = _np.any(alpha > 10, axis=0)
        rmin, rmax = _np.where(rows)[0][[0, -1]]
        cmin, cmax = _np.where(cols)[0][[0, -1]]
        img = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        w, h = img.size
        new_w = int(w * height_px / h)
        img = img.resize((new_w, height_px), _PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


_LOGO_DATA_URL = _load_logo_data_url(height_px=200)

logger = logging.getLogger(__name__)

_BOX_COLORS = ["#00DCDC", "#DC00DC", "#28C828", "#DCA000", "#5050F0"]
_NOT_EXP = "not_exploring"


# ---------------------------------------------------------------------------
# Labeling-tab helpers (module-level so they run cleanly in io_bound threads)
# ---------------------------------------------------------------------------


def _dist_to_box_edge(cx: float, cy: float, box: tuple) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - cx, 0.0, cx - x2)
    dy = max(y1 - cy, 0.0, cy - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def _get_centroid(frame: np.ndarray, background: np.ndarray) -> tuple[int, int] | None:
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    if not contours:
        return None
    moments = cv2.moments(max(contours, key=cv2.contourArea))
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _make_thumbnail(
    frame: np.ndarray,
    boxes: dict[str, tuple],
    highlight: str | None,
    centroid: tuple[int, int] | None,
    width: int = 160,
) -> str:
    """Return a JPEG data-URL thumbnail with box overlays and centroid dot."""
    from PIL import Image

    h, w = frame.shape[:2]
    scale = width / w
    thumb = cv2.resize(frame.copy(), (width, int(h * scale)))
    for i, (name, box) in enumerate(boxes.items()):
        x1, y1, x2, y2 = (int(v * scale) for v in box)
        hx = _BOX_COLORS[i % len(_BOX_COLORS)].lstrip("#")
        bgr = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
        is_hl = name == highlight
        if is_hl:
            ov = thumb.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), bgr, -1)
            cv2.addWeighted(ov, 0.25, thumb, 0.75, 0, thumb)
        cv2.rectangle(thumb, (x1, y1), (x2, y2), bgr, 2 if is_hl else 1)
    if centroid is not None:
        cv2.circle(
            thumb,
            (int(centroid[0] * scale), int(centroid[1] * scale)),
            4,
            (0, 0, 255),
            -1,
        )
    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _quick_background(
    video_path: Path, n: int = 15, skip_s: float = 30.0
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = int(skip_s * fps_v)
    end = max(start + 1, total - start)
    step = max(1, (end - start) // n)
    frames = []
    for i in range(n):
        idx = start + i * step
        if idx >= end:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0).astype(np.float32), axis=0).astype(
        np.uint8
    )


def _sample_label_window(
    video_path_str: str,
    video_boxes: dict[str, tuple],
    start_s: float,
    duration_s: float = 60.0,
) -> list[dict]:
    """Sample all frames from a contiguous window at ~4 fps, assign proximity labels.

    Each returned dict has: frame, time_s, label, thumb, thumb_lg.
    Labels default to *_NOT_EXP*; a frame is labelled as the nearest object only
    when the detected animal centroid is within 100 px of that object's box edge.
    """
    bg = _quick_background(Path(video_path_str))
    cap = cv2.VideoCapture(video_path_str)
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(start_s * fps_v)
    end_f = min(total, int((start_s + duration_s) * fps_v))
    step = max(1, round(fps_v / 4.0))

    results: list[dict] = []
    idx = start_f
    while idx < end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        time_s = idx / fps_v
        label = _NOT_EXP
        centroid = None

        if bg is not None:
            centroid = _get_centroid(frame, bg)
            if centroid is not None and video_boxes:
                dists = {
                    n: _dist_to_box_edge(centroid[0], centroid[1], b)
                    for n, b in video_boxes.items()
                }
                nearest = min(dists, key=lambda k: dists[k])
                if dists[nearest] < 100:
                    label = nearest

        hl = label if label != _NOT_EXP else None
        results.append(
            {
                "frame": frame,
                "time_s": time_s,
                "label": label,
                "thumb": _make_thumbnail(frame, video_boxes, hl, centroid, width=120),
                "thumb_lg": _make_thumbnail(
                    frame, video_boxes, hl, centroid, width=400
                ),
            }
        )
        idx += step

    cap.release()
    return results


def _predict_label_window(
    clf,
    head_class_names: list[str],
    video_path_str: str,
    video_boxes: dict[str, tuple],
    start_s: float,
    duration_s: float = 60.0,
) -> list[dict]:
    """Sample a window and predict labels with the trained classifier.

    Returns the same frame-dict format as *_sample_label_window* so the
    same strip widget can render both proximity-labeled and model-predicted
    windows without distinction.
    """
    cap = cv2.VideoCapture(video_path_str)
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(start_s * fps_v)
    end_f = min(total, int((start_s + duration_s) * fps_v))
    step = max(1, round(fps_v / 4.0))

    raw_frames: list = []
    timestamps: list[float] = []
    idx = start_f
    while idx < end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
        timestamps.append(idx / fps_v)
        idx += step
    cap.release()

    if not raw_frames:
        return []

    embeddings = clf.embed_frames(raw_frames, show_progress=False)
    indices = clf.predict_class_indices(embeddings)
    labels = [head_class_names[int(i)] for i in indices]

    results = []
    for f, t, lbl in zip(raw_frames, timestamps, labels, strict=False):
        hl = lbl if lbl != _NOT_EXP else None
        results.append(
            {
                "frame": f,
                "time_s": t,
                "label": lbl,
                "thumb": _make_thumbnail(f, video_boxes, hl, None, width=120),
                "thumb_lg": _make_thumbnail(f, video_boxes, hl, None, width=400),
            }
        )
    return results


def _build_timeline_html(
    labels: list[str],
    obj_names: list[str],
    start_s: float = 0.0,
) -> str:
    """Return an HTML timeline bar for a sequence of per-frame class labels."""
    if not labels:
        return "<p style='color:#999;font-size:12px;font-style:italic;'>No frames.</p>"

    color_map = {n: _BOX_COLORS[i % len(_BOX_COLORS)] for i, n in enumerate(obj_names)}
    color_map[_NOT_EXP] = "#999999"

    runs: list[tuple[str, int]] = []
    cur, cnt = labels[0], 1
    for lbl in labels[1:]:
        if lbl == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = lbl, 1
    runs.append((cur, cnt))

    total = len(labels)
    fps_eff = 4.0
    duration_s_actual = total / fps_eff
    end_s = start_s + duration_s_actual

    segs = []
    for lbl, cnt in runs:
        c = color_map.get(lbl, "#cccccc")
        dur = cnt / fps_eff
        segs.append(
            f'<div style="flex:{cnt};background:{c};" title="{lbl}: {dur:.1f}s"></div>'
        )

    legend_parts = []
    for i, name in enumerate(obj_names):
        c = _BOX_COLORS[i % len(_BOX_COLORS)]
        legend_parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;">'
            f'<span style="width:12px;height:12px;border-radius:2px;background:{c};'
            f'display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-size:12px;color:#444;">{name}</span></span>'
        )
    legend_parts.append(
        '<span style="display:inline-flex;align-items:center;gap:4px;">'
        '<span style="width:12px;height:12px;border-radius:2px;background:#999;'
        'display:inline-block;flex-shrink:0;"></span>'
        '<span style="font-size:12px;color:#444;">not exploring</span></span>'
    )

    exp_s = (
        sum(cnt for lbl, cnt in runs if lbl != _NOT_EXP and lbl in color_map) / fps_eff
    )
    stats_parts = [f"exploring: {exp_s:.1f}s / {duration_s_actual:.0f}s"]
    for name in obj_names:
        t = sum(cnt for lbl, cnt in runs if lbl == name) / fps_eff
        if t > 0:
            stats_parts.append(f"{name}: {t:.1f}s")

    return (
        f'<div style="width:100%;padding:4px 0 8px 0;">'
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">{"".join(legend_parts)}</div>'
        f'<div style="width:100%;height:36px;display:flex;border-radius:4px;overflow:hidden;border:1px solid #e0e0e0;">{"".join(segs)}</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#aaa;margin-top:3px;">'
        f"<span>{start_s:.0f}s</span>"
        f'<span style="color:#666;">{"  ·  ".join(stats_parts)}</span>'
        f"<span>{end_s:.0f}s</span>"
        f"</div>"
        f"</div>"
    )


_DEFAULT_EXPLORATION = [
    "top-down view of a mouse sniffing and investigating a small object closely",
    "aerial view of a rodent with its nose touching an object, actively exploring",
    "overhead camera: a mouse head close to an object, whiskers touching it",
]
_DEFAULT_NO_EXPLORATION = [
    "top-down view of a mouse walking through the arena away from objects",
    "aerial view of a rodent resting or grooming in the center of the arena",
    "overhead camera: a mouse far from any object, not investigating",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_duration_minutes(video_paths: list[str]) -> int | None:
    """Return the floor-minutes of the shortest video, or None if unreadable."""
    min_s: float | None = None
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if frames > 0 and fps > 0:
            dur = frames / fps
            if min_s is None or dur < min_s:
                min_s = dur
    if min_s is None:
        return None
    return max(1, int(min_s // 60))


def _bgr_to_data_url(frame: np.ndarray) -> str:
    """Convert a BGR numpy frame to a JPEG data-URL string."""
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _random_frame(video_path: Path, skip_s: float = 60.0) -> np.ndarray | None:
    """Return a random BGR frame from the middle of *video_path*."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = int(min(skip_s * fps, total * 0.25))
    end = total - int(min(skip_s * fps, total * 0.25))
    if end <= start:
        start, end = 0, max(1, total)
    idx = random.randint(start, max(start, end - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _pick_files(title: str = "Select video files") -> list[str]:
    """Open a native file-chooser dialog (uses tkinter stdlib — no extra dep)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return list(paths)


def _pick_directory(title: str = "Select output directory") -> str:
    """Open a native folder-chooser dialog."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path or ""


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class ExploreApp:
    """State container and UI builder for the EXPLORE 2.0 NiceGUI app."""

    def __init__(self) -> None:
        # --- Project state ---
        self.project_name: str = ""
        self.project_path: str = str(Path.home() / "explore_projects")
        self.video_paths: list[str] = []
        self.video_duration: int = 5

        # --- Labeling state ---
        self.reference_video_idx: int = 0
        self.reference_frame: np.ndarray | None = None
        self.reference_frame_url: str = ""
        self.objects: list[dict] = []  # [{name, box: (x1,y1,x2,y2)}]

        # --- Box-drawing state ---
        self._drawing: bool = False
        self._draw_start: tuple[float, float] | None = None
        self._draw_current: tuple[float, float] | None = None
        self._pending_box: tuple[int, int, int, int] | None = None

        # --- Per-video localized boxes ---
        # {video_path_str: {obj_name: (x1,y1,x2,y2)}}
        self.video_boxes: dict[str, dict[str, tuple[int, int, int, int]]] = {}
        # {video_path_str: {obj_name: LocalizationResult}} — for quality display
        self._video_results: dict[str, dict] = {}

        # --- Behavior config ---
        self.exploration_prompts: list[str] = list(_DEFAULT_EXPLORATION)
        self.no_exploration_prompts: list[str] = list(_DEFAULT_NO_EXPLORATION)
        self.confidence_threshold: float = 0.5
        self.min_bout_seconds: float = 0.0

        # --- Analysis config ---
        self.bin_duration_seconds: int = 60
        self.compute_di: bool = True
        self.pred_video_hires: bool = False

        # --- Labeling / training loop (Tab 4) ---
        self._label_window_video: str = ""
        self._manual_start_s: float | None = None  # None = random
        self._current_window: dict | None = None  # window being shown/edited
        self._training_pool: list[dict] = []  # past windows saved to pool
        self._head_classes: list[str] = []
        self._head_trained: bool = False
        self._trained_clf = None

        # --- Pipeline ---
        self._pipeline: ExplorationPipeline | None = None

        # --- UI refs (set during build) ---
        self._img: ui.interactive_image | None = None
        self._name_dialog: ui.dialog | None = None
        self._name_input: ui.input | None = None
        self._objects_panel_refresh: callable | None = None
        self._verify_container: ui.column | None = None
        self._current_window_refresh: callable | None = None
        self._label_video_select_refresh: callable | None = None
        self._sample_status: ui.label | None = None
        self._train_status: ui.label | None = None
        self._di_ri_refresh: callable | None = None
        self._log: ui.log | None = None
        self._progress: ui.linear_progress | None = None
        self._run_btn: ui.button | None = None
        self._results_container: ui.column | None = None
        self._video_list_refresh: callable | None = None
        self._tabs: ui.tabs | None = None

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def build(self) -> None:
        ui.query("body").classes("bg-gray-50")

        with ui.header().classes("bg-blue-800 text-white items-center px-6 gap-4"):
            if _LOGO_DATA_URL:
                ui.html(
                    f'<img src="{_LOGO_DATA_URL}" '
                    f'style="height:50px;width:auto;object-fit:contain;" />'
                )
            ui.label("EXPLORE").classes("text-xl font-bold tracking-wide")
            ui.label("Object Tests — behavior analysis").classes("text-sm opacity-60")

        with ui.tabs().classes("w-full bg-white shadow-sm") as tabs:
            self._tabs = tabs
            t1 = ui.tab("1. Project", icon="folder_open")
            t2 = ui.tab("2. Label Objects", icon="crop_free")
            t3 = ui.tab("3. Verify Boxes", icon="check_circle_outline")
            t4 = ui.tab("4. Label Frames", icon="photo_library")
            t5 = ui.tab("5. Analyze", icon="analytics")
            t6 = ui.tab("6. Results", icon="bar_chart")

        with ui.tab_panels(tabs, value=t1).classes("w-full"):
            with ui.tab_panel(t1):
                self._build_project_tab(tabs, t2)
            with ui.tab_panel(t2):
                self._build_objects_tab(tabs, t3)
            with ui.tab_panel(t3):
                self._build_verify_tab(tabs, t4)
            with ui.tab_panel(t4):
                self._build_label_frames_tab(tabs, t5)
            with ui.tab_panel(t5):
                self._build_analyze_tab()
            with ui.tab_panel(t6):
                self._build_results_tab()

        # Name-entry dialog (shared across tabs)
        with ui.dialog() as self._name_dialog, ui.card().classes("p-4 gap-3 min-w-64"):
            ui.label("Name this object").classes("text-base font-semibold")
            self._name_input = (
                ui.input(placeholder="e.g. familiar  /  novel  /  object_1")
                .classes("w-full")
                .props("autofocus")
            )
            with ui.row().classes("gap-2 justify-end"):
                ui.button("Cancel", on_click=self._cancel_draw).props("flat color=grey")
                ui.button("Add", icon="check", on_click=self._confirm_add).classes(
                    "bg-green-600 text-white"
                )

    # ------------------------------------------------------------------
    # Tab 1: Project
    # ------------------------------------------------------------------

    def _build_project_tab(self, tabs: ui.tabs, next_tab: ui.tab) -> None:
        with ui.column().classes("w-full max-w-xl mx-auto mt-8 gap-4 px-4"):
            ui.label("Project Setup").classes("text-xl font-semibold")

            # ── Restore existing session ─────────────────────────────────
            with ui.row().classes("items-center gap-2"):
                ui.icon("restore").classes("text-gray-400 text-base")
                ui.label("Resume an existing project:").classes("text-sm text-gray-500")
                ui.button(
                    "Load project",
                    icon="folder_open",
                    on_click=self._load_project_dialog,
                ).props("flat color=primary dense")

            with ui.card().classes("w-full p-4 gap-3"):
                ui.label("Project name").classes("text-sm text-gray-500")
                ui.input(placeholder="e.g. NOR_cohort_A").classes("w-full").bind_value(
                    self, "project_name"
                )

                ui.label("Output directory").classes("text-sm text-gray-500 mt-2")
                with ui.row().classes("w-full gap-2 items-center"):
                    ui.input(placeholder="/path/to/output").classes(
                        "flex-grow"
                    ).bind_value(self, "project_path")
                    ui.button(
                        icon="folder",
                        on_click=self._browse_output_dir,
                    ).props("flat round").tooltip("Browse")

            with ui.card().classes("w-full p-4 gap-3"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Video files").classes("text-base font-medium")
                    ui.button(
                        "Add videos", icon="add", on_click=self._add_videos
                    ).props("flat color=primary")

                @ui.refreshable
                def video_list() -> None:
                    if not self.video_paths:
                        ui.label("No videos added yet.").classes(
                            "text-gray-400 italic text-sm"
                        )
                        return
                    for i, p in enumerate(self.video_paths):
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.icon("videocam").classes("text-blue-400 text-base")
                            with ui.column().classes("flex-grow min-w-0"):
                                ui.label(Path(p).name).classes(
                                    "text-sm font-medium truncate"
                                )
                                ui.label(str(Path(p).parent)).classes(
                                    "text-xs text-gray-400 truncate"
                                )
                            ui.button(
                                icon="close",
                                on_click=lambda i=i: self._remove_video(i, video_list),
                            ).props("flat round dense color=red").tooltip("Remove")

                self._video_list_refresh = video_list.refresh
                video_list()

            with ui.card().classes("w-full p-4 gap-3"):
                ui.label("Video duration (minutes)").classes("text-sm text-gray-500")
                ui.label(
                    "Auto-filled when videos are added — adjust if needed."
                ).classes("text-xs text-gray-400")
                ui.number(min=1, max=120, step=1).classes("w-32").bind_value(
                    self, "video_duration"
                )

            ui.button(
                "Next: Label Objects →",
                icon="navigate_next",
                on_click=lambda: self._go_to_label(tabs, next_tab),
            ).classes("bg-blue-600 text-white self-end")

    # ------------------------------------------------------------------
    # Tab 2: Label Objects
    # ------------------------------------------------------------------

    def _build_objects_tab(self, tabs: ui.tabs, next_tab: ui.tab) -> None:
        with ui.splitter(value=68).classes("w-full h-full") as splitter:
            with splitter.before:  # noqa: SIM117
                with ui.column().classes("w-full h-full p-3 gap-2"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("Reference video:").classes("text-sm text-gray-600")
                        ref_sel = ui.select(
                            options=[],
                            value=None,
                            on_change=lambda e: setattr(
                                self,
                                "reference_video_idx",
                                e.value if e.value is not None else 0,
                            ),
                        ).classes("flex-grow")
                        ui.button(
                            "Get frame",
                            icon="refresh",
                            on_click=lambda: self._load_reference_frame(ref_sel),
                        ).props("flat color=primary")

                    ui.label(
                        "Click and drag on the image to draw a box around each object."
                    ).classes("text-xs text-gray-400")

                    self._img = ui.interactive_image(
                        source=self.reference_frame_url or "",
                        content="",
                        on_mouse=self._on_mouse,
                        events=["mousedown", "mousemove", "mouseup"],
                    ).classes("w-full border border-gray-200 rounded-lg shadow-sm")

                    self._ref_sel = ref_sel

            with splitter.after:  # noqa: SIM117
                with ui.column().classes("w-full h-full p-4 gap-3"):
                    ui.label("Objects").classes("text-base font-semibold")
                    ui.label("Draw a box, then enter a name for each object.").classes(
                        "text-xs text-gray-400"
                    )

                    @ui.refreshable
                    def objects_panel() -> None:
                        if not self.objects:
                            ui.label("No objects yet.").classes(
                                "text-gray-400 italic text-sm"
                            )
                            return
                        for i, obj in enumerate(self.objects):
                            c = _BOX_COLORS[i % len(_BOX_COLORS)]
                            with ui.row().classes("items-center gap-2 w-full"):
                                ui.element("div").style(
                                    f"width:14px;height:14px;border-radius:3px;"
                                    f"background:{c};flex-shrink:0"
                                )
                                x1, y1, x2, y2 = obj["box"]
                                ui.label(
                                    f"{obj['name']}  [{x1},{y1} → {x2},{y2}]"
                                ).classes("flex-grow text-sm font-mono")
                                ui.button(
                                    icon="delete",
                                    on_click=lambda i=i: self._delete_object(
                                        i, objects_panel
                                    ),
                                ).props("flat round dense color=red")

                    self._objects_panel_refresh = objects_panel.refresh
                    objects_panel()

                    ui.separator().classes("my-1")

                    ui.button(
                        "Next: Verify Boxes →",
                        icon="navigate_next",
                        on_click=lambda: self._go_to_verify(tabs, next_tab),
                    ).classes("bg-blue-600 text-white mt-auto")

        # Populate video selector (may not have been filled yet)
        self._update_ref_select()

    # ------------------------------------------------------------------
    # Tab 3: Verify Boxes
    # ------------------------------------------------------------------

    def _build_verify_tab(self, tabs: ui.tabs, next_tab: ui.tab) -> None:
        with ui.column().classes("w-full p-4 gap-4"):
            ui.label("Verify Bounding Boxes").classes("text-xl font-semibold")
            ui.label(
                "A random frame from each video is shown with the ORB-localized boxes. "
                "The reference video always uses your drawn boxes directly."
            ).classes("text-sm text-gray-500")

            with ui.row().classes("gap-2 mt-2"):
                ui.button(
                    "Run localization",
                    icon="search",
                    on_click=self._run_localization,
                ).classes("bg-indigo-600 text-white")
                ui.button(
                    "Next: Analyze →",
                    icon="navigate_next",
                    on_click=lambda: [self._save_session(), tabs.set_value(next_tab)],
                ).classes("bg-blue-600 text-white")

            self._verify_container = ui.column().classes("w-full gap-4 mt-2")

    # ------------------------------------------------------------------
    # Tab 4: Label Frames
    # ------------------------------------------------------------------

    def _build_label_frames_tab(self, tabs: ui.tabs, next_tab: ui.tab) -> None:
        with ui.column().classes("w-full max-w-5xl mx-auto p-4 gap-4"):
            ui.label("Label Training Frames").classes("text-xl font-semibold")
            ui.label(
                "Sample a 1-minute window → correct labels → Train + Preview. "
                "Each train round saves the current window and shows a new one predicted "
                "by the model. Repeat until labels look correct."
            ).classes("text-sm text-gray-500")

            # ── Sampling controls ────────────────────────────────────────
            with ui.card().classes("w-full p-3 gap-2"):  # noqa: SIM117
                with ui.row().classes("gap-3 items-end flex-wrap"):
                    with ui.column().classes("gap-1"):
                        ui.label("Video").classes("text-xs text-gray-500")

                        @ui.refreshable
                        def video_select_area() -> None:
                            video_options = {p: Path(p).name for p in self.video_paths}
                            if (
                                self.video_paths
                                and self._label_window_video not in video_options
                            ):
                                self._label_window_video = self.video_paths[0]
                            ui.select(
                                options=video_options
                                if video_options
                                else {"": "No videos yet"},
                                value=self._label_window_video
                                if self._label_window_video in video_options
                                else None,
                                on_change=lambda e: setattr(
                                    self, "_label_window_video", e.value
                                ),
                            ).classes("w-52").props("dense outlined")

                        self._label_video_select_refresh = video_select_area.refresh
                        video_select_area()
                    with ui.column().classes("gap-1"):
                        ui.label("Start (s) — leave blank for random").classes(
                            "text-xs text-gray-500"
                        )
                        ui.number(
                            min=0,
                            step=1,
                            placeholder="random",
                        ).classes("w-32").props("dense outlined clearable").bind_value(
                            self, "_manual_start_s"
                        )
                    ui.button(
                        "Sample window",
                        icon="video_label",
                        on_click=self._do_sample_window,
                    ).classes("bg-indigo-600 text-white")
                    self._sample_status = ui.label("").classes(
                        "text-sm text-gray-400 italic self-end pb-1"
                    )

            # ── Working window strip ─────────────────────────────────────
            @ui.refreshable
            def current_window_area() -> None:
                if self._current_window is None:
                    ui.label("Click 'Sample window' to start.").classes(
                        "text-gray-400 italic text-sm py-4"
                    )
                    return

                win = self._current_window
                all_labels = [o["name"] for o in self.objects] + [_NOT_EXP]
                options = {lbl: lbl.replace("_", " ") for lbl in all_labels}
                color_map = {
                    n: _BOX_COLORS[i % len(_BOX_COLORS)]
                    for i, n in enumerate(all_labels)
                }
                color_map[_NOT_EXP] = "#888888"

                dur = len(win["frames"]) / 4.0
                is_predicted = win.get("predicted", False)
                source_icon = "model_training" if is_predicted else "video_label"
                source_label = (
                    "model predictions" if is_predicted else "proximity labels"
                )

                with ui.card().classes("w-full p-2 gap-2 bg-gray-50"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(source_icon).classes("text-indigo-400 text-base")
                        ui.label(
                            f"{Path(win['video']).name}  ·  "
                            f"{win['start_s']:.0f}s – {win['start_s'] + dur:.0f}s  ·  "
                            f"{len(win['frames'])} frames  ·  {source_label}"
                        ).classes("text-xs font-medium text-gray-600")

                    # Compact timeline bar
                    obj_names = [o["name"] for o in self.objects]
                    ui.html(
                        _build_timeline_html(
                            [f["label"] for f in win["frames"]],
                            obj_names,
                            win["start_s"],
                        )
                    ).classes("w-full")

                    # Scrollable frame strip with editable labels
                    with ui.element("div").style(  # noqa: SIM117
                        "overflow-x:auto; overflow-y:hidden; width:100%;"
                        "border:1px solid #e5e7eb; border-radius:6px; background:#f9fafb;"
                    ):
                        with ui.row().classes("flex-nowrap items-start p-2 gap-1"):
                            for frame_d in win["frames"]:
                                c = color_map.get(frame_d["label"], "#ccc")
                                with (
                                    ui.column()
                                    .classes("items-center gap-0")
                                    .style("min-width:92px; max-width:92px;")
                                ):
                                    img = (
                                        ui.image(frame_d["thumb"])
                                        .style(
                                            "width:88px; height:66px;"
                                            "object-fit:cover; border-radius:3px;"
                                        )
                                        .classes("cursor-zoom-in")
                                    )
                                    with (
                                        img,
                                        ui.tooltip().classes(
                                            "bg-transparent shadow-none p-0"
                                        ),
                                    ):
                                        ui.image(frame_d["thumb_lg"]).style(
                                            "width:400px; height:auto;"
                                            "border-radius:6px;"
                                            "box-shadow:0 4px 20px rgba(0,0,0,0.4);"
                                        )
                                    ui.element("div").style(
                                        f"width:88px; height:4px; background:{c};"
                                        "border-radius:2px; margin:2px 0;"
                                    )
                                    ui.label(f"{frame_d['time_s']:.0f}s").style(
                                        "font-size:8px; color:#aaa; text-align:center;"
                                    )
                                    ui.select(
                                        options=options,
                                        value=frame_d["label"]
                                        if frame_d["label"] in options
                                        else _NOT_EXP,
                                        on_change=lambda e, fd=frame_d: fd.update(
                                            {"label": e.value}
                                        ),
                                    ).style("width:88px; font-size:9px;").props(
                                        "dense outlined"
                                    )

            self._current_window_refresh = current_window_area.refresh
            current_window_area()

            # ── Training pool info ───────────────────────────────────────
            with ui.column().classes("w-full gap-1 mt-1"):
                pool_summary = ui.label("").classes("text-xs text-gray-400")
                pool_classes = ui.label("").classes("text-xs text-gray-500")

                def _update_pool_label():
                    all_frames = [f for w in self._training_pool for f in w["frames"]]
                    n = len(all_frames)
                    if not self._training_pool:
                        pool_summary.text = "Training pool: empty"
                        pool_classes.text = ""
                        return
                    pool_summary.text = f"Training pool: {n} frames from {len(self._training_pool)} window(s)"
                    counts: dict[str, int] = {}
                    for f in all_frames:
                        counts[f["label"]] = counts.get(f["label"], 0) + 1
                    all_labels = [o["name"] for o in self.objects] + [_NOT_EXP]
                    parts = [
                        f"{lbl.replace('_', ' ')}: {counts.get(lbl, 0)}"
                        for lbl in all_labels
                    ]
                    pool_classes.text = "  ·  ".join(parts)

                _update_pool_label()
                self._pool_label_update = _update_pool_label

                with ui.row().classes("items-center gap-2 mt-0"):
                    ui.button(
                        "Reset pool",
                        icon="delete_sweep",
                        on_click=lambda: [
                            self._training_pool.clear(),
                            _update_pool_label(),
                        ],
                    ).props("flat dense color=red").classes("text-xs")

            ui.separator()

            # ── Train + Preview ──────────────────────────────────────────
            with ui.row().classes("gap-3 items-center flex-wrap"):
                ui.button(
                    "Train + Preview",
                    icon="model_training",
                    on_click=self._train_and_preview,
                ).classes("bg-green-600 text-white")
                self._train_status = ui.label("").classes("text-sm text-gray-500")

            ui.button(
                "Next: Analyze →",
                icon="navigate_next",
                on_click=lambda: [
                    self._save_session(),
                    tabs.set_value(next_tab),
                    self._di_ri_refresh() if self._di_ri_refresh else None,
                ],
            ).classes("bg-blue-600 text-white mt-2")

    # ------------------------------------------------------------------
    # Tab 5: Analyze
    # ------------------------------------------------------------------

    def _build_analyze_tab(self) -> None:
        with ui.column().classes("w-full max-w-2xl mx-auto p-4 gap-4"):
            ui.label("Analysis Settings").classes("text-xl font-semibold")

            # Classifier status banner
            def _clf_status(trained: bool) -> str:
                if trained:
                    n = len([c for c in self._head_classes if c != _NOT_EXP])
                    return f"✓ Classifier trained — {n} object class(es)"
                return "⚠ No classifier trained yet — complete Step 4 for accurate results."

            ui.label().bind_text_from(self, "_head_trained", _clf_status).classes(
                "text-sm px-3 py-2 rounded border bg-gray-50 text-gray-700"
            )

            with ui.row().classes("gap-6 mt-1"), ui.column().classes("gap-1"):
                ui.label("Min exploration bout (s)").classes("text-sm text-gray-600")
                ui.number(min=0, max=10, step=0.5).bind_value(
                    self, "min_bout_seconds"
                ).classes("w-28")

            @ui.refreshable
            def di_ri_options() -> None:
                obj_names = [o["name"] for o in self.objects]
                with ui.row().classes("items-center gap-3 flex-wrap"):
                    ui.checkbox("Compute DI / RI per object pair").bind_value(
                        self, "compute_di"
                    )
                    if obj_names and len(obj_names) >= 2:
                        pairs = [
                            f"{a} vs {b}"
                            for i, a in enumerate(obj_names)
                            for b in obj_names[i + 1 :]
                        ]
                        ui.label(f"→ {', '.join(pairs)}").classes(
                            "text-xs text-gray-400 italic self-center"
                        )
                    elif len(obj_names) < 2:
                        ui.label("Need at least 2 labeled objects.").classes(
                            "text-xs text-yellow-600 self-center"
                        )

            self._di_ri_refresh = di_ri_options.refresh
            di_ri_options()

            with ui.expansion("Bin duration", icon="schedule").classes(
                "w-full border rounded-lg"
            ):
                ui.label("Bin duration (s)").classes("text-sm")
                ui.number(min=10, max=3600, step=10).bind_value(
                    self, "bin_duration_seconds"
                ).classes("w-28")

            with ui.row().classes("gap-3 items-center mt-2 flex-wrap"):
                self._run_btn = ui.button(
                    "▶ Run Analysis", icon="play_arrow", on_click=self._run_analysis
                ).classes("bg-green-600 text-white")
                ui.toggle(
                    {False: "Low-res (fast)", True: "High-res"},
                    on_change=lambda e: setattr(self, "pred_video_hires", e.value),
                ).bind_value(self, "pred_video_hires").props("dense")

            self._progress = ui.linear_progress(value=0, show_value=False).classes(
                "w-full"
            )
            self._progress.set_visibility(False)

            ui.label("Log").classes("text-sm font-medium text-gray-600 mt-2")
            self._log = ui.log(max_lines=80).classes(
                "w-full h-48 font-mono text-xs bg-gray-900 text-green-300 rounded"
            )

    # ------------------------------------------------------------------
    # Tab 5: Results
    # ------------------------------------------------------------------

    def _build_results_tab(self) -> None:
        with ui.column().classes("w-full p-4 gap-4"):
            ui.label("Results").classes("text-xl font-semibold")
            self._results_container = ui.column().classes("w-full")
            with self._results_container:
                ui.label("No results yet — run analysis first.").classes(
                    "text-gray-400 italic"
                )

    # ------------------------------------------------------------------
    # Event handlers — Project tab
    # ------------------------------------------------------------------

    def _browse_output_dir(self) -> None:
        path = _pick_directory("Select output directory")
        if path:
            self.project_path = path

    def _load_project_dialog(self) -> None:
        """Open a folder picker; if it contains session.json, restore state."""
        folder = _pick_directory("Select project folder")
        if not folder:
            return
        session_file = Path(folder) / "session.json"
        if not session_file.exists():
            ui.notify("No session.json found in that folder.", color="warning")
            return
        try:
            self._load_session(session_file)
        except Exception as exc:
            logger.exception("Failed to load session")
            ui.notify(f"Could not load session: {exc}", color="negative")

    def _add_videos(self) -> None:
        paths = _pick_files("Select video files")
        for p in paths:
            if p not in self.video_paths:
                self.video_paths.append(p)
        if self._video_list_refresh:
            self._video_list_refresh()
        if self._label_video_select_refresh:
            self._label_video_select_refresh()
        self._update_ref_select()
        self._update_duration_from_videos()

    def _remove_video(self, idx: int, refresh_fn) -> None:
        if 0 <= idx < len(self.video_paths):
            self.video_paths.pop(idx)
        refresh_fn()
        if self._label_video_select_refresh:
            self._label_video_select_refresh()
        self._update_ref_select()
        self._update_duration_from_videos()

    def _update_duration_from_videos(self) -> None:
        inferred = _infer_duration_minutes(self.video_paths)
        if inferred is not None:
            self.video_duration = inferred

    def _go_to_label(self, tabs: ui.tabs, next_tab: ui.tab) -> None:
        if not self.project_name.strip():
            ui.notify("Please enter a project name.", color="warning")
            return
        if not self.video_paths:
            ui.notify("Please add at least one video.", color="warning")
            return
        self._update_ref_select()
        self._save_session()
        tabs.set_value(next_tab)

    # ------------------------------------------------------------------
    # Event handlers — Label tab
    # ------------------------------------------------------------------

    def _update_ref_select(self) -> None:
        if not hasattr(self, "_ref_sel") or self._ref_sel is None:
            return
        options = {i: Path(p).name for i, p in enumerate(self.video_paths)}
        self._ref_sel.options = options
        if self.video_paths:
            self._ref_sel.value = 0

    def _load_reference_frame(self, ref_sel: ui.select) -> None:
        if not self.video_paths:
            ui.notify("No videos loaded.", color="warning")
            return
        idx = ref_sel.value if ref_sel.value is not None else 0
        self.reference_video_idx = int(idx)
        video_path = Path(self.video_paths[self.reference_video_idx])
        ui.notify(f"Loading frame from {video_path.name}…", color="info", timeout=2000)

        async def _load() -> None:
            frame = await run.io_bound(_random_frame, video_path)
            if frame is None:
                ui.notify("Could not read frame from video.", color="negative")
                return
            self.reference_frame = frame
            self.reference_frame_url = _bgr_to_data_url(frame)
            if self._img is not None:
                self._img.set_source(self.reference_frame_url)
                self._update_svg()

        asyncio.ensure_future(_load())

    def _on_mouse(self, e) -> None:  # type: ignore[no-untyped-def]
        if e.type == "mousedown":
            self._drawing = True
            self._draw_start = (e.image_x, e.image_y)
            self._draw_current = (e.image_x, e.image_y)

        elif e.type == "mousemove":
            if self._drawing:
                self._draw_current = (e.image_x, e.image_y)
                self._update_svg()

        elif e.type == "mouseup":
            if self._drawing and self._draw_start:
                sx, sy = self._draw_start
                cx, cy = e.image_x, e.image_y
                if abs(cx - sx) > 8 and abs(cy - sy) > 8:
                    self._pending_box = (
                        int(min(sx, cx)),
                        int(min(sy, cy)),
                        int(max(sx, cx)),
                        int(max(sy, cy)),
                    )
                    if self._name_input:
                        self._name_input.value = ""
                    if self._name_dialog:
                        self._name_dialog.open()
            self._drawing = False
            self._draw_start = None
            self._draw_current = None
            self._update_svg()

    def _update_svg(self) -> None:
        if self._img is None:
            return
        parts: list[str] = []

        # Drop-shadow filter for text legibility
        parts.append(
            '<defs><filter id="sh"><feDropShadow dx="0" dy="0" stdDeviation="2" '
            'flood-color="black" flood-opacity="0.9"/></filter></defs>'
        )

        # Confirmed boxes
        for i, obj in enumerate(self.objects):
            c = _BOX_COLORS[i % len(_BOX_COLORS)]
            x1, y1, x2, y2 = obj["box"]
            w_, h_ = x2 - x1, y2 - y1
            parts.append(
                f'<rect x="{x1}" y="{y1}" width="{w_}" height="{h_}" '
                f'fill="{c}33" stroke="{c}" stroke-width="2" rx="2"/>'
            )
            label_y = max(y1 - 5, 16)
            parts.append(
                f'<text x="{x1 + 4}" y="{label_y}" fill="{c}" '
                f'font-size="14" font-family="sans-serif" font-weight="bold" '
                f'filter="url(#sh)">{obj["name"]}</text>'
            )

        # In-progress draw rectangle
        if self._drawing and self._draw_start and self._draw_current:
            sx, sy = self._draw_start
            cx, cy = self._draw_current
            dx, dy = min(sx, cx), min(sy, cy)
            dw, dh = abs(cx - sx), abs(cy - sy)
            parts.append(
                f'<rect x="{dx}" y="{dy}" width="{dw}" height="{dh}" '
                f'fill="#FFFF0015" stroke="#FFE040" stroke-width="2" '
                f'stroke-dasharray="8 4" rx="2"/>'
            )

        self._img.set_content("".join(parts))

    def _confirm_add(self) -> None:
        name = (self._name_input.value or "").strip() if self._name_input else ""
        if not name:
            ui.notify("Please enter a name.", color="warning")
            return
        if self._pending_box is None:
            return
        self.objects.append({"name": name, "box": self._pending_box})
        self._pending_box = None
        if self._name_dialog:
            self._name_dialog.close()
        if self._objects_panel_refresh:
            self._objects_panel_refresh()
        self._update_svg()

    def _cancel_draw(self) -> None:
        self._pending_box = None
        if self._name_dialog:
            self._name_dialog.close()
        self._update_svg()

    def _delete_object(self, idx: int, refresh_fn) -> None:
        if 0 <= idx < len(self.objects):
            self.objects.pop(idx)
        refresh_fn()
        self._update_svg()

    def _go_to_verify(self, tabs: ui.tabs, next_tab: ui.tab) -> None:
        if not self.objects:
            ui.notify("Label at least one object first.", color="warning")
            return
        if self.reference_frame is None:
            ui.notify(
                "Load a reference frame first (click 'Get frame').", color="warning"
            )
            return
        self._save_session()
        tabs.set_value(next_tab)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    @property
    def _session_path(self) -> Path | None:
        if not self.project_name.strip() or not self.project_path.strip():
            return None
        return Path(self.project_path) / self.project_name / "session.json"

    def _save_session(self) -> None:
        path = self._session_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)

        pool_meta = [
            {
                "video": w["video"],
                "start_s": w["start_s"],
                "predicted": w.get("predicted", False),
                "frames": [
                    {"time_s": f["time_s"], "label": f["label"]} for f in w["frames"]
                ],
            }
            for w in self._training_pool
        ]

        data = {
            "version": 1,
            "project_name": self.project_name,
            "project_path": self.project_path,
            "video_paths": list(self.video_paths),
            "video_duration": self.video_duration,
            "objects": [
                {
                    "name": o["name"],
                    "box": list(o["box"]) if o.get("box") else None,
                }
                for o in self.objects
            ],
            "video_boxes": {
                vp: {name: list(box) for name, box in boxes.items()}
                for vp, boxes in self.video_boxes.items()
            },
            "exploration_prompts": list(self.exploration_prompts),
            "no_exploration_prompts": list(self.no_exploration_prompts),
            "confidence_threshold": self.confidence_threshold,
            "min_bout_seconds": self.min_bout_seconds,
            "bin_duration_seconds": self.bin_duration_seconds,
            "compute_di": self.compute_di,
            "head_trained": self._head_trained,
            "head_classes": list(self._head_classes),
            "training_pool": pool_meta,
        }

        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

        # Persist classifier head alongside the session
        if self._trained_clf is not None and self._head_trained:
            head_path = path.parent / "model" / "head.pkl"
            head_path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                self._trained_clf.save_head(head_path)

        logger.info("Session saved to '%s'.", path)

    def _load_session(self, path: Path) -> None:
        with open(path) as fh:
            data = json.load(fh)

        self.project_name = data["project_name"]
        self.project_path = data["project_path"]
        self.video_paths = data.get("video_paths", [])
        self.video_duration = data.get("video_duration", 5)

        self.objects = [
            {
                "name": o["name"],
                "box": tuple(o["box"]) if o.get("box") else None,
            }
            for o in data.get("objects", [])
        ]
        self.video_boxes = {
            vp: {name: tuple(box) for name, box in boxes.items()}
            for vp, boxes in data.get("video_boxes", {}).items()
        }

        self.exploration_prompts = data.get(
            "exploration_prompts", list(_DEFAULT_EXPLORATION)
        )
        self.no_exploration_prompts = data.get(
            "no_exploration_prompts", list(_DEFAULT_NO_EXPLORATION)
        )
        self.confidence_threshold = data.get("confidence_threshold", 0.5)
        self.min_bout_seconds = data.get("min_bout_seconds", 0.0)
        self.bin_duration_seconds = data.get("bin_duration_seconds", 60)
        self.compute_di = data.get("compute_di", True)
        self._head_trained = data.get("head_trained", False)
        self._head_classes = data.get("head_classes", [])

        # Restore training pool — frames are dehydrated (no BGR data yet)
        self._training_pool = [
            {
                "video": w["video"],
                "start_s": w["start_s"],
                "predicted": w.get("predicted", False),
                "frames": [
                    {
                        "time_s": f["time_s"],
                        "label": f["label"],
                        "frame": None,
                        "thumb": "",
                        "thumb_lg": "",
                    }
                    for f in w["frames"]
                ],
                "_needs_reload": True,
            }
            for w in data.get("training_pool", [])
        ]
        self._current_window = None

        # Try to reload classifier head
        head_path = path.parent / "model" / "head.pkl"
        if head_path.exists() and self._head_trained:
            try:
                from explore.classification.clip_classifier import CLIPClassifier

                clf = CLIPClassifier()
                clf.load_head(head_path)
                self._trained_clf = clf
                logger.info("Classifier head restored from '%s'.", head_path)
            except Exception as exc:
                logger.warning("Could not restore classifier head: %s", exc)

        # Refresh all UI panels that bind to state
        for fn in [
            self._video_list_refresh,
            self._label_video_select_refresh,
            self._objects_panel_refresh,
            self._current_window_refresh,
        ]:
            if fn:
                fn()
        if self._pool_label_update:
            self._pool_label_update()
        self._update_ref_select()
        if self._di_ri_refresh:
            self._di_ri_refresh()

        ui.notify(f"Session '{data['project_name']}' restored.", color="positive")

    def _ensure_pool_frames_loaded(self) -> None:
        """Re-read BGR frames for any dehydrated pool windows before training."""
        for win in self._training_pool:
            if not win.get("_needs_reload"):
                continue
            video = win["video"]
            cap = cv2.VideoCapture(video)
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
            boxes = self.video_boxes.get(video, {})
            for fd in win["frames"]:
                frame_idx = int(fd["time_s"] * fps_v)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if ok:
                    hl = fd["label"] if fd["label"] != _NOT_EXP else None
                    fd["frame"] = frame
                    fd["thumb"] = _make_thumbnail(frame, boxes, hl, None, width=120)
                    fd["thumb_lg"] = _make_thumbnail(frame, boxes, hl, None, width=400)
            cap.release()
            win["_needs_reload"] = False

    # ------------------------------------------------------------------
    # Event handlers — Verify tab
    # ------------------------------------------------------------------

    async def _run_localization(self) -> None:
        if not self.objects:
            ui.notify("No objects labeled.", color="warning")
            return
        if self.reference_frame is None:
            ui.notify("No reference frame set.", color="warning")
            return
        if not self.video_paths:
            ui.notify("No video files loaded.", color="warning")
            return

        container = self._verify_container
        if container is None:
            return
        container.clear()

        with container, ui.row().classes("items-center gap-3"):
            ui.spinner("dots", size="md")
            ui.label("Running ORB localization…").classes("text-sm text-gray-500")

        ref_frame = self.reference_frame
        ref_video = Path(self.video_paths[self.reference_video_idx])
        localizer = BoxLocalizer()

        from explore.detection.box_localizer import LocalizationResult

        for video_path_str in self.video_paths:
            video_path = Path(video_path_str)
            boxes: dict[str, tuple[int, int, int, int]] = {}
            results: dict[str, LocalizationResult] = {}

            if video_path == ref_video:
                for obj in self.objects:
                    boxes[obj["name"]] = obj["box"]
                    results[obj["name"]] = LocalizationResult(
                        box=obj["box"],
                        translation=(0.0, 0.0),
                        n_matches=-1,
                        success=True,
                    )
            else:
                for obj in self.objects:
                    result = await run.io_bound(
                        localizer.localize_from_video,
                        ref_frame,
                        obj["box"],
                        video_path,
                    )
                    boxes[obj["name"]] = result.box
                    results[obj["name"]] = result

            self.video_boxes[video_path_str] = boxes
            self._video_results[video_path_str] = results

        # Render interactive verification cards
        container.clear()
        n_failed = sum(
            1
            for res_dict in self._video_results.values()
            for res in res_dict.values()
            if not res.success and res.n_matches != -1
        )

        with container:
            for video_path_str in self.video_paths:
                video_path = Path(video_path_str)
                boxes = self.video_boxes.get(video_path_str, {})
                results = self._video_results.get(video_path_str, {})
                is_ref = video_path == ref_video

                frame = await run.io_bound(_random_frame, video_path)
                if frame is None:
                    continue

                self._build_one_verify_card(
                    video_path_str, boxes, results, frame, is_ref
                )

        if n_failed:
            ui.notify(
                f"{n_failed} object(s) could not be re-localized — original boxes kept. "
                "Drag any red-badged box to correct its position.",
                color="warning",
                timeout=6000,
            )
        else:
            ui.notify("Localization complete — all objects found.", color="positive")

    def _build_one_verify_card(
        self,
        video_path_str: str,
        initial_boxes: dict[str, tuple[int, int, int, int]],
        results: dict,
        frame: np.ndarray,
        is_ref: bool,
    ) -> None:
        """Interactive card with a sample frame and draggable box overlays."""
        video_path = Path(video_path_str)
        fh, fw = frame.shape[:2]
        current_boxes: dict[str, tuple[int, int, int, int]] = dict(initial_boxes)
        drag: dict = {"name": None, "mouse_start": None, "box_start": None}
        widget_holder: dict = {}

        def _svg() -> str:
            parts: list[str] = [
                '<defs><filter id="vsh"><feDropShadow dx="0" dy="0" '
                'stdDeviation="2" flood-color="black" flood-opacity="0.9"/>'
                "</filter></defs>"
            ]
            for i, obj in enumerate(self.objects):
                name = obj["name"]
                box = current_boxes.get(name)
                if box is None:
                    continue
                c = _BOX_COLORS[i % len(_BOX_COLORS)]
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                stroke_w = 3 if drag["name"] == name else 2
                parts.append(
                    f'<rect x="{x1}" y="{y1}" width="{bw}" height="{bh}" '
                    f'fill="{c}33" stroke="{c}" stroke-width="{stroke_w}" rx="2" '
                    f'style="cursor:grab"/>'
                )
                label_y = max(y1 - 5, 16)
                parts.append(
                    f'<text x="{x1 + 4}" y="{label_y}" fill="{c}" '
                    f'font-size="14" font-family="sans-serif" font-weight="bold" '
                    f'filter="url(#vsh)">{name}</text>'
                )
            return "".join(parts)

        def on_mouse(e) -> None:
            if e.type == "mousedown":
                for obj in self.objects:
                    name = obj["name"]
                    box = current_boxes.get(name)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    if x1 <= e.image_x <= x2 and y1 <= e.image_y <= y2:
                        drag["name"] = name
                        drag["mouse_start"] = (e.image_x, e.image_y)
                        drag["box_start"] = box
                        break

            elif e.type == "mousemove":
                if drag["name"] is not None:
                    mx, my = drag["mouse_start"]
                    dx = e.image_x - mx
                    dy = e.image_y - my
                    bx1, by1, bx2, by2 = drag["box_start"]
                    bw_box, bh_box = bx2 - bx1, by2 - by1
                    new_x1 = max(0, min(fw - bw_box, int(bx1 + dx)))
                    new_y1 = max(0, min(fh - bh_box, int(by1 + dy)))
                    current_boxes[drag["name"]] = (
                        new_x1,
                        new_y1,
                        new_x1 + bw_box,
                        new_y1 + bh_box,
                    )
                    img = widget_holder.get("img")
                    if img:
                        img.set_content(_svg())

            elif e.type == "mouseup":
                drag["name"] = None
                drag["mouse_start"] = None
                drag["box_start"] = None
                self.video_boxes[video_path_str] = dict(current_boxes)
                img = widget_holder.get("img")
                if img:
                    img.set_content(_svg())

        def _reset() -> None:
            current_boxes.update(initial_boxes)
            self.video_boxes[video_path_str] = dict(initial_boxes)
            img = widget_holder.get("img")
            if img:
                img.set_content(_svg())

        with ui.card().classes("w-full max-w-2xl p-3 gap-2"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("star" if is_ref else "videocam").classes(
                    "text-yellow-500" if is_ref else "text-blue-400"
                )
                ui.label(video_path.name).classes("font-medium text-sm flex-grow")
                if is_ref:
                    ui.badge("reference").props("color=blue")
                ui.label("drag boxes to reposition").classes(
                    "text-xs text-gray-400 italic"
                )
                ui.button(icon="restart_alt", on_click=_reset).props(
                    "flat round dense color=grey"
                ).tooltip("Reset boxes to ORB positions")

            widget_holder["img"] = ui.interactive_image(
                source=_bgr_to_data_url(frame),
                content=_svg(),
                on_mouse=on_mouse,
                events=["mousedown", "mousemove", "mouseup"],
            ).classes("w-full rounded")

            with ui.column().classes("gap-1 mt-1"):
                for i, obj in enumerate(self.objects):
                    name = obj["name"]
                    res = results.get(name)
                    c = _BOX_COLORS[i % len(_BOX_COLORS)]
                    with ui.row().classes("items-center gap-2"):
                        ui.element("div").style(
                            f"width:10px;height:10px;border-radius:2px;"
                            f"background:{c};flex-shrink:0"
                        )
                        ui.label(name).classes("text-xs font-medium w-28")
                        if res is None:
                            ui.badge("no result").props("color=grey")
                        elif is_ref or res.n_matches == -1:
                            ui.badge("drawn directly").props("color=blue")
                        elif not res.success:
                            ui.badge(
                                f"ORB failed ({res.n_matches} matches) — drag to fix"
                            ).props("color=red")
                        elif res.n_matches >= 10:
                            ui.badge(f"ORB  {res.n_matches} matches").props(
                                "color=green"
                            )
                        else:
                            ui.badge(f"ORB borderline  {res.n_matches} matches").props(
                                "color=orange"
                            )

    # ------------------------------------------------------------------
    # Event handlers — Label Frames tab
    # ------------------------------------------------------------------

    async def _do_sample_window(self) -> None:
        """Show a new window in the strip without touching the training pool.

        Uses model predictions if a classifier has been trained, otherwise
        falls back to proximity labels.
        """
        if not self.objects:
            ui.notify("No objects labeled.", color="warning")
            return
        if not self.video_paths:
            ui.notify("No videos loaded.", color="warning")
            return
        if not self.video_boxes:
            ui.notify("Run localization in Step 3 first.", color="warning")
            return

        video = self._label_window_video or self.video_paths[0]
        start_s = (
            float(self._manual_start_s)
            if self._manual_start_s is not None
            else self._random_preview_start(video)
        )
        boxes = self.video_boxes.get(video, {})
        use_model = self._trained_clf is not None and self._head_trained

        if self._sample_status:
            self._sample_status.text = "Predicting…" if use_model else "Sampling…"

        if use_model:
            frames = await run.io_bound(
                _predict_label_window,
                self._trained_clf,
                self._head_classes,
                video,
                boxes,
                start_s,
            )
            predicted = True
        else:
            frames = await run.io_bound(_sample_label_window, video, boxes, start_s)
            predicted = False

        if not frames:
            if self._sample_status:
                self._sample_status.text = "No frames found — check boxes in Step 3."
            ui.notify("No frames sampled.", color="warning")
            return

        self._current_window = {
            "video": video,
            "start_s": start_s,
            "frames": frames,
            "predicted": predicted,
        }
        source = "model predictions" if predicted else "proximity labels"
        if self._sample_status:
            self._sample_status.text = f"{len(frames)} frames ({source}) — correct if needed, then Train + Preview."
        if self._current_window_refresh:
            self._current_window_refresh()

    async def _train_and_preview(self) -> None:
        from explore.classification.clip_classifier import CLIPClassifier

        if self._current_window is None:
            ui.notify("Sample a window first.", color="warning")
            return

        # Save current window (with any user corrections) to the training pool
        self._training_pool.append(self._current_window)
        if hasattr(self, "_pool_label_update") and self._pool_label_update:
            self._pool_label_update()

        all_samples = [
            {"frame": f["frame"], "label": f["label"]}
            for w in self._training_pool
            for f in w["frames"]
        ]

        all_classes = [o["name"] for o in self.objects] + [_NOT_EXP]
        present = {s["label"] for s in all_samples}
        missing = [c for c in all_classes if c not in present]
        if missing:
            ui.notify(
                f"Missing labels for: {', '.join(missing)}. Correct some labels first.",
                color="warning",
            )
            # Remove the window we just added so the user can fix it and retry
            self._training_pool.pop()
            if hasattr(self, "_pool_label_update") and self._pool_label_update:
                self._pool_label_update()
            return

        if self._train_status:
            self._train_status.text = "Embedding frames and training…"

        # Re-read frames for any pool windows restored from a session file
        await run.io_bound(self._ensure_pool_frames_loaded)
        all_samples = [
            {"frame": f["frame"], "label": f["label"]}
            for w in self._training_pool
            for f in w["frames"]
            if f.get("frame") is not None
        ]

        raw_frames = [s["frame"] for s in all_samples]
        label_to_int = {lbl: i for i, lbl in enumerate(all_classes)}
        y = np.array([label_to_int[s["label"]] for s in all_samples])

        def _do_train() -> CLIPClassifier:
            clf = CLIPClassifier() if self._trained_clf is None else self._trained_clf
            embeddings = clf.embed_frames(raw_frames, show_progress=False)
            clf.fit(embeddings, y)
            return clf

        try:
            clf = await run.io_bound(_do_train)
            self._trained_clf = clf
            self._head_classes = all_classes
            self._head_trained = True

            if self._pipeline is not None:
                self._pipeline._classifier = clf
                self._pipeline.set_head_class_names(all_classes)

            if self._train_status:
                self._train_status.text = f"✓ {len(all_samples)} frames · {len(all_classes)} classes · predicting next window…"

            # Pick a new random window (from any video) and predict with the model
            video = self._label_window_video or self.video_paths[0]
            start_s = self._random_preview_start(
                video,
                exclude_start=self._current_window["start_s"]
                if self._current_window
                else -999.0,
            )
            boxes = self.video_boxes.get(video, {})
            new_frames = await run.io_bound(
                _predict_label_window, clf, all_classes, video, boxes, start_s
            )

            if new_frames:
                self._current_window = {
                    "video": video,
                    "start_s": start_s,
                    "frames": new_frames,
                    "predicted": True,
                }
                if self._current_window_refresh:
                    self._current_window_refresh()

            if self._train_status:
                self._train_status.text = f"✓ Trained on {len(all_samples)} frames · {len(all_classes)} classes"
            self._save_session()
            ui.notify(
                "Trained — correct any wrong labels, then train again.",
                color="positive",
            )

        except Exception as exc:
            logger.exception("Training failed")
            if self._train_status:
                self._train_status.text = f"Error: {exc}"
            ui.notify(f"Training error: {exc}", color="negative")

    def _random_preview_start(
        self, video_path_str: str, exclude_start: float = -999.0
    ) -> float:
        """Return a random valid 60-second preview start time for the video."""
        cap = cv2.VideoCapture(video_path_str)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration_s = total / fps
        lo = 30.0
        hi = max(lo + 1.0, duration_s - 60.0)
        if hi <= lo:
            return lo
        for _ in range(10):
            s = random.uniform(lo, hi)
            if abs(s - exclude_start) > 30.0:
                return s
        return random.uniform(lo, hi)

    # ------------------------------------------------------------------
    # Event handlers — Analyze tab
    # ------------------------------------------------------------------

    async def _run_analysis(self) -> None:
        err = self._validate_for_analysis()
        if err:
            ui.notify(err, color="negative")
            return

        cfg = self._build_config()
        if self._pipeline is None:
            self._pipeline = ExplorationPipeline(cfg, headless=True)
        else:
            self._pipeline.config = cfg

        if self.reference_frame is not None and self.video_paths:
            self._pipeline.set_reference_frame(
                self.reference_frame,
                Path(self.video_paths[self.reference_video_idx]),
            )

        # Pass GUI-verified boxes → skips redundant ORB during run()
        if self.video_boxes:
            self._pipeline.set_per_video_boxes(self.video_boxes)

        # Wire trained classifier + class names into the pipeline
        if self._head_trained and self._trained_clf is not None:
            self._pipeline._classifier = self._trained_clf
            self._pipeline.set_head_class_names(self._head_classes)

        self._pipeline.pred_video_hires = self.pred_video_hires

        if self._run_btn:
            self._run_btn.disable()
        if self._progress:
            self._progress.set_visibility(True)
            self._progress.set_value(0)
        if self._log:
            self._log.clear()

        # Route pipeline log output to the NiceGUI log widget
        handler = _NiceGuiLogHandler(self._log)
        logging.getLogger("explore").addHandler(handler)

        try:
            results = await run.io_bound(self._pipeline.run)
            self._show_results(results)
            if self._tabs:
                self._tabs.set_value("6. Results")
            ui.notify("Analysis complete!", color="positive")
        except Exception as exc:
            logger.exception("Analysis failed")
            ui.notify(f"Error: {exc}", color="negative", timeout=0)
        finally:
            logging.getLogger("explore").removeHandler(handler)
            if self._run_btn:
                self._run_btn.enable()
            if self._progress:
                self._progress.set_visibility(False)

    def _validate_for_analysis(self) -> str:
        if not self.project_name.strip():
            return "Project name is required."
        if not self.video_paths:
            return "No video files loaded."
        if not self.objects:
            return "No objects labeled."
        missing = [o["name"] for o in self.objects if o.get("box") is None]
        if missing:
            return f"Objects without boxes: {', '.join(missing)}"
        return ""

    def _build_config(self) -> ExperimentConfig:
        objects = [
            ObjectConfig(name=o["name"], bounding_box=o["box"]) for o in self.objects
        ]
        behavior = BehaviorConfig(
            exploration_prompts=self.exploration_prompts,
            no_exploration_prompts=self.no_exploration_prompts,
            confidence_threshold=float(self.confidence_threshold),
            min_bout_seconds=float(self.min_bout_seconds),
        )
        analysis = AnalysisConfig(
            bin_duration_minutes=self.bin_duration_seconds / 60.0,
            compute_di=self.compute_di,
        )
        return ExperimentConfig(
            project_name=self.project_name.strip(),
            project_path=Path(self.project_path),
            video_paths=[Path(p) for p in self.video_paths],
            video_duration_minutes=int(self.video_duration),
            objects=objects,
            behavior=behavior,
            model=ModelConfig(),
            analysis=analysis,
        )

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _show_results(self, df) -> None:  # type: ignore[no-untyped-def]
        import pandas as pd

        container = self._results_container
        if container is None:
            return
        container.clear()

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            with container:
                ui.label("No results produced.").classes("text-gray-400 italic")
            return

        cfg = self._build_config()
        csv_path = cfg.project_dir / "results" / f"{cfg.project_name}.csv"

        with container:
            with ui.row().classes("items-center gap-3 mb-2"):
                ui.label("Results saved to:").classes("text-sm text-gray-500")
                ui.label(str(csv_path)).classes("text-sm font-mono text-blue-700")

            # Render as a NiceGUI table
            cols = [{"name": c, "label": c, "field": c} for c in df.columns]
            rows = df.round(3).to_dict("records")
            ui.table(columns=cols, rows=rows, row_key=df.columns[0]).classes(
                "w-full text-xs"
            ).props("dense flat bordered")


# ---------------------------------------------------------------------------
# Logging bridge
# ---------------------------------------------------------------------------


class _NiceGuiLogHandler(logging.Handler):
    """Forward Python log records to a ``ui.log`` widget."""

    def __init__(self, log_widget: ui.log | None) -> None:
        super().__init__()
        self._widget = log_widget

    def emit(self, record: logging.LogRecord) -> None:
        if self._widget is None:
            return
        with contextlib.suppress(Exception):
            self._widget.push(self.format(record))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    """Start the EXPLORE 2.0 NiceGUI browser application."""
    state = ExploreApp()

    @ui.page("/")
    def index() -> None:
        state.build()

    ui.run(
        title="EXPLORE",
        port=8080,
        reload=False,
        favicon=str(_LOGO_PATH) if _LOGO_PATH.exists() else "🔬",
        show=True,
    )
