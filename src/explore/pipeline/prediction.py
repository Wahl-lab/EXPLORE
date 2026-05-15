"""End-to-end exploration analysis pipeline.

Orchestrates ORB-based box localization, CLIP classification, active
learning, and behavioral statistics into a single ``ExplorationPipeline``
object.

Typical usage
-------------
>>> from explore import ExperimentConfig, ExplorationPipeline
>>> cfg = ExperimentConfig.from_yaml("experiment.yaml")
>>> pipeline = ExplorationPipeline(cfg)
>>> pipeline.set_reference_frame(frame)   # frame on which boxes were drawn
>>> results = pipeline.run()              # returns tidy pd.DataFrame
>>> results.to_csv("results.csv", index=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from explore.classification.active_learning import ActiveLearner
from explore.classification.clip_classifier import CLIPClassifier
from explore.config import ExperimentConfig, ObjectConfig
from explore.detection.box_localizer import BoxLocalizer
from explore.pipeline.analysis import BehaviorAnalyzer
from explore.utils.video import VideoReader

logger = logging.getLogger(__name__)

_SKIP_SECONDS = 5
_ANALYSIS_FPS = 4.0
_EMBED_BATCH = 32
_PRED_LO_FPS = 12.0  # low-res: output FPS  (3 repeats of each 4-fps analysis frame)
_PRED_LO_SCALE = 0.5  # low-res: spatial scale (half resolution)
_PRED_HI_FPS = 25.0  # high-res: output FPS  (overridden by actual video FPS at runtime)
_PRED_HI_SCALE = 1.0  # high-res: full resolution


def _filter_bouts(arr: np.ndarray, min_frames: int) -> np.ndarray:
    """Zero out exploration runs shorter than *min_frames*."""
    if min_frames <= 1:
        return arr.copy()  # type: ignore[no-any-return]
    out = arr.copy()
    in_bout = False
    start = 0
    for i in range(len(arr) + 1):
        val = arr[i] if i < len(arr) else 0
        if val == 1 and not in_bout:
            in_bout = True
            start = i
        elif val == 0 and in_bout:
            in_bout = False
            if (i - start) < min_frames:
                out[start:i] = 0
    return out  # type: ignore[no-any-return]


class ExplorationPipeline:
    """Full EXPLORE 2.0 analysis pipeline.

    Parameters
    ----------
    config:
        Experiment configuration.
    headless:
        When ``True``, skip any interactive steps.
        All bounding boxes must already be set in ``config.objects``.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        headless: bool = False,
    ) -> None:
        self.config = config
        self.headless = headless

        self._classifier = CLIPClassifier(
            model_name=config.model.clip_model,
            pretrained=config.model.clip_pretrained,
        )
        self._learner = ActiveLearner(self._classifier)
        self.pred_video_hires: bool = False
        self._reference_frame: np.ndarray | None = None
        self._reference_video: Path | None = None
        # Pre-verified boxes from GUI (str path → {obj_name → box}).
        # When populated, _localize_boxes uses these directly and skips ORB.
        self._per_video_boxes: dict[str, dict[str, tuple[int, int, int, int]]] = {}
        self._head_class_names: list[str] | None = None

        # Auto-load persisted reference frame if present
        ref_jpg = config.project_dir / "reference_frame.jpg"
        if ref_jpg.exists():
            img = cv2.imread(str(ref_jpg))
            if img is not None:
                self._reference_frame = img
                logger.info("Loaded reference frame from '%s'.", ref_jpg)

    # ------------------------------------------------------------------
    # Reference frame and pre-verified boxes
    # ------------------------------------------------------------------

    def set_reference_frame(
        self,
        frame: np.ndarray,
        video_path: Path | None = None,
    ) -> None:
        """Set the frame on which bounding boxes were drawn.

        The pipeline uses this frame as the reference for ORB localization
        when processing videos other than *video_path*.  The frame is also
        persisted to ``<project_dir>/reference_frame.jpg`` so that the
        pipeline can resume across sessions.

        Parameters
        ----------
        frame:
            BGR image (as returned by cv2).
        video_path:
            The video this frame came from.  Boxes are used as-is for that
            video; ORB localization only runs for the remaining videos.
        """
        self._reference_frame = frame
        self._reference_video = video_path

        out_dir = self.config.project_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "reference_frame.jpg"), frame)

    def set_per_video_boxes(
        self,
        video_boxes: dict[str, dict[str, tuple[int, int, int, int]]],
    ) -> None:
        """Supply pre-verified bounding boxes from the GUI verification step.

        When these are set, ``_localize_boxes`` uses them directly and skips
        ORB re-localization entirely — no redundant computation at run time.

        Parameters
        ----------
        video_boxes:
            Mapping ``{video_path_str: {object_name: (x1, y1, x2, y2)}}``,
            exactly as stored in ``ExploreApp.video_boxes`` after Tab 3.
        """
        self._per_video_boxes = {str(k): v for k, v in video_boxes.items()}
        logger.info(
            "Pre-verified boxes loaded for %d video(s) — ORB will be skipped.",
            len(self._per_video_boxes),
        )

    def set_head_class_names(self, class_names: list[str]) -> None:
        """Map trained head output indices to object/class names.

        Parameters
        ----------
        class_names:
            Ordered list matching the integer labels used in ``fit()``.
            Typically ``[obj1_name, obj2_name, ..., "not_exploring"]``.
        """
        self._head_class_names = class_names
        logger.info("Head class names: %s", class_names)

    # ------------------------------------------------------------------
    # Zero-shot initialisation
    # ------------------------------------------------------------------

    def initialize_classifier(
        self,
        sample_frames: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Embed a sample of frames and auto-label confident ones."""
        if sample_frames is None:
            sample_frames = self._sample_frames_for_init()

        logger.info("Embedding %d initialisation frames …", len(sample_frames))
        embeddings = self._classifier.embed_frames(sample_frames)

        probas = self._classifier.zero_shot_predict(
            embeddings,
            self.config.behavior.exploration_prompts,
            self.config.behavior.no_exploration_prompts,
        )
        self._learner.auto_label_confident(embeddings, probas)
        return embeddings

    # ------------------------------------------------------------------
    # Active learning round
    # ------------------------------------------------------------------

    def get_uncertain_frames(self, probas: np.ndarray) -> np.ndarray:
        """Return frame indices to show the user for correction."""
        return self._learner.query(
            probas, already_labeled=set(self._learner.labeled_indices)
        )

    def update_with_corrections(
        self,
        embeddings: np.ndarray,
        corrections: dict[int, int],
    ) -> None:
        """Incorporate user-supplied frame corrections and refit the head."""
        self._learner.update(embeddings, corrections)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Run the full pipeline on all videos in ``config.video_paths``."""
        self._prepare_output_dirs()
        cfg = self.config
        all_results: list[pd.DataFrame] = []

        for video_path in tqdm(cfg.video_paths, desc="Processing videos"):
            animal_id = video_path.stem
            logger.info("Processing '%s' …", animal_id)

            # Localize bounding boxes for this video
            boxes = self._localize_boxes(video_path)

            # Pass 1: embed frames in streaming batches
            embeddings, effective_fps = self._embed_streaming(video_path)
            if embeddings.shape[0] == 0:
                logger.warning("No frames extracted from '%s', skipping.", video_path)
                continue

            background = self._estimate_background(video_path)
            precomputed: dict[str, Any] | None = None

            if (
                self._head_class_names is not None
                and self._classifier._head is not None
            ):
                # Multi-class trained head: prediction directly encodes object
                class_indices = self._classifier.predict_class_indices(embeddings)
                precomputed = {
                    o.name: np.zeros(len(embeddings), dtype=np.int32)
                    for o in cfg.objects
                }
                for i, ci in enumerate(class_indices):
                    cls = self._head_class_names[ci]
                    if cls in precomputed:
                        precomputed[cls][i] = 1
                exploration_mask = np.zeros(len(embeddings), dtype=bool)
                for arr in precomputed.values():
                    exploration_mask |= arr.astype(bool)
                logger.info(
                    "Multi-class head: %d exploration frames out of %d.",
                    int(exploration_mask.sum()),
                    len(exploration_mask),
                )
            elif self._classifier._head is not None:
                # Head loaded without class names — can't do multi-class assignment;
                # fall back to zero-shot so output is at least meaningful.
                logger.warning(
                    "Classifier head present but head_class_names not set — "
                    "falling back to zero-shot prediction. "
                    "Call set_head_class_names() after loading the head."
                )
                probas = self._classifier.zero_shot_predict(
                    embeddings,
                    cfg.behavior.exploration_prompts,
                    cfg.behavior.no_exploration_prompts,
                )
                exploration_mask = probas >= cfg.behavior.confidence_threshold
            else:
                probas = self._classifier.zero_shot_predict(
                    embeddings,
                    cfg.behavior.exploration_prompts,
                    cfg.behavior.no_exploration_prompts,
                )
                exploration_mask = probas >= cfg.behavior.confidence_threshold

            # Pass 2: assign objects per frame (raw, unfiltered)
            raw_labels = self._assign_labels(
                video_path,
                exploration_mask,
                cfg.objects,
                boxes,
                background,
                precomputed_labels=precomputed,
            )

            # Apply bout filter so video and CSV are consistent
            min_bout_frames = int(cfg.behavior.min_bout_seconds * effective_fps)
            object_labels = {
                name: _filter_bouts(arr, min_bout_frames)
                for name, arr in raw_labels.items()
            }

            # Pass 3: write annotated video with filtered labels
            self._write_prediction_video(video_path, object_labels, cfg.objects, boxes)

            # Pass 4: track animal position and save CSV + trajectory plot
            tracking_df = self._track_animal(video_path, background, effective_fps)
            tracking_csv = (
                cfg.project_dir
                / "results"
                / "tracking"
                / f"{video_path.stem}_tracking.csv"
            )
            tracking_df.to_csv(tracking_csv, index=False)
            logger.info("Tracking CSV saved to '%s'.", tracking_csv)
            self._save_trajectory_plot(tracking_df, boxes, video_path)

            # Labels are already filtered — pass min_bout_seconds=0 to avoid double-filtering
            analyzer = BehaviorAnalyzer(
                fps=effective_fps,
                bin_duration_seconds=cfg.analysis.bin_duration_minutes * 60.0,
                min_bout_seconds=0,
            )
            df = analyzer.compute(object_labels, animal_id, cfg.project_name)

            if cfg.analysis.compute_di and len(cfg.objects) >= 2:
                # DI/RI for every pair of objects — from TOTAL session exploration.
                # Per-bin DI is erratic (empty bin → extreme value), so sum first.
                from itertools import combinations

                df = df.copy()
                obj_names = [o.name for o in cfg.objects]
                for obj_a, obj_b in combinations(obj_names, 2):
                    col_a = f"{obj_a}_time_s"
                    col_b = f"{obj_b}_time_s"
                    if col_a not in df.columns or col_b not in df.columns:
                        continue
                    t_a = float(df[col_a].sum())
                    t_b = float(df[col_b].sum())
                    total = t_a + t_b
                    pair = f"{obj_a}_vs_{obj_b}"
                    df[f"DI_{pair}"] = (t_a - t_b) / total if total > 0 else 0.0
                    df[f"RI_{pair}"] = t_a / total if total > 0 else 0.5

            all_results.append(df)

        if not all_results:
            logger.warning("No results produced — check video paths and configuration.")
            return pd.DataFrame()

        combined = BehaviorAnalyzer.aggregate(all_results)
        output_csv = cfg.project_dir / "results" / f"{cfg.project_name}.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_csv, index=False)
        logger.info("Results saved to '%s'.", output_csv)

        cfg.save()
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _localize_boxes(
        self,
        video_path: Path,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Return per-object pixel boxes for *video_path*.

        Priority:
        1. Pre-verified boxes set via :meth:`set_per_video_boxes` (GUI Tab 3).
        2. Reference video → use drawn boxes directly (no shift needed).
        3. Other videos → run ORB localization.
        """
        key = str(video_path)

        # 1. GUI-verified boxes take priority — no ORB needed
        if key in self._per_video_boxes:
            logger.info("Using pre-verified boxes for '%s'.", video_path.name)
            return self._per_video_boxes[key]

        objects = self.config.objects
        original: dict[str, tuple[int, int, int, int]] = {
            o.name: o.bounding_box for o in objects if o.bounding_box is not None
        }

        # 2. Reference video or no reference frame → use drawn boxes as-is
        if self._reference_frame is None or video_path == self._reference_video:
            return original

        # 3. ORB localization for remaining videos
        localizer = BoxLocalizer()
        result: dict[str, tuple[int, int, int, int]] = {}
        for name, box in original.items():
            loc = localizer.localize_from_video(self._reference_frame, box, video_path)
            result[name] = loc.box
            if not loc.success:
                logger.warning(
                    "ORB localization failed for '%s' in '%s'; using original box.",
                    name,
                    video_path.name,
                )
        return result

    def _prepare_output_dirs(self) -> None:
        for d in [
            self.config.project_dir / "results",
            self.config.project_dir / "results" / "prediction_videos",
            self.config.project_dir / "results" / "tracking",
            self.config.project_dir / "model",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _frame_range(self, video_path: Path) -> tuple[int, int, int, float]:
        """Return (skip, max_frame, step, effective_fps) for a video."""
        fps = self._get_fps(video_path)
        skip = int(_SKIP_SECONDS * fps)
        max_frame = skip + int(self.config.video_duration_minutes * 60 * fps)
        step = max(1, round(fps / _ANALYSIS_FPS))
        effective_fps = fps / step
        return skip, max_frame, step, effective_fps

    def _embed_streaming(self, video_path: Path) -> tuple[np.ndarray, float]:
        """Stream frames in small batches, embed with CLIP, return embeddings."""
        skip, max_frame, step, effective_fps = self._frame_range(video_path)
        reader = VideoReader(video_path)

        all_embeddings: list[np.ndarray] = []
        batch: list[np.ndarray] = []

        for _, frame in reader.iter_frames(start=skip, end=max_frame, step=step):
            batch.append(frame)
            if len(batch) >= _EMBED_BATCH:
                all_embeddings.append(
                    self._classifier.embed_frames(batch, show_progress=False)
                )
                batch = []

        if batch:
            all_embeddings.append(
                self._classifier.embed_frames(batch, show_progress=False)
            )

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32), effective_fps

        return np.concatenate(all_embeddings, axis=0), effective_fps

    def _estimate_background(
        self, video_path: Path, n_frames: int = 30
    ) -> np.ndarray | None:
        """Compute a median background from uniformly sampled frames.

        Since objects and arena floor are static, the median converges to the
        background quickly — the mouse, which moves across frames, averages out.
        This is used for foreground (mouse) detection in ``_nearest_object``.
        """
        skip, max_frame, step, _ = self._frame_range(video_path)
        reader = VideoReader(video_path)
        total_range = max_frame - skip
        sample_step = max(step, total_range // n_frames)

        buffer: list[np.ndarray] = []
        for _, frame in reader.iter_frames(start=skip, end=max_frame, step=sample_step):
            buffer.append(frame)
            if len(buffer) >= n_frames:
                break

        if not buffer:
            return None

        stack = np.stack(buffer, axis=0).astype(np.float32)
        bg = np.median(stack, axis=0).astype(np.uint8)
        logger.debug("Background estimated from %d frames.", len(buffer))
        return bg

    def _assign_labels(
        self,
        video_path: Path,
        exploration_mask: np.ndarray,
        objects: list[ObjectConfig],
        localized_boxes: dict[str, tuple[int, int, int, int]],
        background: np.ndarray | None = None,
        precomputed_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """First pass: assign exploration labels per frame per object.

        When *precomputed_labels* is supplied (multi-class trained head), the
        labels are already assigned and no frame iteration is needed.  Otherwise
        frames are read to run :meth:`_nearest_object` proximity assignment.
        """
        n = len(exploration_mask)

        if precomputed_labels is not None:
            return {k: v.copy() for k, v in precomputed_labels.items()}

        labels = {o.name: np.zeros(n, dtype=np.int32) for o in objects}
        no_bboxes = not localized_boxes

        skip, max_frame, step, _ = self._frame_range(video_path)
        reader = VideoReader(video_path)

        for frame_idx, (_, frame) in enumerate(
            reader.iter_frames(start=skip, end=max_frame, step=step)
        ):
            if frame_idx >= n:
                break
            if bool(exploration_mask[frame_idx]):
                if no_bboxes:
                    labels[objects[0].name][frame_idx] = 1
                else:
                    nearest = self._nearest_object(
                        frame, objects, localized_boxes, background
                    )
                    if nearest:
                        labels[nearest][frame_idx] = 1

        return labels

    def _write_prediction_video(
        self,
        video_path: Path,
        labels: dict[str, np.ndarray],
        objects: list[ObjectConfig],
        localized_boxes: dict[str, tuple[int, int, int, int]],
    ) -> None:
        """Second pass: write the annotated prediction video from filtered labels."""
        skip, max_frame, step, _ = self._frame_range(video_path)
        if self.pred_video_hires:
            out_fps = self._get_fps(video_path)
            scale = _PRED_HI_SCALE
        else:
            out_fps = _PRED_LO_FPS
            scale = _PRED_LO_SCALE
        reader = VideoReader(video_path)

        n = max((len(arr) for arr in labels.values()), default=0)

        out_path = (
            self.config.project_dir
            / "results"
            / "prediction_videos"
            / f"{video_path.stem}_predicted.mp4"
        )
        writer: cv2.VideoWriter | None = None

        colors = [
            (0, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (80, 80, 240),
        ]
        obj_names = [o.name for o in objects]

        frame_idx = 0
        for _, frame in reader.iter_frames(start=skip, end=max_frame, step=step):
            if frame_idx >= n:
                break

            if writer is None:
                h, w = frame.shape[:2]
                out_w = max(1, int(w * scale))
                out_h = max(1, int(h * scale))
                repeat = max(1, round(out_fps / _ANALYSIS_FPS))
                # avc1 (H.264) avoids the green-frame bug on macOS with mp4v
                fourcc = cv2.VideoWriter_fourcc(*"avc1")  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
                if not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                    writer = cv2.VideoWriter(
                        str(out_path), fourcc, out_fps, (out_w, out_h)
                    )

            out_frame = frame.copy()
            for j, name in enumerate(obj_names):
                box = localized_boxes.get(name)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                color = colors[j % len(colors)]

                if labels.get(name, np.zeros(1))[frame_idx]:
                    # Active exploration: colored fill + solid border + label
                    overlay = out_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.3, out_frame, 0.7, 0, out_frame)
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        out_frame,
                        name,
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                else:
                    # Not exploring: dim grey outline only
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

            out_frame = cv2.resize(
                out_frame, (out_w, out_h), interpolation=cv2.INTER_AREA
            )
            for _ in range(repeat):
                writer.write(out_frame)

            frame_idx += 1

        if writer is not None:
            writer.release()
            logger.info("Prediction video saved to '%s'.", out_path)

        return

    def _nearest_object(
        self,
        frame: np.ndarray,
        objects: list[ObjectConfig],
        localized_boxes: dict[str, tuple[int, int, int, int]],
        background: np.ndarray | None = None,
    ) -> str | None:
        """Find which object the mouse is closest to.

        When a median *background* is available, foreground detection via
        background subtraction isolates the mouse accurately.  Without it,
        the method falls back to a raw brightness threshold, which is
        unreliable on typical arena videos.
        """
        if background is not None:
            diff = cv2.absdiff(frame, background)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            # Morphological cleanup: fill gaps, remove noise
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Discard tiny blobs (noise); keep only meaningful foreground regions
        contours = [c for c in contours if cv2.contourArea(c) > 200]

        if not contours:
            return objects[0].name if objects else None

        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return objects[0].name if objects else None

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        best_name: str | None = None
        best_dist = float("inf")
        for obj in objects:
            box = localized_boxes.get(obj.name)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            ox, oy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_name = obj.name

        return best_name

    def _track_animal(
        self,
        video_path: Path,
        background: np.ndarray | None,
        effective_fps: float,
    ) -> pd.DataFrame:
        """Run background-subtraction centroid tracking over all analysis frames.

        Returns a DataFrame with columns: frame, time_s, x, y.
        Frames where the animal is not detected have NaN for x and y.
        """
        skip, max_frame, step, _ = self._frame_range(video_path)
        reader = VideoReader(video_path)

        records: list[dict[str, Any]] = []

        for frame_idx, (_, frame) in enumerate(
            reader.iter_frames(start=skip, end=max_frame, step=step)
        ):
            time_s = frame_idx / effective_fps

            cx: float | None = None
            cy_val: float | None = None

            if background is not None:
                diff = cv2.absdiff(frame, background)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = [c for c in contours if cv2.contourArea(c) > 200]
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    moments = cv2.moments(largest)
                    if moments["m00"] > 0:
                        cx = moments["m10"] / moments["m00"]
                        cy_val = moments["m01"] / moments["m00"]

            records.append({"frame": frame_idx, "time_s": time_s, "x": cx, "y": cy_val})
            frame_idx += 1

        return pd.DataFrame(records)

    def _save_trajectory_plot(
        self,
        tracking_df: pd.DataFrame,
        localized_boxes: dict[str, tuple[int, int, int, int]],
        video_path: Path,
    ) -> None:
        """Save a trajectory plot colour-coded by time with object boxes overlaid."""
        out_path = (
            self.config.project_dir
            / "results"
            / "tracking"
            / f"{video_path.stem}_trajectory.png"
        )

        if (
            tracking_df.empty
            or "x" not in tracking_df.columns
            or "y" not in tracking_df.columns
        ):
            logger.warning(
                "No animal detections for '%s'; skipping trajectory plot.",
                video_path.name,
            )
            return
        detected = tracking_df.dropna(subset=["x", "y"])
        if detected.empty:
            logger.warning(
                "No animal detections for '%s'; skipping trajectory plot.",
                video_path.name,
            )
            return

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle

        fig = Figure(figsize=(7, 7))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)

        xs = detected["x"].to_numpy(dtype=float)
        ys = detected["y"].to_numpy(dtype=float)
        ts = detected["time_s"].to_numpy(dtype=float)

        # Faint grey connecting line
        ax.plot(xs, ys, color="lightgrey", linewidth=0.5, zorder=1)

        # Scatter points colour-coded by time
        sc = ax.scatter(xs, ys, c=ts, cmap="plasma", s=6, linewidths=0, zorder=2)
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.7)

        # Mark start and end
        ax.scatter([xs[0]], [ys[0]], color="green", s=60, zorder=4, label="Start")
        ax.scatter([xs[-1]], [ys[-1]], color="red", s=60, zorder=4, label="End")

        # Object bounding boxes
        box_colors_plot = ["#00DCDC", "#DC00DC", "#28C828", "#DCA000", "#5050F0"]
        for i, (name, box) in enumerate(localized_boxes.items()):
            x1, y1, x2, y2 = box
            color = box_colors_plot[i % len(box_colors_plot)]
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    zorder=3,
                )
            )
            ax.text(
                x1,
                y1 - 4,
                name,
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        # Detection rate annotation
        det_pct = 100 * len(detected) / max(len(tracking_df), 1)
        ax.set_title(
            f"{video_path.stem}  —  trajectory\n"
            f"({len(detected)}/{len(tracking_df)} frames detected, {det_pct:.0f}%)",
            fontsize=10,
        )
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.invert_yaxis()  # image coords: y increases downward
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        logger.info("Trajectory plot saved to '%s'.", out_path)

    def _sample_frames_for_init(self, n: int = 200) -> list[np.ndarray]:
        """Sample frames uniformly from the first video for initialisation."""
        video_path = self.config.video_paths[0]
        skip, max_frame, step, _ = self._frame_range(video_path)
        reader = VideoReader(video_path)
        total = max_frame - skip
        sample_step = max(step, (total // n) if n > 0 else step)
        frames = []
        for _, frame in reader.iter_frames(start=skip, end=max_frame, step=sample_step):
            frames.append(frame)
            if len(frames) >= n:
                break
        return frames

    @staticmethod
    def _get_fps(video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps or 25.0
