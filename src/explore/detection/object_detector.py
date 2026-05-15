"""Grounding DINO wrapper for text-prompted object detection.

Users describe their objects in plain language; the detector finds them in the
reference frame automatically, eliminating the manual bounding-box drawing step.

Example
-------
>>> detector = ObjectDetector()
>>> frame = read_reference_frame("animal_01.mp4")
>>> results = detector.detect(frame, ["blue plastic bottle", "brown wooden cube"])
>>> for r in results:
...     print(r.label, r.box)
blue plastic bottle (45, 120, 180, 310)
brown wooden cube   (310, 130, 440, 290)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """A single detected object.

    Attributes
    ----------
    label:
        The matched text description.
    box:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    score:
        Detection confidence in ``[0, 1]``.
    """

    label: str
    box: tuple[int, int, int, int]
    score: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def centre(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class ObjectDetector:
    """Detect objects described in natural language using Grounding DINO.

    The model is loaded lazily on first use so that importing the module
    does not trigger a large download.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.
    box_threshold:
        Minimum box confidence for detection (default 0.35).
    text_threshold:
        Minimum text-box alignment confidence (default 0.25).
    device:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  Auto-selected when ``None``.
    """

    #: Animal terms appended to every DINO caption so the model can label
    #: animal detections separately and we can discard them by label.
    _ANIMAL_TERMS = "mouse . rat . rodent . animal"

    #: Any detected label containing one of these words is the animal.
    _ANIMAL_LABEL_WORDS = frozenset({"mouse", "rat", "rodent", "animal"})

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or self._auto_device()
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        descriptions: list[str],
    ) -> list[DetectionResult]:
        """Detect objects in a single BGR frame, excluding the experimental animal.

        One Grounding DINO call is made **per description**.  Animal terms
        (``mouse . rat . rodent . animal``) are appended to every caption so
        that DINO can assign explicit animal labels to animal-shaped regions.
        Any returned box whose label contains an animal word is discarded;
        the highest-scoring remaining box is kept.  This avoids the IoU-overlap
        problem of a separate animal-detection pass — even when the mouse sits
        directly on an object, DINO still returns a box labelled with the
        object description that we can accept.

        Parameters
        ----------
        frame:
            BGR image as returned by ``cv2.imread`` / ``cv2.VideoCapture``.
        descriptions:
            Natural-language descriptions, one per object.

        Returns
        -------
        list[DetectionResult]
            One entry per description that was localised as a non-animal box.
        """
        self._ensure_loaded()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        h, w = frame.shape[:2]

        results: list[DetectionResult] = []
        for desc in descriptions:
            # Append animal terms so DINO labels animal detections explicitly.
            caption = f"{desc.strip().rstrip('.')} . {self._ANIMAL_TERMS} ."
            inputs = self._processor(
                images=pil_image,
                text=caption,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            raw = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]

            boxes = raw.get("boxes", torch.empty(0, 4))
            scores = raw.get("scores", torch.empty(0))
            labels = raw.get("text_labels") or raw.get("labels", [])

            if len(scores) == 0:
                logger.warning(
                    "No detection found for '%s' (threshold=%.2f).",
                    desc,
                    self.box_threshold,
                )
                continue

            # Walk candidates highest-score-first; skip any that DINO labelled
            # as an animal.
            chosen_box: tuple[int, int, int, int] | None = None
            chosen_score = -1.0
            for idx in scores.argsort(descending=True).tolist():
                label = labels[idx] if idx < len(labels) else ""
                if any(w in label.lower() for w in self._ANIMAL_LABEL_WORDS):
                    logger.debug(
                        "  '%s' — skipping animal-labelled box '%s'", desc, label
                    )
                    continue
                x1, y1, x2, y2 = boxes[idx].tolist()
                chosen_box = (
                    max(0, int(x1)),
                    max(0, int(y1)),
                    min(w, int(x2)),
                    min(h, int(y2)),
                )
                chosen_score = float(scores[idx])
                break

            if chosen_box is not None:
                results.append(
                    DetectionResult(label=desc, box=chosen_box, score=chosen_score)
                )
            else:
                logger.warning(
                    "  '%s' — only animal-labelled detections found in this frame.",
                    desc,
                )

        return results

    def detect_from_video(
        self,
        video_path: Path | str,
        descriptions: list[str],
        reference_frame: int = 50,
        n_candidates: int = 10,
        skip_start_s: float = 60.0,
        skip_end_s: float = 60.0,
    ) -> tuple[list[DetectionResult], np.ndarray]:
        """Detect objects by sampling random frames from the middle of the video.

        Skips the first and last *skip_start_s* / *skip_end_s* seconds to
        avoid experimenter-handling artefacts (placing/removing objects or the
        animal).  Samples *n_candidates* frames **randomly** from the
        remaining window so that the mouse's position is uncorrelated across
        frames.  For each object description the detection with the highest
        confidence score is kept.

        Parameters
        ----------
        video_path:
            Path to the video file.
        descriptions:
            Natural-language descriptions, one per object.
        reference_frame:
            Used only when *n_candidates* == 1 (backward-compat fallback).
        n_candidates:
            Number of randomly sampled frames.  Default 10.
        skip_start_s:
            Seconds to exclude at the beginning of the video.  Default 60.
        skip_end_s:
            Seconds to exclude at the end of the video.  Default 60.

        Returns
        -------
        (results, best_frame)
            Best detection per description and the annotated BGR frame on
            which that detection was found.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_candidates == 1:
            candidate_indices = [reference_frame]
        else:
            # Clamp skips to at most 25 % of the video each side so short
            # recordings don't end up with an empty sampling window.
            max_skip = total * 0.25
            start = int(min(skip_start_s * fps, max_skip))
            end = total - int(min(skip_end_s * fps, max_skip))
            if end <= start:
                start, end = 0, total  # fallback: sample the whole video
            pool = range(start, end)
            k = min(n_candidates, len(pool))
            candidate_indices = random.sample(pool, k)

        logger.info(
            "Sampling %d random frames from [%s..%s] in '%s' …",
            len(candidate_indices),
            candidate_indices[0] if candidate_indices else "?",
            candidate_indices[-1] if candidate_indices else "?",
            Path(video_path).name,
        )

        # best[desc] = (score, DetectionResult, frame_img)
        best: dict[str, tuple[float, DetectionResult, np.ndarray]] = {}

        for idx in sorted(candidate_indices):  # sorted for sequential seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            for result in self.detect(frame, descriptions):
                prev_score = best.get(result.label, (-1.0, None, None))[0]
                if result.score > prev_score:
                    best[result.label] = (result.score, result, frame.copy())
                    logger.debug(
                        "  '%s' — new best %.2f at frame %d",
                        result.label,
                        result.score,
                        idx,
                    )

        cap.release()

        if not best:
            raise RuntimeError(
                f"No objects detected in any of the {n_candidates} candidate "
                f"frames from '{video_path}'."
            )

        for d in descriptions:
            if d in best:
                logger.info("  '%s' — best score %.2f", d, best[d][0])
            else:
                logger.warning(
                    "  '%s' — no detection found across %d frames.", d, n_candidates
                )

        final_results = [best[d][1] for d in descriptions if d in best]
        best_frame = max(best.values(), key=lambda t: t[0])[2]
        return final_results, best_frame

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        detections: list[DetectionResult],
        names: list[str],
    ) -> np.ndarray:
        """Draw detection boxes on a copy of *frame*.

        Each box is labelled with its assigned name and a truncated description
        so the user can verify which description matched which region before
        committing to role labels (familiar / novel / …).

        Parameters
        ----------
        frame:
            BGR source image.
        detections:
            One ``DetectionResult`` per object (may be shorter than *names*
            if some descriptions had no match).
        names:
            Display labels, one per detection (e.g. ``["object_0", "object_1"]``
            or final role names).
        """
        # Colours chosen to be distinct on typical grey/brown arena floors
        palette = [
            (0, 220, 220),  # cyan
            (220, 0, 220),  # magenta
            (40, 200, 40),  # green
            (220, 160, 0),  # amber
            (80, 80, 240),  # blue
        ]
        out = frame.copy()
        for i, (det, name) in enumerate(zip(detections, names, strict=False)):
            color = palette[i % len(palette)]
            x1, y1, x2, y2 = det.box

            # Semi-transparent fill
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

            # Solid border
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label: "object_0: small wooden cube…  (0.82)"
            desc_short = det.label if len(det.label) <= 35 else det.label[:33] + "…"
            label = f"{name}: {desc_short}  ({det.score:.2f})"
            label_y = max(y1 - 8, 18)
            # Dark background behind text for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                out, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), (0, 0, 0), -1
            )
            cv2.putText(
                out,
                label,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return out  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.38 is required for object detection. "
                "Install with: pip install transformers>=4.38"
            ) from exc

        logger.info("Loading Grounding DINO model '%s' …", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self._model.eval()
        logger.info("Grounding DINO loaded on %s.", self.device)

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
