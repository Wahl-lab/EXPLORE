"""ORB-based bounding box re-localization across video frames.

When a user draws bounding boxes on a reference frame, this module transfers
those boxes to frames from other videos by computing the median keypoint
translation from ORB feature matches near each object region.

Objects in NOR/ORT arenas are stationary, so a rigid translation model is
sufficient — scale and rotation invariance are not needed.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BOX_COLORS = [
    (0, 220, 220),  # cyan
    (220, 0, 220),  # magenta
    (40, 200, 40),  # green
    (220, 160, 0),  # amber
    (80, 80, 240),  # blue
]


@dataclass
class LocalizationResult:
    """Result of re-localizing a bounding box to a new frame.

    Attributes
    ----------
    box:
        Re-localized ``(x1, y1, x2, y2)`` in the target frame.
    translation:
        Estimated pixel shift ``(tx, ty)`` applied to the reference box.
    n_matches:
        Number of good ORB matches used for the estimate.
    success:
        ``False`` if too few matches were found; ``box`` equals the original
        reference box unchanged.
    """

    box: tuple[int, int, int, int]
    translation: tuple[float, float]
    n_matches: int
    success: bool


class BoxLocalizer:
    """Re-localize bounding boxes across videos using ORB feature matching.

    Strategy
    --------
    1. Detect ORB keypoints on the full reference and target frames.
    2. Filter reference keypoints to those inside the (margin-expanded) box.
    3. Match filtered keypoints to the target frame with Lowe's ratio test.
    4. Compute the median translation from the matched pairs.
    5. Apply translation to box corners; clamp to frame bounds.

    Parameters
    ----------
    n_features:
        Maximum ORB keypoints per frame.  Default 2000.
    min_matches:
        Minimum good matches required to trust the translation.  Default 6.
    search_margin:
        Extra pixels around the box used to broaden keypoint selection.
        Default 30.
    ratio_thresh:
        Lowe's ratio test threshold.  Default 0.75.
    """

    def __init__(
        self,
        n_features: int = 2000,
        min_matches: int = 6,
        search_margin: int = 30,
        ratio_thresh: float = 0.75,
    ) -> None:
        self.n_features = n_features
        self.min_matches = min_matches
        self.search_margin = search_margin
        self.ratio_thresh = ratio_thresh
        self._orb = cv2.ORB_create(n_features)  # type: ignore[attr-defined]
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(
        self,
        reference_frame: np.ndarray,
        reference_box: tuple[int, int, int, int],
        target_frame: np.ndarray,
    ) -> LocalizationResult:
        """Shift *reference_box* to match the same region in *target_frame*.

        Parameters
        ----------
        reference_frame:
            BGR frame on which the box was originally drawn.
        reference_box:
            ``(x1, y1, x2, y2)`` in *reference_frame* pixel coordinates.
        target_frame:
            BGR frame to transfer the box to.

        Returns
        -------
        LocalizationResult
            ``success=False`` means too few matches were found; the returned
            ``box`` is the unshifted original.
        """
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = reference_box
        m = self.search_margin

        ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

        ref_kps, ref_desc = self._orb.detectAndCompute(ref_gray, None)
        tgt_kps, tgt_desc = self._orb.detectAndCompute(tgt_gray, None)

        fallback = LocalizationResult(
            box=reference_box, translation=(0.0, 0.0), n_matches=0, success=False
        )

        if ref_desc is None or tgt_desc is None or not ref_kps or not tgt_kps:
            logger.warning("ORB: no keypoints detected; keeping original box.")
            return fallback

        # Keep only reference keypoints inside the margin-expanded box
        box_idx = [
            i
            for i, kp in enumerate(ref_kps)
            if (x1 - m) <= kp.pt[0] <= (x2 + m) and (y1 - m) <= kp.pt[1] <= (y2 + m)
        ]
        if len(box_idx) < 2:
            logger.warning("ORB: <2 keypoints near box; keeping original.")
            return fallback

        box_kps = [ref_kps[i] for i in box_idx]
        box_desc = ref_desc[box_idx]

        raw_matches = self._matcher.knnMatch(box_desc, tgt_desc, k=2)
        good = [
            m1
            for pair in raw_matches
            if len(pair) == 2
            for m1, m2 in [pair]
            if m1.distance < self.ratio_thresh * m2.distance
        ]

        if len(good) < self.min_matches:
            logger.warning(
                "ORB: only %d good matches (need %d) for box %s; keeping original.",
                len(good),
                self.min_matches,
                reference_box,
            )
            return LocalizationResult(
                box=reference_box,
                translation=(0.0, 0.0),
                n_matches=len(good),
                success=False,
            )

        translations = np.array(
            [
                np.array(tgt_kps[g.trainIdx].pt) - np.array(box_kps[g.queryIdx].pt)
                for g in good
            ]
        )
        tx = float(np.median(translations[:, 0]))
        ty = float(np.median(translations[:, 1]))

        new_box = (
            max(0, int(x1 + tx)),
            max(0, int(y1 + ty)),
            min(w, int(x2 + tx)),
            min(h, int(y2 + ty)),
        )
        logger.info(
            "ORB: %s → %s  shift=(%.1f, %.1f)  matches=%d",
            reference_box,
            new_box,
            tx,
            ty,
            len(good),
        )
        return LocalizationResult(
            box=new_box,
            translation=(tx, ty),
            n_matches=len(good),
            success=True,
        )

    def localize_from_video(
        self,
        reference_frame: np.ndarray,
        reference_box: tuple[int, int, int, int],
        video_path: Path | str,
        n_candidates: int = 5,
        skip_s: float = 60.0,
    ) -> LocalizationResult:
        """Sample frames from *video_path* and return the best localization.

        Skips the first and last *skip_s* seconds (experimenter handling),
        then randomly samples *n_candidates* frames from the remaining window.
        Returns the result with the highest number of good matches.
        """
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        start = int(min(skip_s * fps, total * 0.25))
        end = total - int(min(skip_s * fps, total * 0.25))
        if end <= start:
            start, end = 0, max(1, total)

        pool = list(range(start, end))
        k = min(n_candidates, len(pool))
        indices = random.sample(pool, k) if k > 0 else [0]

        best: LocalizationResult | None = None
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            result = self.localize(reference_frame, reference_box, frame)
            if result.success and (best is None or result.n_matches > best.n_matches):
                best = result

        cap.release()
        return best or LocalizationResult(
            box=reference_box,
            translation=(0.0, 0.0),
            n_matches=0,
            success=False,
        )

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        boxes: dict[str, tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Draw labeled boxes on a copy of *frame* for visual verification."""
        out = frame.copy()
        for i, (name, box) in enumerate(boxes.items()):
            color = _BOX_COLORS[i % len(_BOX_COLORS)]
            x1, y1, x2, y2 = box
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(
                out,
                name,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return out  # type: ignore[no-any-return]
