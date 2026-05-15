"""Video reading utilities.

Provides a clean iterator interface over OpenCV VideoCapture so the rest of
the codebase never has to manage capture objects directly.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Context-manager wrapper around ``cv2.VideoCapture``.

    Parameters
    ----------
    path:
        Path to the video file.

    Example
    -------
    >>> with VideoReader("animal_01.mp4") as vr:
    ...     for idx, frame in vr.iter_frames(start=150, end=9000, step=1):
    ...         process(frame)
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> VideoReader:
        self._open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Frames per second reported by the video container."""
        cap = self._get_cap()
        return cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        cap = self._get_cap()
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        cap = self._get_cap()
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        cap = self._get_cap()
        return int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_seconds(self) -> float:
        fps = self.fps
        return self.frame_count / fps if fps > 0 else 0.0

    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------

    def read_frame(self, index: int) -> np.ndarray:
        """Read a single frame by index.

        Parameters
        ----------
        index:
            Zero-based frame index.

        Returns
        -------
        np.ndarray
            BGR image.

        Raises
        ------
        RuntimeError
            If the frame cannot be read.
        """
        cap = self._get_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {index} from '{self.path}'")
        return frame

    def iter_frames(
        self,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
        crop: tuple[int, int, int, int] | None = None,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield ``(frame_index, frame)`` tuples.

        Parameters
        ----------
        start:
            First frame index (inclusive).
        end:
            Last frame index (exclusive).  Defaults to ``frame_count``.
        step:
            Read every ``step``-th frame (default 1 = every frame).
        crop:
            Optional ``(x1, y1, x2, y2)`` crop applied to each frame.
        """
        cap = self._get_cap()
        end = end if end is not None else self.frame_count
        end = min(end, self.frame_count)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        current = start

        while current < end:
            ok, frame = cap.read()
            if not ok:
                break
            if (current - start) % step == 0:
                if crop is not None:
                    x1, y1, x2, y2 = crop
                    frame = frame[y1:y2, x1:x2]
                yield current, frame
            current += 1

    def close(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: '{self.path}'")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open '{self.path}'")

    def _get_cap(self) -> cv2.VideoCapture:
        if self._cap is None:
            self._open()
        assert self._cap is not None
        return self._cap
