"""Behavioral statistics for object recognition tests.

Standard metrics
----------------
* **Exploration time** — total seconds spent exploring each object per time bin.
* **Exploration frequency** — number of discrete bouts per time bin.
* **Discrimination Index (DI)** — ``(novel - familiar) / (novel + familiar)``.
  Positive values indicate a preference for the novel object (memory).
* **Recognition Index (RI)** — ``novel / (novel + familiar)``.
  Values > 0.5 indicate novelty preference.

All metrics are computed per animal (video) and per time bin, then aggregated
into a single tidy ``pd.DataFrame`` for downstream statistics in R/Python.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BehaviorAnalyzer:
    """Compute exploration statistics from per-frame predictions.

    Parameters
    ----------
    fps:
        Video frame rate (Hz).
    bin_duration_seconds:
        Length of each time bin in seconds (default 60 s = 1 min).
    min_bout_seconds:
        Minimum duration for a continuous exploration bout to be counted
        (default 1.0 s).  Shorter bouts are discarded.
    """

    def __init__(
        self,
        fps: float,
        bin_duration_seconds: float = 60.0,
        min_bout_seconds: float = 1.0,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if bin_duration_seconds <= 0:
            raise ValueError(
                f"bin_duration_seconds must be > 0, got {bin_duration_seconds}"
            )
        if min_bout_seconds < 0:
            raise ValueError(f"min_bout_seconds must be >= 0, got {min_bout_seconds}")

        self.fps = fps
        self.bin_duration_seconds = bin_duration_seconds
        self.min_bout_frames = int(min_bout_seconds * fps)

    # ------------------------------------------------------------------
    # Per-object statistics
    # ------------------------------------------------------------------

    def compute(
        self,
        object_labels: dict[str, np.ndarray],
        animal_id: str,
        experiment_id: str = "",
    ) -> pd.DataFrame:
        """Compute per-bin statistics for a single animal.

        Parameters
        ----------
        object_labels:
            Mapping ``{object_name: binary_array}`` where the array has one
            entry per analysed frame (``1`` = exploring that object).
        animal_id:
            Identifier for this animal / video (used in output rows).
        experiment_id:
            Optional experiment label.

        Returns
        -------
        pd.DataFrame
            Columns: ``experiment``, ``animal``, ``minute``,
            ``<obj>_time_s``, ``<obj>_freq``, plus DI/RI when applicable.
        """
        if not object_labels:
            raise ValueError("object_labels must not be empty")

        # Enforce minimum bout duration by zeroing short runs
        filtered = {
            name: self._filter_bouts(arr) for name, arr in object_labels.items()
        }

        n_frames = next(iter(filtered.values())).size
        bin_frames = int(self.bin_duration_seconds * self.fps)
        n_bins = max(1, n_frames // bin_frames)

        rows = []
        for b in range(n_bins):
            start = b * bin_frames
            end = min(start + bin_frames, n_frames)
            row: dict[str, Any] = {
                "experiment": experiment_id,
                "animal": animal_id,
                "minute": (b + 1) * (self.bin_duration_seconds / 60),
            }
            for name, arr in filtered.items():
                segment = arr[start:end]
                row[f"{name}_time_s"] = float(segment.sum()) / self.fps
                row[f"{name}_freq"] = int(self._count_bouts(segment))
            rows.append(row)

        return pd.DataFrame(rows)

    def add_di_ri(
        self,
        df: pd.DataFrame,
        novel_col: str,
        familiar_col: str,
    ) -> pd.DataFrame:
        """Append DI and RI columns derived from exploration-time columns.

        Parameters
        ----------
        df:
            DataFrame from :meth:`compute`.
        novel_col:
            Column name for novel object exploration time (e.g. ``"novel_time_s"``).
        familiar_col:
            Column name for familiar object exploration time.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with two new columns: ``DI`` and ``RI``.
        """
        missing = {novel_col, familiar_col} - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

        t_n = df[novel_col]
        t_f = df[familiar_col]
        total = t_n + t_f

        df = df.copy()
        df["DI"] = np.where(total > 0, (t_n - t_f) / total, 0.0)
        df["RI"] = np.where(total > 0, t_n / total, 0.5)
        return df

    # ------------------------------------------------------------------
    # Aggregate across animals
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate per-animal DataFrames into a single tidy table."""
        if not frames:
            raise ValueError("frames list must not be empty")
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def summary(df: pd.DataFrame) -> pd.DataFrame:
        """Group-by animal, compute mean ± SEM across time bins."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "minute"]

        agg = df.groupby("animal")[numeric_cols].agg(["mean", "sem"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        return agg.reset_index()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def discrimination_index(t_novel: float, t_familiar: float) -> float:
        """``(novel - familiar) / (novel + familiar)``."""
        total = t_novel + t_familiar
        return (t_novel - t_familiar) / total if total > 0 else 0.0

    @staticmethod
    def recognition_index(t_novel: float, t_familiar: float) -> float:
        """``novel / (novel + familiar)``."""
        total = t_novel + t_familiar
        return t_novel / total if total > 0 else 0.5

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _filter_bouts(self, arr: np.ndarray) -> np.ndarray:
        """Zero-out exploration runs shorter than min_bout_frames."""
        if self.min_bout_frames <= 1:
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
                if (i - start) < self.min_bout_frames:
                    out[start:i] = 0
        return out  # type: ignore[no-any-return]

    def _count_bouts(self, arr: np.ndarray) -> int:
        """Count rising edges (0→1 transitions) in a binary array."""
        if len(arr) == 0:
            return 0
        padded = np.concatenate([[0], arr])
        return int(np.sum(np.diff(padded) == 1))
