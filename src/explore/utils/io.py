"""Filesystem and result IO utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def save_results(
    df: pd.DataFrame, path: Path | str, *, avoid_overwrite: bool = True
) -> Path:
    """Save a results DataFrame to CSV.

    Parameters
    ----------
    df:
        Results table to save.
    path:
        Target CSV path.
    avoid_overwrite:
        When ``True`` (default), appends a counter suffix if the file already
        exists rather than overwriting it.

    Returns
    -------
    Path
        The actual path written.
    """
    path = Path(path)
    if avoid_overwrite and path.exists():
        stem, suffix = path.stem, path.suffix
        counter = 1
        while path.exists():
            path = path.parent / f"{stem}_{counter}{suffix}"
            counter += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Results saved to '%s'.", path)
    return path


def save_head(head: object, path: Path | str) -> None:
    """Serialise a fitted sklearn estimator (logistic head)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, path)
    logger.info("Model head saved to '%s'.", path)


def load_head(path: Path | str) -> object:
    """Load a previously saved sklearn estimator."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Head file not found: '{path}'")
    return joblib.load(path)


def write_training_log(
    log: dict,  # type: ignore[type-arg]
    path: Path | str,
) -> None:
    """Write training metadata to a JSON log file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(log, fh, indent=2, default=str)
    logger.info("Training log written to '%s'.", path)
