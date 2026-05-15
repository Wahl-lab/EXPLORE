"""Tests for ObjectDetector (Grounding DINO wrapper)."""

from __future__ import annotations

import numpy as np
import pytest

from explore.detection.object_detector import DetectionResult, ObjectDetector

# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


def test_detection_result_area():
    r = DetectionResult(label="bottle", box=(0, 0, 100, 50), score=0.9)
    assert r.area == 5000


def test_detection_result_centre():
    r = DetectionResult(label="cube", box=(10, 20, 90, 80), score=0.7)
    assert r.centre() == (50, 50)


def test_detection_result_zero_area():
    r = DetectionResult(label="x", box=(50, 50, 50, 50), score=0.1)
    assert r.area == 0


# ---------------------------------------------------------------------------
# ObjectDetector — canned mock (no model loading)
# ---------------------------------------------------------------------------


def test_detect_returns_one_result_per_description(mock_detector):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    results = mock_detector.detect(frame, ["blue plastic bottle", "brown wooden cube"])
    assert len(results) == 2


def test_detect_result_fields(mock_detector):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    results = mock_detector.detect(frame, ["blue plastic bottle"])
    r = results[0]
    assert isinstance(r.label, str)
    assert len(r.box) == 4
    assert 0 <= r.score <= 1


def test_detect_from_video_called_with_right_frame(mock_detector, fake_video_path):
    results, frame = mock_detector.detect_from_video(
        fake_video_path, ["bottle"], reference_frame=5
    )
    assert all(isinstance(r, DetectionResult) for r in results)
    assert isinstance(frame, np.ndarray)


# ---------------------------------------------------------------------------
# ObjectDetector — lazy loading guard (no actual model download)
# ---------------------------------------------------------------------------


def test_detect_raises_import_error_if_transformers_missing(
    monkeypatch, random_bgr_frame
):
    """If transformers is not installed, detect() raises ImportError."""
    import builtins

    real_import = builtins.__import__

    def _block_transformers(name, *args, **kwargs):
        if "transformers" in name:
            raise ImportError("transformers not found")
        return real_import(name, *args, **kwargs)

    detector = ObjectDetector()  # not loaded yet
    monkeypatch.setattr(builtins, "__import__", _block_transformers)

    with pytest.raises(ImportError, match="transformers"):
        detector._ensure_loaded()
