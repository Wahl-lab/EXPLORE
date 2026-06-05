"""Shared pytest fixtures for EXPLORE 2.0.

All fixtures that involve ML models use mocks so the test suite runs on CI
without downloading model weights or requiring a GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture
def random_bgr_frame() -> np.ndarray:
    """Single 150×150 BGR frame filled with random noise."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (150, 150, 3), dtype=np.uint8)


@pytest.fixture
def random_bgr_frames() -> list[np.ndarray]:
    """20 synthetic BGR frames."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (150, 150, 3), dtype=np.uint8) for _ in range(20)]


@pytest.fixture
def random_embeddings() -> np.ndarray:
    """Synthetic L2-normalised CLIP embeddings, shape (20, 512)."""
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((20, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


@pytest.fixture
def binary_labels() -> np.ndarray:
    """10 positive + 10 negative labels, shape (20,)."""
    return np.array([1] * 10 + [0] * 10, dtype=np.int32)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config(tmp_path: Path):
    """Minimal valid ExperimentConfig pointing to tmp_path."""
    from explore.config import (
        AnalysisConfig,
        BehaviorConfig,
        ExperimentConfig,
        ModelConfig,
        ObjectConfig,
    )

    return ExperimentConfig(
        project_name="test_project",
        project_path=tmp_path,
        video_paths=[tmp_path / "video.mp4"],
        video_duration_minutes=2,
        objects=[
            ObjectConfig(
                name="familiar",
                bounding_box=(10, 10, 80, 80),
            ),
            ObjectConfig(
                name="novel",
                bounding_box=(100, 10, 170, 80),
            ),
        ],
        behavior=BehaviorConfig(
            exploration_prompts=["mouse sniffing object"],
            no_exploration_prompts=["mouse walking away"],
        ),
        model=ModelConfig(),
        analysis=AnalysisConfig(),
    )


# ---------------------------------------------------------------------------
# Mock CLIP classifier
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_clip_classifier(random_embeddings):
    """CLIPClassifier with mocked backbone — no torch/open_clip required."""
    from explore.classification.clip_classifier import CLIPClassifier

    clf = CLIPClassifier.__new__(CLIPClassifier)
    clf.model_name = "ViT-B-32"
    clf.pretrained = "openai"
    clf.batch_size = 64
    clf.device = "cpu"
    clf._model = None
    clf._preprocess = None
    clf._tokenizer = None
    clf._head = None

    # Prevent _ensure_loaded from triggering open_clip download
    clf._ensure_loaded = MagicMock()

    # Stub embed_frames to return pre-computed embeddings
    clf.embed_frames = MagicMock(return_value=random_embeddings)

    # Stub embed_texts to return unit vectors
    def _embed_texts(texts):
        rng = np.random.default_rng(len(texts))
        emb = rng.standard_normal((len(texts), 512)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    clf.embed_texts = MagicMock(side_effect=_embed_texts)
    return clf


# ---------------------------------------------------------------------------
# Mock object detector
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_detector():
    """ObjectDetector whose detect() returns pre-canned results."""
    from explore.detection.object_detector import DetectionResult, ObjectDetector

    detector = ObjectDetector.__new__(ObjectDetector)
    detector.model_id = "IDEA-Research/grounding-dino-base"
    detector.box_threshold = 0.35
    detector.text_threshold = 0.25
    detector.device = "cpu"
    detector._model = None
    detector._processor = None

    canned = [
        DetectionResult(label="blue plastic bottle", box=(10, 10, 80, 80), score=0.82),
        DetectionResult(label="brown wooden cube", box=(100, 10, 170, 80), score=0.74),
    ]
    canned_frame = np.zeros((150, 150, 3), dtype=np.uint8)
    detector.detect = MagicMock(return_value=canned)
    detector.detect_from_video = MagicMock(return_value=(canned, canned_frame))
    return detector


# ---------------------------------------------------------------------------
# Fake video (synthetic .mp4 via OpenCV)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_video_path(tmp_path: Path) -> Path:
    """Write a tiny 30-frame synthetic video and return its path."""
    import cv2

    out_path = tmp_path / "test_video.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (150, 150),
    )
    rng = np.random.default_rng(7)
    for _ in range(30):
        frame = rng.integers(0, 255, (150, 150, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return out_path
