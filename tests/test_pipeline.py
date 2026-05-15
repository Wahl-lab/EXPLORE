"""Tests for ExplorationPipeline (end-to-end, with mocked models)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from explore.pipeline.prediction import ExplorationPipeline

# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def test_pipeline_constructs(minimal_config):
    pipeline = ExplorationPipeline(minimal_config, headless=True)
    assert pipeline.config is minimal_config


# ---------------------------------------------------------------------------
# _nearest_object
# ---------------------------------------------------------------------------


def test_nearest_object_returns_closest(minimal_config):
    pipeline = ExplorationPipeline(minimal_config, headless=True)
    # Bright region near (45, 45) → centre of "familiar" object box (10,10,80,80)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[30:60, 30:60] = 255
    localized_boxes = {o.name: o.bounding_box for o in minimal_config.objects}
    nearest = pipeline._nearest_object(frame, minimal_config.objects, localized_boxes)
    assert nearest == "familiar"


def test_nearest_object_no_contours_returns_first(minimal_config):
    """All-black frame → no contour → falls back to first object."""
    pipeline = ExplorationPipeline(minimal_config, headless=True)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    localized_boxes = {o.name: o.bounding_box for o in minimal_config.objects}
    result = pipeline._nearest_object(frame, minimal_config.objects, localized_boxes)
    assert result == minimal_config.objects[0].name


# ---------------------------------------------------------------------------
# _assign_labels (precomputed path — no video I/O needed)
# ---------------------------------------------------------------------------


def test_assign_labels_precomputed_returns_copy(minimal_config):
    """When precomputed_labels are supplied, _assign_labels returns a copy."""
    pipeline = ExplorationPipeline(minimal_config, headless=True)
    precomputed = {
        "familiar": np.array([1, 0, 1, 0, 0], dtype=np.int32),
        "novel": np.array([0, 0, 0, 1, 0], dtype=np.int32),
    }
    mask = np.array([1, 0, 1, 1, 0], dtype=bool)
    # Use any path — precomputed path never reads the file
    labels = pipeline._assign_labels(
        minimal_config.video_paths[0],
        mask,
        minimal_config.objects,
        localized_boxes={},
        precomputed_labels=precomputed,
    )
    assert (labels["familiar"] == precomputed["familiar"]).all()
    assert (labels["novel"] == precomputed["novel"]).all()
    # Must be copies, not the same array
    assert labels["familiar"] is not precomputed["familiar"]


def test_assign_labels_no_bboxes(minimal_config, fake_video_path):
    """Without bounding boxes, all exploration is assigned to first object."""
    from explore.config import ObjectConfig

    pipeline = ExplorationPipeline(minimal_config, headless=True)
    objs_no_bbox = [
        ObjectConfig(name="familiar"),
        ObjectConfig(name="novel"),
    ]
    n = 5
    mask = np.ones(n, dtype=bool)
    with (
        patch.object(
            ExplorationPipeline,
            "_frame_range",
            return_value=(0, n, 1, 4.0),
        ),
        patch(
            "explore.pipeline.prediction.VideoReader",
        ) as mock_reader,
    ):
        frames = [np.zeros((150, 150, 3), dtype=np.uint8) for _ in range(n)]
        mock_reader.return_value.iter_frames.return_value = enumerate(frames)
        labels = pipeline._assign_labels(
            fake_video_path, mask, objs_no_bbox, localized_boxes={}
        )
    assert labels["familiar"].sum() == n
    assert labels["novel"].sum() == 0


# ---------------------------------------------------------------------------
# Full run (heavily mocked)
# ---------------------------------------------------------------------------


def test_run_returns_dataframe(minimal_config, fake_video_path):
    """Full pipeline run with mocked CLIP and Grounding DINO."""
    minimal_config.video_paths = [fake_video_path]

    fake_embeddings = (
        np.random.default_rng(0).standard_normal((5, 512)).astype(np.float32)
    )
    fake_labels = {
        "familiar": np.array([1, 0, 0, 0, 0], dtype=np.int32),
        "novel": np.array([0, 0, 1, 0, 1], dtype=np.int32),
    }

    import pandas as pd

    with (
        patch.object(
            ExplorationPipeline,
            "_embed_streaming",
            return_value=(fake_embeddings, 4.0),
        ),
        patch(
            "explore.pipeline.prediction.CLIPClassifier.zero_shot_predict",
            return_value=np.array([0.8, 0.2, 0.9, 0.1, 0.7]),
        ),
        patch.object(
            ExplorationPipeline,
            "_assign_labels",
            return_value=fake_labels,
        ),
        patch.object(
            ExplorationPipeline,
            "_write_prediction_video",
            return_value=None,
        ),
        patch.object(
            ExplorationPipeline,
            "_track_animal",
            return_value=pd.DataFrame(columns=["frame", "time_s", "x", "y"]),
        ),
    ):
        pipeline = ExplorationPipeline(minimal_config, headless=True)
        result = pipeline.run()

    assert not result.empty
    assert "animal" in result.columns


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------


def test_prepare_output_dirs_creates_paths(minimal_config):
    pipeline = ExplorationPipeline(minimal_config, headless=True)
    pipeline._prepare_output_dirs()

    assert (minimal_config.project_dir / "results").exists()
    assert (minimal_config.project_dir / "results" / "prediction_videos").exists()
    assert (minimal_config.project_dir / "model").exists()
