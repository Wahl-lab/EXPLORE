"""Tests for ExperimentConfig and related dataclasses."""

from __future__ import annotations

import pytest
import yaml

from explore.config import (
    AnalysisConfig,
    BehaviorConfig,
    ExperimentConfig,
    ObjectConfig,
)

# ---------------------------------------------------------------------------
# ObjectConfig
# ---------------------------------------------------------------------------


def test_object_config_valid():
    obj = ObjectConfig(name="novel")
    assert obj.name == "novel"
    assert obj.bounding_box is None


def test_object_config_with_bbox():
    obj = ObjectConfig(name="obj", bounding_box=(0, 0, 50, 50))
    assert obj.bounding_box == (0, 0, 50, 50)


def test_object_config_empty_name_raises():
    with pytest.raises(ValueError, match="name"):
        ObjectConfig(name="")


def test_object_config_whitespace_name_raises():
    with pytest.raises(ValueError, match="name"):
        ObjectConfig(name="  ")


def test_object_config_invalid_bbox_raises():
    with pytest.raises(ValueError, match="bounding_box"):
        ObjectConfig(name="obj", bounding_box=(50, 0, 10, 50))


# ---------------------------------------------------------------------------
# BehaviorConfig
# ---------------------------------------------------------------------------


def test_behavior_config_valid():
    b = BehaviorConfig(
        exploration_prompts=["mouse sniffing object"],
        no_exploration_prompts=["mouse walking away"],
    )
    assert b.confidence_threshold == 0.5
    assert b.min_bout_seconds == 1.0


def test_behavior_config_empty_exploration_raises():
    with pytest.raises(ValueError, match="exploration prompt"):
        BehaviorConfig(exploration_prompts=[], no_exploration_prompts=["x"])


def test_behavior_config_empty_no_exploration_raises():
    with pytest.raises(ValueError, match="no-exploration prompt"):
        BehaviorConfig(exploration_prompts=["x"], no_exploration_prompts=[])


def test_behavior_config_invalid_threshold_raises():
    with pytest.raises(ValueError, match="confidence_threshold"):
        BehaviorConfig(
            exploration_prompts=["x"],
            no_exploration_prompts=["y"],
            confidence_threshold=1.5,
        )


# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------


def test_analysis_config_defaults():
    a = AnalysisConfig()
    assert a.bin_duration_minutes == 1
    assert a.compute_di is True


def test_analysis_config_custom():
    a = AnalysisConfig(bin_duration_minutes=2, compute_di=False)
    assert a.bin_duration_minutes == 2
    assert a.compute_di is False


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


def test_experiment_config_constructs(tmp_path, minimal_config):
    assert minimal_config.project_name == "test_project"
    assert minimal_config.project_dir == tmp_path / "test_project"


def test_project_dir_property(tmp_path, minimal_config):
    assert minimal_config.project_dir == tmp_path / "test_project"


def test_experiment_config_save_load(tmp_path, minimal_config):
    saved = minimal_config.save(tmp_path / "config.yaml")
    assert saved.exists()

    loaded = ExperimentConfig.from_yaml(saved)
    assert loaded.project_name == minimal_config.project_name
    assert len(loaded.objects) == len(minimal_config.objects)
    assert loaded.objects[0].name == "familiar"
    assert loaded.objects[0].bounding_box == (10, 10, 80, 80)
    assert (
        loaded.behavior.exploration_prompts
        == minimal_config.behavior.exploration_prompts
    )


def test_experiment_config_from_dict_defaults(tmp_path):
    raw = {
        "project_name": "x",
        "project_path": str(tmp_path),
        "video_paths": [str(tmp_path / "v.mp4")],
        "video_duration_minutes": 3,
        "objects": [{"name": "obj", "bounding_box": None}],
        "behavior": {
            "exploration_prompts": ["sniffing"],
            "no_exploration_prompts": ["walking"],
        },
    }
    cfg = ExperimentConfig.from_dict(raw)
    assert cfg.model.clip_model == "ViT-B-32"
    assert cfg.analysis.bin_duration_minutes == 1
    assert cfg.analysis.compute_di is True


def test_to_dict_round_trip(minimal_config):
    d = minimal_config.to_dict()
    loaded = ExperimentConfig.from_dict(d)
    assert loaded.project_name == minimal_config.project_name
    assert loaded.objects[0].name == minimal_config.objects[0].name
    assert loaded.objects[1].name == minimal_config.objects[1].name


def test_yaml_file_is_human_readable(tmp_path, minimal_config):
    saved = minimal_config.save(tmp_path / "cfg.yaml")
    with open(saved) as fh:
        raw = yaml.safe_load(fh)
    assert "project_name" in raw
    assert "objects" in raw
    assert isinstance(raw["objects"][0]["name"], str)


def test_experiment_config_empty_name_raises(tmp_path):
    with pytest.raises(ValueError, match="project_name"):
        ExperimentConfig(
            project_name="  ",
            project_path=tmp_path,
            video_paths=[tmp_path / "v.mp4"],
            video_duration_minutes=5,
            objects=[ObjectConfig(name="familiar")],
            behavior=BehaviorConfig(["sniff"], ["walk"]),
        )


def test_experiment_config_no_objects_raises(tmp_path):
    with pytest.raises(ValueError, match="object"):
        ExperimentConfig(
            project_name="x",
            project_path=tmp_path,
            video_paths=[tmp_path / "v.mp4"],
            video_duration_minutes=5,
            objects=[],
            behavior=BehaviorConfig(["sniff"], ["walk"]),
        )


def test_experiment_config_from_dict_ignores_unknown_keys(tmp_path):
    raw = {
        "project_name": "x",
        "project_path": str(tmp_path),
        "video_paths": [str(tmp_path / "v.mp4")],
        "video_duration_minutes": 3,
        "objects": [{"name": "familiar", "bounding_box": None}],
        "behavior": {
            "exploration_prompts": ["sniffing"],
            "no_exploration_prompts": ["walking"],
        },
        "unknown_future_key": "some_value",
    }
    cfg = ExperimentConfig.from_dict(raw)
    assert cfg.project_name == "x"
