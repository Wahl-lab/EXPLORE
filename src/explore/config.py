"""Experiment configuration — dataclasses + YAML round-trip.

Every experiment is fully described by a single ``ExperimentConfig`` object
that can be serialised to and loaded from a YAML file.

Example YAML
------------
.. code-block:: yaml

    project_name: NOR_cohort_A
    project_path: /data/experiments/cohort_A
    video_paths:
      - /data/videos/animal_01.mp4
      - /data/videos/animal_02.mp4
    video_duration_minutes: 5

    objects:
      - name: familiar
        bounding_box: [120, 80, 310, 260]
      - name: novel
        bounding_box: [450, 90, 640, 275]

    behavior:
      exploration_prompts:
        - a mouse actively sniffing and investigating an object
        - a rodent with nose close to an object exploring it
      no_exploration_prompts:
        - a mouse walking away from or ignoring objects
        - a rodent resting or grooming away from objects

    analysis:
      familiar_object: familiar
      novel_object: novel
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ObjectConfig:
    """A single tracked object in the arena.

    Parameters
    ----------
    name:
        Short identifier used in output columns (e.g. ``"novel"``).
    bounding_box:
        ``(x1, y1, x2, y2)`` pixel coordinates drawn on the reference frame.
        ``None`` means the object has not been labeled yet.
    """

    name: str
    bounding_box: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ObjectConfig.name must not be empty")
        if self.bounding_box is not None:
            x1, y1, x2, y2 = self.bounding_box
            if x1 >= x2 or y1 >= y2:
                raise ValueError(
                    f"Invalid bounding_box {self.bounding_box}: "
                    "x1 must be < x2 and y1 must be < y2"
                )


@dataclass
class BehaviorConfig:
    """Text-based behavioural definition for CLIP classification.

    Parameters
    ----------
    exploration_prompts:
        Sentences describing frames that *count* as exploration.
        Multiple prompts are averaged — use them to cover different phrasings.
    no_exploration_prompts:
        Sentences describing frames that do *not* count as exploration.
    confidence_threshold:
        Probability cut-off for the exploration class (default 0.5).
    min_bout_seconds:
        Consecutive exploration frames shorter than this are discarded to
        avoid counting brief pass-bys (default 1.0 s).
    """

    exploration_prompts: list[str]
    no_exploration_prompts: list[str]
    confidence_threshold: float = 0.5
    min_bout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.exploration_prompts:
            raise ValueError("At least one exploration prompt is required")
        if not self.no_exploration_prompts:
            raise ValueError("At least one no-exploration prompt is required")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be in (0, 1)")
        if self.min_bout_seconds < 0:
            raise ValueError("min_bout_seconds must be >= 0")


@dataclass
class ModelConfig:
    """Model identifiers for CLIP.

    Parameters
    ----------
    clip_model:
        OpenCLIP model name (default ``"ViT-B-32"``).
    clip_pretrained:
        OpenCLIP pretrained weights tag (default ``"openai"``).
    """

    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"


@dataclass
class AnalysisConfig:
    """Output statistics configuration.

    Parameters
    ----------
    bin_duration_minutes:
        Time window for per-bin statistics (default 1 min).
    compute_di:
        Compute DI/RI for every pair of objects.
        DI = (t_A - t_B) / (t_A + t_B),  RI = t_A / (t_A + t_B).
    """

    bin_duration_minutes: int = 1
    compute_di: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Parameters
    ----------
    project_name:
        Short name used for folder creation and output filenames.
    project_path:
        Root directory where the project folder is created.
    video_paths:
        List of video files to analyse.
    video_duration_minutes:
        Recording duration to analyse (the first 5 s are always skipped).
    objects:
        Arena objects (minimum 1, each with a drawn bounding box).
    behavior:
        Behavioural definition via CLIP text prompts.
    model:
        CLIP model selection.
    analysis:
        Output statistics parameters.
    """

    project_name: str
    project_path: Path
    video_paths: list[Path]
    video_duration_minutes: int
    objects: list[ObjectConfig]
    behavior: BehaviorConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def __post_init__(self) -> None:
        self.project_path = Path(self.project_path)
        self.video_paths = [Path(p) for p in self.video_paths]
        if not self.project_name.strip():
            raise ValueError("project_name must not be empty")
        if not self.objects:
            raise ValueError("At least one object must be defined")
        if self.video_duration_minutes <= 0:
            raise ValueError("video_duration_minutes must be > 0")

    @property
    def project_dir(self) -> Path:
        """Resolved project output directory."""
        return self.project_path / self.project_name

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def _bbox(bb: tuple[int, int, int, int] | None) -> list[int] | None:
            return list(bb) if bb is not None else None

        return {
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "video_paths": [str(p) for p in self.video_paths],
            "video_duration_minutes": self.video_duration_minutes,
            "objects": [
                {"name": o.name, "bounding_box": _bbox(o.bounding_box)}
                for o in self.objects
            ],
            "behavior": {
                "exploration_prompts": self.behavior.exploration_prompts,
                "no_exploration_prompts": self.behavior.no_exploration_prompts,
                "confidence_threshold": self.behavior.confidence_threshold,
                "min_bout_seconds": self.behavior.min_bout_seconds,
            },
            "model": {
                "clip_model": self.model.clip_model,
                "clip_pretrained": self.model.clip_pretrained,
            },
            "analysis": {
                "bin_duration_minutes": self.analysis.bin_duration_minutes,
                "compute_di": self.analysis.compute_di,
            },
        }

    def save(self, path: Path | None = None) -> Path:
        """Serialise to YAML.  Defaults to ``<project_dir>/config.yaml``."""
        out = path or (self.project_dir / "config.yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Construct from a plain dict (e.g. loaded from YAML).

        Tolerates old-format YAMLs that carry a ``description`` field or a
        ``gdino_model`` key — those are silently ignored.
        """
        data = copy.deepcopy(data)

        raw_objects = data.pop("objects")
        objects = []
        for o in raw_objects:
            bb = o.get("bounding_box")
            objects.append(
                ObjectConfig(
                    name=o["name"],
                    bounding_box=tuple(bb) if bb else None,
                )
            )

        beh = data.pop("behavior")
        behavior = BehaviorConfig(
            exploration_prompts=beh["exploration_prompts"],
            no_exploration_prompts=beh["no_exploration_prompts"],
            confidence_threshold=beh.get("confidence_threshold", 0.5),
            min_bout_seconds=beh.get("min_bout_seconds", 1.0),
        )

        mdl = data.pop("model", {})
        model = ModelConfig(
            clip_model=mdl.get("clip_model", "ViT-B-32"),
            clip_pretrained=mdl.get("clip_pretrained", "openai"),
        )

        ana = data.pop("analysis", {})
        analysis = AnalysisConfig(
            bin_duration_minutes=ana.get("bin_duration_minutes", 1),
            compute_di=ana.get("compute_di", True),
        )

        # Drop any unknown top-level keys for forward compatibility
        known = {
            "project_name",
            "project_path",
            "video_paths",
            "video_duration_minutes",
        }
        extra = set(data.keys()) - known
        for k in extra:
            data.pop(k)

        return cls(
            objects=objects,
            behavior=behavior,
            model=model,
            analysis=analysis,
            **data,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load from a YAML file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)
