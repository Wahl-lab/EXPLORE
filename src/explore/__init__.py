"""
EXPLORE 2.0 — Automated exploration behavior analysis for object recognition tests.

Pipeline overview
-----------------
1. **Manual labeling** — draw bounding boxes around each object on a reference
   frame in the GUI; assign a name to each object.
2. **ORB re-localization** — the pipeline uses ORB feature matching to
   transfer boxes to every other video, accounting for minor view shifts.
3. **Behavioral definition** — describe what counts as exploration in your
   experiment using natural language; CLIP classifies every frame zero-shot.
4. **Active learning** — the system surfaces only the frames it is uncertain
   about; you label ~20–50 frames to fine-tune a lightweight linear head.
5. **Analysis** — exploration time, frequency, Discrimination Index and
   Recognition Index are computed per animal per time bin.

Quickstart
----------
>>> from explore import ExperimentConfig, ExplorationPipeline
>>> cfg = ExperimentConfig.from_yaml("my_experiment.yaml")
>>> pipeline = ExplorationPipeline(cfg)
>>> pipeline.set_reference_frame(reference_bgr_frame)
>>> results = pipeline.run()
>>> results.to_csv("results.csv", index=False)
"""

from explore.config import (
    AnalysisConfig,
    BehaviorConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectConfig,
)
from explore.detection.box_localizer import BoxLocalizer, LocalizationResult
from explore.pipeline.analysis import BehaviorAnalyzer
from explore.pipeline.prediction import ExplorationPipeline

__version__ = "2.0.0"
__author__ = "Victor Ibañez"

__all__ = [
    "ExperimentConfig",
    "ObjectConfig",
    "BehaviorConfig",
    "ModelConfig",
    "AnalysisConfig",
    "ExplorationPipeline",
    "BehaviorAnalyzer",
    "BoxLocalizer",
    "LocalizationResult",
]
