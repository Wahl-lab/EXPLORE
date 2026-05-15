"""CLIP-based behavioral classification and active learning."""

from explore.classification.active_learning import ActiveLearner
from explore.classification.clip_classifier import CLIPClassifier

__all__ = ["CLIPClassifier", "ActiveLearner"]
