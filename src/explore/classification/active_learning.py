"""Active-learning loop for minimal-labeling model refinement.

The key idea: instead of showing the researcher a full video to scroll through,
we identify the frames the model is *genuinely unsure about* and ask only about
those.  A typical correction session involves 20–50 frames and takes 2–5 min.

Workflow
--------
1. Run ``CLIPClassifier.zero_shot_predict`` on all frames.
2. ``ActiveLearner.query`` returns the indices of uncertain frames.
3. These frames are shown in the review GUI.
4. The researcher labels each as ``1`` (exploration) or ``0`` (not).
5. ``ActiveLearner.update`` accumulates all labels and re-fits the head.
6. Repeat until accuracy is satisfactory (or the pool is exhausted).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from explore.classification.clip_classifier import CLIPClassifier

logger = logging.getLogger(__name__)


class ActiveLearner:
    """Uncertainty-sampling active learner on top of :class:`CLIPClassifier`.

    Parameters
    ----------
    classifier:
        The CLIP classifier whose linear head will be updated.
    uncertainty_band:
        Probability interval ``(low, high)`` defining "uncertain" frames.
        Default ``(0.35, 0.65)``.
    max_query_size:
        Maximum number of uncertain frames to show per round (default 50).
    n_confident_samples:
        Number of high-confidence frames added automatically (without user
        input) to anchor the classifier (default 20 per class).
    """

    def __init__(
        self,
        classifier: CLIPClassifier,
        uncertainty_band: tuple[float, float] = (0.35, 0.65),
        max_query_size: int = 50,
        n_confident_samples: int = 20,
    ) -> None:
        self.classifier = classifier
        self.uncertainty_band = uncertainty_band
        self.max_query_size = max_query_size
        self.n_confident_samples = n_confident_samples

        # Accumulated labeled pool: index → label
        self._labeled: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        probas: np.ndarray,
        already_labeled: set[int] | None = None,
    ) -> np.ndarray:
        """Return frame indices to present to the user for labeling.

        Frames within ``uncertainty_band`` are returned first (most uncertain
        first), followed by a random sample from outside the band so the user
        can spot-check high-confidence predictions.

        Parameters
        ----------
        probas:
            Exploration probabilities from
            :meth:`~explore.classification.clip_classifier.CLIPClassifier.zero_shot_predict`
            or :meth:`~explore.classification.clip_classifier.CLIPClassifier.predict_proba`,
            shape ``(N,)``.
        already_labeled:
            Indices to exclude (already reviewed in a previous round).

        Returns
        -------
        np.ndarray
            Frame indices to show to the user.
        """
        skip = already_labeled or set()
        all_idx = np.arange(len(probas))

        lo, hi = self.uncertainty_band
        uncertain_mask = (probas >= lo) & (probas <= hi)
        uncertain_mask[list(skip)] = False

        uncertain_idx = all_idx[uncertain_mask]
        # Sort by distance to 0.5 — most uncertain first
        uncertain_idx = uncertain_idx[np.argsort(np.abs(probas[uncertain_idx] - 0.5))]
        uncertain_idx = uncertain_idx[: self.max_query_size]

        logger.info(
            "ActiveLearner.query: %d uncertain frames (band=[%.2f, %.2f]).",
            len(uncertain_idx),
            lo,
            hi,
        )
        return uncertain_idx

    def auto_label_confident(
        self,
        embeddings: np.ndarray,
        probas: np.ndarray,
    ) -> None:
        """Pseudo-label high-confidence frames to anchor the classifier.

        Automatically adds ``n_confident_samples`` frames from each end of the
        probability distribution (high confidence exploration / non-exploration)
        to the labeled pool without requiring user input.

        This gives the linear head a foundation before the first user-labeled
        correction round.
        """
        n = self.n_confident_samples
        sorted_idx = np.argsort(probas)

        neg_idx = sorted_idx[:n]  # lowest probabilities → not exploring
        pos_idx = sorted_idx[-n:]  # highest probabilities → exploring

        for i in neg_idx:
            self._labeled[int(i)] = 0
        for i in pos_idx:
            self._labeled[int(i)] = 1

        logger.info(
            "Auto-labeled %d confident frames (%d pos, %d neg).",
            len(neg_idx) + len(pos_idx),
            len(pos_idx),
            len(neg_idx),
        )
        self._refit(embeddings)

    def update(
        self,
        embeddings: np.ndarray,
        new_labels: dict[int, int],
    ) -> None:
        """Add user-provided labels and re-fit the classifier head.

        Parameters
        ----------
        embeddings:
            Full-video embeddings array, shape ``(N, D)``.
        new_labels:
            Mapping ``{frame_index: label}`` where ``label`` is ``1``
            (exploration) or ``0`` (not).
        """
        self._labeled.update(new_labels)
        logger.info(
            "ActiveLearner.update: %d new labels; total pool = %d.",
            len(new_labels),
            len(self._labeled),
        )
        self._refit(embeddings)

    @property
    def labeled_indices(self) -> list[int]:
        """Sorted list of all labeled frame indices."""
        return sorted(self._labeled.keys())

    @property
    def label_counts(self) -> dict[str, int]:
        """Counts of each class in the labeled pool."""
        labels = list(self._labeled.values())
        return {
            "exploration": sum(labels),
            "not_exploration": len(labels) - sum(labels),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refit(self, embeddings: np.ndarray) -> None:
        if not self._labeled:
            return
        idx = np.array(sorted(self._labeled.keys()))
        y = np.array([self._labeled[i] for i in idx])
        self.classifier.fit(embeddings[idx], y)
