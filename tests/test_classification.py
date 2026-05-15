"""Tests for CLIPClassifier and ActiveLearner.

All tests use pre-computed synthetic embeddings; no torch or open_clip
import is triggered.
"""

from __future__ import annotations

import numpy as np
import pytest

from explore.classification.active_learning import ActiveLearner
from explore.classification.clip_classifier import CLIPClassifier

# ---------------------------------------------------------------------------
# CLIPClassifier — zero-shot
# ---------------------------------------------------------------------------


def test_zero_shot_predict_shape(mock_clip_classifier, random_embeddings):
    probas = mock_clip_classifier.zero_shot_predict(
        random_embeddings,
        pos_prompts=["mouse sniffing"],
        neg_prompts=["mouse walking"],
    )
    assert probas.shape == (len(random_embeddings),)


def test_zero_shot_predict_range(mock_clip_classifier, random_embeddings):
    probas = mock_clip_classifier.zero_shot_predict(
        random_embeddings,
        pos_prompts=["mouse sniffing"],
        neg_prompts=["mouse walking"],
    )
    assert np.all(probas >= 0)
    assert np.all(probas <= 1)


def test_zero_shot_predict_sums_to_one_with_complement(
    mock_clip_classifier, random_embeddings
):
    """pos_score + neg_score softmax should sum to 1 per frame."""

    # Use the same text for both so scores are equal → proba ≈ 0.5
    def _embed_texts_equal(texts):
        # Return identical embedding regardless of text
        e = np.ones((len(texts), 512), dtype=np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return e

    mock_clip_classifier.embed_texts.side_effect = _embed_texts_equal
    probas = mock_clip_classifier.zero_shot_predict(
        random_embeddings,
        pos_prompts=["x"],
        neg_prompts=["x"],
    )
    np.testing.assert_allclose(probas, 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# CLIPClassifier — fit / predict
# ---------------------------------------------------------------------------


def test_fit_and_predict_proba(mock_clip_classifier, random_embeddings, binary_labels):
    # Install the real fit method (not mocked)
    mock_clip_classifier.fit = CLIPClassifier.fit.__get__(mock_clip_classifier)
    mock_clip_classifier.predict_proba = CLIPClassifier.predict_proba.__get__(
        mock_clip_classifier
    )

    mock_clip_classifier.fit(random_embeddings, binary_labels)
    probas = mock_clip_classifier.predict_proba(random_embeddings)

    assert probas.shape == (len(random_embeddings),)
    assert np.all(probas >= 0)
    assert np.all(probas <= 1)


def test_predict_proba_raises_without_head(random_embeddings):
    clf = CLIPClassifier.__new__(CLIPClassifier)
    clf._head = None
    clf.predict_proba = CLIPClassifier.predict_proba.__get__(clf)

    with pytest.raises(RuntimeError, match="No linear head"):
        clf.predict_proba(random_embeddings)


def test_fit_single_class_raises(mock_clip_classifier, random_embeddings):
    mock_clip_classifier.fit = CLIPClassifier.fit.__get__(mock_clip_classifier)
    all_ones = np.ones(len(random_embeddings), dtype=np.int32)

    with pytest.raises(ValueError, match="at least 2 distinct"):
        mock_clip_classifier.fit(random_embeddings, all_ones)


# ---------------------------------------------------------------------------
# CLIPClassifier — persistence
# ---------------------------------------------------------------------------


def test_save_load_head(
    tmp_path, mock_clip_classifier, random_embeddings, binary_labels
):
    mock_clip_classifier.fit = CLIPClassifier.fit.__get__(mock_clip_classifier)
    mock_clip_classifier.save_head = CLIPClassifier.save_head.__get__(
        mock_clip_classifier
    )
    mock_clip_classifier.load_head = CLIPClassifier.load_head.__get__(
        mock_clip_classifier
    )
    mock_clip_classifier.predict_proba = CLIPClassifier.predict_proba.__get__(
        mock_clip_classifier
    )

    mock_clip_classifier.fit(random_embeddings, binary_labels)
    head_path = tmp_path / "head.pkl"
    mock_clip_classifier.save_head(head_path)

    # Clear and reload
    mock_clip_classifier._head = None
    mock_clip_classifier.load_head(head_path)

    probas = mock_clip_classifier.predict_proba(random_embeddings)
    assert probas.shape == (len(random_embeddings),)


def test_save_head_without_fit_raises(mock_clip_classifier, tmp_path):
    mock_clip_classifier.save_head = CLIPClassifier.save_head.__get__(
        mock_clip_classifier
    )
    with pytest.raises(RuntimeError, match="No head"):
        mock_clip_classifier.save_head(tmp_path / "head.pkl")


# ---------------------------------------------------------------------------
# ActiveLearner
# ---------------------------------------------------------------------------


def test_query_returns_uncertain_frames(mock_clip_classifier, random_embeddings):
    learner = ActiveLearner(mock_clip_classifier, uncertainty_band=(0.4, 0.6))
    # All probabilities in uncertain band
    probas = np.full(len(random_embeddings), 0.5)
    idx = learner.query(probas)
    assert len(idx) > 0
    assert len(idx) <= learner.max_query_size


def test_query_excludes_already_labeled(mock_clip_classifier, random_embeddings):
    learner = ActiveLearner(mock_clip_classifier, uncertainty_band=(0.0, 1.0))
    probas = np.full(len(random_embeddings), 0.5)
    already = set(range(10))
    idx = learner.query(probas, already_labeled=already)
    assert not set(idx).intersection(already)


def test_query_respects_max_size(mock_clip_classifier, random_embeddings):
    learner = ActiveLearner(
        mock_clip_classifier, uncertainty_band=(0.0, 1.0), max_query_size=5
    )
    probas = np.full(len(random_embeddings), 0.5)
    idx = learner.query(probas)
    assert len(idx) <= 5


def test_update_refits_head(mock_clip_classifier, random_embeddings, binary_labels):
    mock_clip_classifier.fit = CLIPClassifier.fit.__get__(mock_clip_classifier)

    learner = ActiveLearner(mock_clip_classifier)
    corrections = {i: int(binary_labels[i]) for i in range(len(binary_labels))}
    learner.update(random_embeddings, corrections)

    assert mock_clip_classifier._head is not None
    assert learner.label_counts["exploration"] == 10
    assert learner.label_counts["not_exploration"] == 10


def test_auto_label_confident(mock_clip_classifier, random_embeddings):
    mock_clip_classifier.fit = CLIPClassifier.fit.__get__(mock_clip_classifier)

    learner = ActiveLearner(mock_clip_classifier, n_confident_samples=3)
    probas = np.linspace(0, 1, len(random_embeddings))
    learner.auto_label_confident(random_embeddings, probas)

    assert len(learner.labeled_indices) == 6  # 3 pos + 3 neg
    assert mock_clip_classifier._head is not None


def test_labeled_indices_sorted(mock_clip_classifier, random_embeddings):
    mock_clip_classifier.fit = CLIPClassifier.fit.__get__(mock_clip_classifier)
    learner = ActiveLearner(mock_clip_classifier)
    learner.update(random_embeddings, {5: 1, 2: 0, 8: 1})
    assert learner.labeled_indices == [2, 5, 8]
