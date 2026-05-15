"""CLIP-based behavioral classifier.

Three operational modes
-----------------------
1. **Zero-shot** — classify using text prompts only; no labeled frames needed.
2. **Few-shot** — fine-tune a logistic-regression head on labeled frames
   (typically 30–80 corrections are sufficient).
3. **Query** — return per-frame probabilities for the active-learning loop.

The CLIP backbone (ViT-B/32 by default) is frozen at all times; only the
lightweight logistic-regression head is fitted to the user's data.  This means:

* Training takes ~2 s on CPU regardless of video length.
* The behavioral definition is fully portable: the head weights are ~5 kB,
  while the backbone (150 MB) is shared across all experiments.
* Re-use a head from a similar experiment as a warm start.

Example
-------
>>> clf = CLIPClassifier()
>>> embeddings = clf.embed_frames(frames)          # (N, 512) array
>>> probas = clf.zero_shot_predict(               # zero-shot: no labels
...     embeddings,
...     pos_prompts=["mouse sniffing object"],
...     neg_prompts=["mouse walking away"],
... )
>>> clf.fit(embeddings[labeled_idx], labels)       # fine-tune on corrections
>>> probas = clf.predict_proba(embeddings)          # use fine-tuned head
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dimension of the CLIP ViT-B/32 image embedding
_EMBED_DIM = 512


class CLIPClassifier:
    """CLIP-based behavioral frame classifier.

    Parameters
    ----------
    model_name:
        OpenCLIP architecture string (default ``"ViT-B-32"``).
    pretrained:
        OpenCLIP pretrained weights (default ``"openai"``).
    device:
        Compute device.  Auto-selected when ``None``.
    batch_size:
        Frames to encode per forward pass (default 64).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = device or self._auto_device()

        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._head: Any = None  # sklearn LogisticRegression, fitted after corrections

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_frames(
        self,
        frames: list[np.ndarray],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode BGR video frames into CLIP image embeddings.

        Parameters
        ----------
        frames:
            BGR images (as from ``cv2.VideoCapture``).
        show_progress:
            Display a tqdm progress bar (default ``True``).

        Returns
        -------
        np.ndarray
            Shape ``(N, embed_dim)``, L2-normalised.
        """
        self._ensure_loaded()
        embeddings: list[np.ndarray] = []

        batches = [
            frames[i : i + self.batch_size]
            for i in range(0, len(frames), self.batch_size)
        ]

        for batch in tqdm(batches, desc="Encoding frames", disable=not show_progress):
            tensors = torch.stack(
                [self._preprocess(Image.fromarray(f[:, :, ::-1])) for f in batch]
            ).to(self.device)

            with torch.no_grad():
                emb = self._model.encode_image(tensors)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.cpu().float().numpy())

        return np.concatenate(embeddings, axis=0)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode text strings into CLIP text embeddings.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), embed_dim)``, L2-normalised.
        """
        self._ensure_loaded()
        tokens = self._tokenizer(texts).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def zero_shot_predict(
        self,
        embeddings: np.ndarray,
        pos_prompts: list[str],
        neg_prompts: list[str],
    ) -> np.ndarray:
        """Predict exploration probability using text prompts only.

        Each positive prompt is dot-producted with every frame embedding;
        the averaged positive score is then soft-maxed against the averaged
        negative score to produce a probability in ``[0, 1]``.

        Parameters
        ----------
        embeddings:
            Frame embeddings from :meth:`embed_frames`, shape ``(N, D)``.
        pos_prompts:
            Sentences describing exploration.
        neg_prompts:
            Sentences describing non-exploration.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — probability of exploration per frame.
        """
        self._ensure_loaded()
        pos_emb = self.embed_texts(pos_prompts)  # (P, D)
        neg_emb = self.embed_texts(neg_prompts)  # (Q, D)

        pos_scores = (embeddings @ pos_emb.T).mean(axis=1)  # (N,)
        neg_scores = (embeddings @ neg_emb.T).mean(axis=1)  # (N,)

        # Scale by CLIP's learned temperature so the softmax is decisive.
        # Without this, cosine similarities (~0.2–0.4) differ by only ~0.01
        # between classes, making the softmax output ≈ 0.5 for every frame.
        logit_scale = (
            float(self._model.logit_scale.exp().item())
            if hasattr(self._model, "logit_scale")
            else 100.0
        )
        pos_scores = pos_scores * logit_scale
        neg_scores = neg_scores * logit_scale

        stack = np.stack([pos_scores, neg_scores], axis=1)  # (N, 2)
        stack -= stack.max(axis=1, keepdims=True)  # numerical stability
        exp_stack = np.exp(stack)
        proba = exp_stack[:, 0] / exp_stack.sum(axis=1)  # (N,)
        return proba  # type: ignore[no-any-return]

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        c: float = 1.0,
    ) -> None:
        """Fit a logistic-regression head on labeled frame embeddings.

        Parameters
        ----------
        embeddings:
            Shape ``(N, D)`` — typically the subset returned by
            :class:`~explore.classification.active_learning.ActiveLearner`.
        labels:
            Binary array of shape ``(N,)``: ``1`` = exploration, ``0`` = not.
        c:
            Inverse regularisation strength (default 1.0).
        """
        from sklearn.linear_model import LogisticRegression

        if len(np.unique(labels)) < 2:
            raise ValueError("Need at least 2 distinct classes.")

        self._head = LogisticRegression(
            C=c,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=0,
        )
        self._head.fit(embeddings, labels)
        n_classes = len(np.unique(labels))
        logger.info("Head fitted on %d frames, %d classes.", len(labels), n_classes)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict exploration probability using the fitted linear head.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._head is None:
            raise RuntimeError(
                "No linear head fitted yet. "
                "Call fit() after collecting correction labels, "
                "or use zero_shot_predict() for label-free prediction."
            )
        return self._head.predict_proba(embeddings)[:, 1]  # type: ignore[no-any-return]

    def predict_class_indices(self, embeddings: np.ndarray) -> np.ndarray:
        """Return integer class predictions from the fitted head."""
        if self._head is None:
            raise RuntimeError("No head fitted yet. Call fit() first.")
        return self._head.predict(embeddings)  # type: ignore[no-any-return]

    def predict(self, embeddings: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary exploration prediction.

        Uses the fitted head if available, otherwise zero-shot with the last
        prompts passed to :meth:`zero_shot_predict`.
        """
        return (self.predict_proba(embeddings) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_head(self, path: Path | str) -> None:
        """Save only the fitted linear head (not the CLIP backbone).

        The backbone is ~150 MB; the head is ~5 kB and fully describes the
        experiment-specific behavioral definition.
        """
        if self._head is None:
            raise RuntimeError("No head to save — call fit() first.")
        joblib.dump(self._head, path)
        logger.info("Linear head saved to %s.", path)

    def load_head(self, path: Path | str) -> None:
        """Load a previously saved linear head."""
        self._head = joblib.load(path)
        logger.info("Linear head loaded from %s.", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is required. Install with: "
                "pip install open-clip-torch>=2.24"
            ) from exc

        logger.info(
            "Loading CLIP model '%s' ('%s') on %s …",
            self.model_name,
            self.pretrained,
            self.device,
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("CLIP backbone loaded.")

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
