# Changelog

All notable changes to EXPLORE will be documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [2.0.0] - 2026-05-15

Complete rewrite. The package is now a fully installable Python library
(`pip install explore`) with a browser-based GUI powered by NiceGUI.

### Added
- **Browser GUI** (`explore gui`) — six-tab NiceGUI app replacing the
  legacy tkinter scripts; session state saved automatically to `session.json`
- **CLIP classifier** — zero-shot scoring via OpenCLIP (ViT-B/32) plus an
  iterative logistic-regression head trained on user-corrected proximity labels
- **Balanced training** — `class_weight="balanced"` on the logistic-regression
  head to handle the natural skew toward non-exploration frames
- **ORB box localization** — bounding boxes drawn on a reference frame are
  automatically re-localized across all other videos using ORB feature matching
- **All-pair DI / RI** — Discrimination Index and Recognition Index computed
  for every pair of objects (`itertools.combinations`); columns named
  `DI_<A>_vs_<B>` / `RI_<A>_vs_<B>`
- **Prediction videos** — annotated MP4s with Low-res (12 fps, 0.5× scale,
  fast) and High-res (original fps, full resolution) toggle in the GUI
- **Trajectory plots** — thread-safe OOP matplotlib; grey path + plasma
  colour-coded scatter points; one plot per video saved to
  `results/tracking/<video>_trajectory.png`
- **Animal tracking** — centroid (x, y) per analysis frame saved to
  `results/tracking/<video>_tracking.csv`
- **Headless CLI** (`explore run CONFIG`) — run a full analysis from a YAML
  config without the GUI
- **Interactive config init** (`explore init`) — guided prompt to create a
  starter YAML
- **`pyproject.toml` packaging** — `setuptools-scm`, extras (`[dev]`,
  `[docs]`), ruff, mypy, pytest with coverage
- **Test suite** — 80 unit tests; all ML calls mocked; no GPU or model
  download required for CI
- **Assets** — logo bundled inside the package at `explore/assets/`

### Changed
- Package renamed from the legacy script collection to the installable
  `explore` library (`src/` layout)
- `AnalysisConfig` simplified — `familiar_object` / `novel_object` dropdowns
  removed; DI/RI is now computed automatically for all object pairs
- Output CSV uses tidy per-animal, per-minute rows

### Removed
- Legacy tkinter GUIs (`main_training.py`, `video_scroll.py`,
  `video_distribution.py`, tkinter box-selector)
- `setup.py` / `requirements.txt` in favour of `pyproject.toml`
- Community behavioral templates (prompts are entered directly in the GUI)

---

## [1.0.0] - 2023-03-01

Initial public release accompanying the *Scientific Reports* publication.

> Ibañez, V., Bohlen, L., Manuella, F. et al. EXPLORE: a novel deep learning-based analysis method for exploration behaviour in object recognition tests. Sci Rep 13, 4249 (2023). https://doi.org/10.1038/s41598-023-31094-w
