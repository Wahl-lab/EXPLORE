# EXPLORE

**Automated exploration behavior analysis for object recognition tests**  
using CLIP and Grounding DINO.

---

## What is EXPLORE?

EXPLORE is an open-source pipeline for automated behavioral scoring in
**Novel Object Recognition (NOR)** and related paradigms.  It replaces the
manual frame-labeling workflow of the original EXPLORE with a text-first
interface: researchers describe their objects and behavioral definition in plain
language, and the system handles the rest.

### Key features

| Feature | Description |
|---|---|
| **Text-defined behavior** | Write sentences describing exploration; CLIP classifies every frame zero-shot — no labeled frames required for standard setups |
| **Auto object detection** | Describe objects in natural language; Grounding DINO localizes them in the reference frame automatically |
| **Active learning** | Only uncertain frames (≈ 20–50) are presented for correction; fitting takes seconds on CPU |
| **Standard metrics** | Exploration time, frequency, Discrimination Index and Recognition Index per animal per time bin |
| **Community templates** | Pre-validated prompt sets for standard NOR, OLM, social recognition and more |
| **Reproducibility** | The behavioral definition (text prompts) is saved with the data and quotable in Methods |

---

## Installation

```bash
# CPU-only (recommended for most lab workstations)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install explore

# GPU (CUDA 12)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install explore
```

---

## Quick start

### GUI (recommended for new users)

```bash
explore gui
```

### Command line

```bash
# 1. Create a config interactively
explore init

# 2. Auto-detect object bounding boxes
explore detect experiment.yaml

# 3. Run analysis
explore run experiment.yaml
```

### Python API

```python
from explore import ExperimentConfig, ExplorationPipeline

cfg = ExperimentConfig.from_yaml("experiment.yaml")
pipeline = ExplorationPipeline(cfg, headless=True)
results = pipeline.run()
results.to_csv("results.csv", index=False)
```

---

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

tutorials/quickstart
tutorials/behavioral_definition
tutorials/active_learning
tutorials/cli_reference
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/config
api/detection
api/classification
api/pipeline
api/utils
```

---

## Behavioral definition as methods text

Because behavioral definitions are stored as text, they can be quoted directly
in the manuscript:

> *Exploration was operationally defined using the following text prompts
> supplied to CLIP (ViT-B/32, OpenAI weights):  
> Positive: "a mouse actively sniffing and investigating an object with its
> nose close to the object surface."  
> Negative: "a mouse walking past or resting away from objects."
> (EXPLORE v2.0.0, config hash: `a3f7b2c`)*

---

## Citing

If you use EXPLORE, please cite:

> Ibañez V. *et al.* (2024). EXPLORE: Text-guided behavioral analysis
> for object recognition tests using vision-language models.
> *Under review.*

---

## Contact

- **Victor Ibañez** — victor.ibanez@uzh.ch  
- **Caroline Wahl** — wahl@hifo.uzh.ch  
- **Issues / PRs** — https://github.com/victorjonathanibanez/EXPLORE/issues
