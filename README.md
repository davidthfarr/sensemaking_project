# sensemaking

A systems-oriented framework for modeling emerging narratives in dynamic information environments — grounded in situational awareness theory and designed to support analyst comprehension of large-scale, evolving social media discourse.

> Associated paper: *Emerging Narrative Detection in Dynamic Information Environments* (arXiv) and *Modeling Narrative Emergence in Dynamic Information Environments* (Submitted IC²S² 2026).
> Interactive dashboard: [davidthfarr.github.io/sensemaking](https://davidthfarr.github.io/sensemaking/)

---

## Overview

Rather than relying on predefined labels or static topic categories, this system represents discourse as **temporally evolving semantic structures** that can emerge, persist, drift, and dissipate over time. Posts are embedded into a shared semantic state space, clustered via HDBSCAN, and tracked across rolling time windows using centroid-based identity alignment.

The pipeline supports three levels of analyst output:
- **Structural monitoring** — track narrative emergence, persistence, and disruption over time
- **Confidence signals** — centroid proximity scores to triage high-signal content
- **Narrative synthesis** — LLM-generated summaries of representative cluster content

---

## Architecture

```
External Event
    │
    ▼
Data Ingestion / Embeddings
    │
    ▼
Temporal Clustering (HDBSCAN)
    │
    ├──► Structural Monitoring
    │
    ├──► Representative Content Selection
    │         │
    │         ▼
    │    Centroid-Based Confidence Signal
    │
    └──► Narrative Synthesis ──► Analyst Review
                                      │
                                      └──► Parameter Tuning (feedback loop)
```

---

## Repository Structure

```
sensemaking/
├── sensemaking/
│   ├── data/
│   │   └── schemas.py          # Post dataclass definition
│   ├── embeddings/
│   │   ├── encoder.py          # Sentence-level semantic embedding (all-mpnet-base-v2)
│   │   └── stance.py           # Zero-shot stance labeling (bart-large-mnli)
│   └── clustering/
│       ├── hdbscan.py          # Joint semantic + stance clustering
│       └── alignment.py        # Cross-window cluster identity alignment
├── scripts/
│   ├── build_representation.py # Encode + stance-label posts and persist to parquet
│   └── run_single_window.py    # Run full pipeline on a single time window
├── tests/
│   ├── test_alignment.py
│   └── test_stance.py
└── data/
    ├── processed/              # Cleaned input parquets
    └── evaluated/              # Pipeline outputs
```

---

## Installation

```bash
git clone https://github.com/davidthfarr/sensemaking_project.git
cd sensemaking_project
pip install -e .
```

**Dependencies** include `sentence-transformers`, `hdbscan`, `torch`, `transformers`, `pandas`, and `scikit-learn`. A CUDA-capable GPU is strongly recommended for embedding and stance inference at scale.

---

## Quickstart

### 1. Build post representations (embeddings + stance)

```python
from sensemaking.embeddings.encoder import EmbeddingEncoder
from sensemaking.embeddings.stance import ZeroShotStanceLabeler
from sensemaking.data.schemas import Post

encoder = EmbeddingEncoder(require_cuda=True)
stance  = ZeroShotStanceLabeler(require_cuda=True)

posts = encoder(posts)
posts = stance(posts)
```

Or run the build script directly:

```bash
python scripts/build_representation.py
```

### 2. Cluster a single time window

```bash
python scripts/run_single_window.py
```

Key parameters (edit in the script):

| Parameter | Default | Description |
|---|---|---|
| `WINDOW_DAYS` | `7` | Rolling window size in days |
| `STANCE_WEIGHT` | `0.1` | Relative weight of stance in joint embedding |
| `MIN_CLUSTER_SIZE` | `20` | HDBSCAN minimum cluster size |
| `MIN_SAMPLES` | `5` | HDBSCAN min samples |
| `EMBED_MODEL` | `all-mpnet-base-v2` | Sentence transformer model |
| `STANCE_MODEL` | `facebook/bart-large-mnli` | NLI model for stance |

---

## Methods

### Semantic Representation

Posts are encoded with [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) into a shared 768-dimensional embedding space. This space is held fixed across all time windows to enable direct cross-temporal comparison.

### Stance Labeling

Zero-shot stance classification is performed via [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli). Labels are encoded as `+1` (aligned), `-1` (opposed), or `0` (neutral) and appended as a weighted dimension to the semantic embedding before clustering.

### Clustering

HDBSCAN is applied to joint semantic + stance vectors within each rolling time window. Noise points are preserved as a diagnostic signal — rising noise fractions can indicate narrative disruption or consolidation events.

### Narrative Identity Alignment

Cluster identity is tracked across adjacent windows by computing centroid cosine similarity. Clusters with similarity ≥ 0.85 are treated as persistent narratives; clusters below threshold are treated as dissipated, and novel coherent groupings as emergent.

---

## Data

The paper applies this framework to two datasets:

- **Venezuela (Jan 2–7, 2026)** — 9,914 X posts from 3,799 users filtered by keywords `maduro` / `venezuela`, minimum 50 reposts
- **Ukraine** — English-language X posts on the Russia-Ukraine conflict

Raw data is not included in this repository due to platform terms of service. Processed parquet files (post text, timestamps, user IDs) should be placed under `data/processed/`.

---

## Tests

```bash
pytest tests/
```

Tests cover cluster alignment logic and stance model output ranges. CUDA-dependent tests are automatically skipped if no GPU is available.

---

## Citation

If you use this code or framework, please cite:

```bibtex
@misc{farr2026temporalnarrativemonitoringdynamic,
      title={Temporal Narrative Monitoring in Dynamic Information Environments}, 
      author={David Farr and Stephen Prochaska and Jack Moody and Lynnette Hui Xian Ng and Iain Cruickshank and Kate Starbird and Jevin West},
      year={2026},
      eprint={2603.17617},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2603.17617}, 
}
```

---

## Acknowledgments

Generative AI tools (GPT and Claude) were used to assist with software development and drafting. All code, analyses, and outputs were reviewed and verified by the authors.
