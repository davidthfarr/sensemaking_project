# sensemaking

A temporal narrative monitoring framework for conflict information environments — grounded in situational awareness theory and designed to support analyst comprehension of large-scale, evolving social media discourse.

> Associated paper: *Emerging Narrative Detection in Dynamic Information Environments* (arXiv 2603.17617) and *Modeling Narrative Emergence in Dynamic Information Environments* (Submitted IC²S² 2026).

---

## Overview

Posts are embedded into a shared semantic state space, clustered via HDBSCAN within rolling time windows, and tracked across windows using centroid-based identity alignment. Each cluster is assigned a **stationary theme label** at birth (generated once by GPT and never updated). After clustering, each post is classified for **stance** (support / oppose / neutral) relative to its cluster's stationary theme using post-hoc GPT classification.

The pipeline supports multiple named datasets. Each case is fully self-contained under `data/processed/<case>/` and `data/evaluated/<case>/`.

---

## Architecture

```
data/processed/<case>/
  └── *_en_clean.parquet  or  posts_repr.parquet
          │
          ▼
    EmbeddingEncoder                   (sentence-transformers/all-mpnet-base-v2)
          │
          ▼  [per rolling window]
    HDBSCANClusterer                   (pure semantic embeddings)
          │
          ├──► align_clusters          (Hungarian matching, centroid cosine similarity)
          │         │
          │         ▼
          │    global_cluster_id       (stable identity across windows)
          │
          ├──► StationaryThemeLabeler  (GPT, assigned once at cluster birth)
          │         │
          │         ▼
          │    cluster_theme           (fixed claim-like label)
          │
          └──► PosthocGPTStanceClassifier  (GPT, per post vs. cluster theme)
                    │
                    ▼
              stance: support / oppose / neutral

data/evaluated/<case>/
  ├── results.parquet    (full output — see schema below)
  └── themes.json        (stationary theme store, persists across runs)
```

---

## Repository Structure

```
sensemaking_project/
├── sensemaking/
│   ├── data/
│   │   └── schemas.py              # Post dataclass
│   ├── embeddings/
│   │   └── encoder.py              # Sentence-level semantic embedding
│   ├── clustering/
│   │   ├── hdbscan.py              # Semantic-only HDBSCAN clustering
│   │   └── alignment.py            # Cross-window cluster identity alignment
│   ├── windows/
│   │   └── rolling.py              # Rolling time window utilities
│   ├── stance/
│   │   └── posthoc_gpt.py          # Post-hoc GPT stance classification
│   └── themes/
│       └── stationary_labeler.py   # Stationary theme generation + ThemeStore
├── scripts/
│   ├── run_pipeline.py             # ← Main entry point (multi-case, CLI)
│   ├── prepare_processed_data.py   # Raw CSV → clean parquet
│   └── build_representation.py     # One-off embedding computation
├── analysis/
│   ├── narrative_lifecycle.py      # Cluster birth/death timelines
│   ├── noise_over_time.py          # Noise fraction per window per case
│   ├── stance_distribution.py      # Stance breakdown per cluster
│   ├── centroid_drift.py           # Cluster centroid trajectories (PCA)
│   └── cross_case_overlay.py       # Side-by-side multi-case comparison
├── tests/
│   ├── test_alignment.py
│   ├── test_encoder.py
│   ├── test_hdbscan.py
│   ├── test_rolling.py
│   ├── test_posthoc_gpt.py
│   └── test_stationary_labeler.py
└── data/
    ├── processed/<case>/           # Input data (one directory per case)
    └── evaluated/<case>/           # Pipeline outputs (one directory per case)
```

---

## Installation

```bash
git clone https://github.com/davidthfarr/sensemaking_project.git
cd sensemaking_project
pip install -e .
```

A CUDA-capable GPU is strongly recommended for embedding at scale. The GPT steps require an OpenAI API key.

---

## Running the Pipeline

### 1. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

### 2. Prepare data for a case

Place a clean parquet file under `data/processed/<case_name>/`. The file must contain the columns:

| Column | Type | Description |
|--------|------|-------------|
| `post_id` | str | Unique post identifier |
| `text` | str | Post text content |
| `timestamp` | datetime (UTC) | Post publication time |
| `user_id` | str | Author identifier |

Naming conventions checked in order:
1. `data/processed/<case>/posts_repr.parquet` — pre-computed embeddings (fastest, skips encoder)
2. `data/processed/<case>/*_en_clean.parquet` — clean text; embeddings computed and cached
3. `data/processed/<case>/clean.parquet` — same as above

Use `scripts/prepare_processed_data.py` as a starting point for converting raw exports. Edit `RAW_PATH` and `OUT_PATH` in that script to match your data source.

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py --case <case_name>
```

**Examples:**

```bash
# Venezuela (hourly windows)
python scripts/run_pipeline.py --case venezuela --window-hours 12 --step-hours 4

# Ukraine (weekly windows)
python scripts/run_pipeline.py --case ukraine --window-hours 168 --step-hours 24

# New case on CPU with larger clusters
python scripts/run_pipeline.py --case conflict_ie --device cpu --min-cluster-size 15 --min-samples 3
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--case` | *(required)* | Case name — resolves to `data/processed/<case>/` |
| `--window-hours` | `12` | Rolling window size in hours |
| `--step-hours` | `4` | Step between windows in hours |
| `--min-cluster-size` | `8` | HDBSCAN minimum cluster size |
| `--min-samples` | `2` | HDBSCAN min samples |
| `--cluster-epsilon` | `0.0` | Raise to merge fragmented sub-clusters (0.1–0.5) |
| `--embed-model` | `all-mpnet-base-v2` | Sentence transformer model |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--gpt-model` | `gpt-4o-mini` | OpenAI model for theme generation and stance |
| `--n-representative` | `10` | Posts per cluster passed to GPT for theme generation |
| `--stance-batch-size` | `20` | Posts per GPT call during stance classification |
| `--align-threshold` | `0.5` | Minimum cosine similarity to accept cluster alignment |

### 4. Output schema

Results are written to `data/evaluated/<case>/results.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `post_id` | str | Post identifier |
| `text` | str | Post text |
| `timestamp` | datetime | Post publication time |
| `user_id` | str | Author identifier |
| `window_start` | datetime | Start of the window this row belongs to |
| `global_cluster_id` | int or None | Stable cross-window cluster identifier (None = noise) |
| `cluster_theme` | str or None | Stationary theme label for the cluster |
| `stance` | str or None | `support`, `oppose`, or `neutral` (None = noise) |
| `is_noise` | bool | True if the post was not assigned to any cluster |

The theme store is saved to `data/evaluated/<case>/themes.json`. Re-running the pipeline resumes from the existing store — already-labelled clusters are not re-sent to the API.

---

## Adding a New Case

1. Obtain data and export to CSV or parquet.
2. Run `scripts/prepare_processed_data.py` (edit `RAW_PATH` / `OUT_PATH`) to produce a clean parquet with the required columns. Place the output under `data/processed/<new_case>/`.
3. Set `OPENAI_API_KEY`.
4. Run: `python scripts/run_pipeline.py --case <new_case>`
5. Tune `--min-cluster-size`, `--window-hours`, and `--step-hours` to match the volume and temporal density of the dataset.

---

## Analysis

All analysis scripts read from `data/evaluated/<case>/results.parquet`.

```bash
# Narrative lifecycle — birth/death timelines per case
python analysis/narrative_lifecycle.py --cases venezuela ukraine --output lifecycle.png

# Noise fraction over time — all cases on one axis
python analysis/noise_over_time.py --cases venezuela ukraine --output noise.png

# Stance distribution per cluster (single case)
python analysis/stance_distribution.py --case venezuela --output stance.png

# Centroid drift — cluster trajectories in PCA space (single case)
python analysis/centroid_drift.py --case venezuela --output drift.png

# Cross-case overlay — lifecycle + noise panels combined
python analysis/cross_case_overlay.py --cases venezuela ukraine --output overlay.png
```

All scripts accept `--output <path>` to save to file, or omit it to display interactively.

---

## Methods

### Semantic Representation

Posts are encoded with [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) into a shared 768-dimensional embedding space held fixed across all windows to enable direct cross-temporal comparison. Representations are cached to `posts_repr.parquet` on first run.

### Clustering

HDBSCAN is applied to L2-normalised semantic embeddings within each rolling time window. Noise points (posts not assigned to any cluster) are preserved as a diagnostic signal — rising noise fractions can indicate narrative disruption or consolidation events.

### Narrative Identity Alignment

Cluster identity is tracked across adjacent windows by computing centroid cosine similarity and applying Hungarian matching. Clusters with similarity ≥ 0.5 (configurable via `--align-threshold`) are treated as persistent; others are treated as born or dissipated.

### Stationary Theme Generation

At cluster birth, a short claim-like theme label is generated by GPT from the posts closest to the cluster centroid. The label is **fixed for the lifetime of the cluster** and stored in `themes.json`. It is never updated, even as the cluster's centroid drifts.

### Post-hoc Stance Classification

After clustering and theme assignment, each non-noise post is classified for stance relative to its cluster's stationary theme using the OpenAI chat API (`gpt-4o-mini` by default). Labels are `support`, `oppose`, or `neutral`. Posts are batched per cluster to amortise API calls.

---

## Data

The paper applies this framework to two datasets:

- **Venezuela (Jan 2–7, 2026)** — 9,914 X posts from 3,799 users filtered by keywords `maduro` / `venezuela`, minimum 50 reposts
- **Ukraine** — English-language X posts on the Russia-Ukraine conflict

Raw data is not included due to platform terms of service. Processed parquet files (post text, timestamps, user IDs) should be placed under `data/processed/<case>/`.

---

## Tests

```bash
pytest tests/
```

CUDA-dependent tests are automatically skipped if no GPU is available. GPT-dependent tests use mocked API calls and do not require an API key.

---

## Citation

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
