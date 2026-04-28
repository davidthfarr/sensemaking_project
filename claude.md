# CLAUDE.md — Project Briefing for Claude Code

## What this repo is
A temporal narrative monitoring framework applied to conflict information environments.
Paper: arXiv 2603.17617. Current pipeline: embed → HDBSCAN cluster → centroid alignment.

## What needs to change (new project: conflict IE)

### 0. Remove html files or other miscellaneous notebooks not needed.

### 1. Remove inline stance from clustering
- `sensemaking/embeddings/stance.py` currently uses `facebook/bart-large-mnli` and appends 
  stance as a weighted dimension before clustering. REMOVE this from the clustering step.
- `sensemaking/clustering/hdbscan.py` uses a joint semantic+stance vector. Revert to 
  pure semantic embeddings only.

### 2. Add post-hoc GPT stance classification
- After clustering and temporal linkage are complete, classify posts using the OpenAI API 
  against the stationary cluster theme/label.
- New module: `sensemaking/stance/posthoc_gpt.py`
- Stance labels: support / oppose / neutral toward the cluster's stationary theme.

### 3. Add stationary theme generation
- Each global cluster gets a theme label assigned once (at cluster birth), fixed thereafter.
- New module: `sensemaking/themes/stationary_labeler.py`
- Use GPT to generate a short claim-like summary from representative posts at cluster birth.

### 4. Multi-case support
- Pipeline must support multiple named datasets (Venezuela, Ukraine, + new case).
- Data lives in `data/processed/<case_name>/`
- Each evaluated dataset should have results dataset that includes post, timestamp, cluster belongs to, cluster name, stance of post in relation to cluster theme.

### 5. Comparative plots
- Build scripts/notebooks in `analysis/` for cross-case comparison:
  - Narrative lifecycle plots (birth/death timelines) per case
  - Noise fraction over time per case
  - Stance distribution per cluster per case
  - Centroid drift over time for top narratives
  - Cross-case overlay plots for comparison
