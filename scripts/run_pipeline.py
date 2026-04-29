#!/usr/bin/env python
"""
Unified case-aware pipeline: embed → cluster → align → theme → stance → results.

Each named case is self-contained under data/processed/<case>/ and
data/evaluated/<case>/. Running this script for a new case requires only
placing clean data in the expected location and setting OPENAI_API_KEY.

Usage
-----
python scripts/run_pipeline.py --case venezuela
python scripts/run_pipeline.py --case ukraine --window-hours 168 --step-hours 24
python scripts/run_pipeline.py --case conflict_ie --device cpu --gpt-model gpt-4o

Input (checked in order)
------------------------
1. data/processed/<case>/posts_repr.parquet  — pre-computed embeddings (fastest)
2. data/processed/<case>/*_en_clean.parquet  — clean text, embeddings computed and cached
3. data/processed/<case>/clean.parquet       — same as above, standard name

Output
------
data/evaluated/<case>/results.parquet
    post_id, text, timestamp, user_id, window_start,
    global_cluster_id, cluster_theme, stance, is_noise

data/evaluated/<case>/themes.json
    Stationary theme store (global_cluster_id → theme string).
    Persists across runs so re-running the pipeline does not re-label clusters.
"""

import argparse
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

from sensemaking.clustering.alignment import align_clusters
from sensemaking.clustering.hdbscan import HDBSCANClusterer
from sensemaking.data.schemas import Post
from sensemaking.embeddings.encoder import EmbeddingEncoder
from sensemaking.stance.posthoc_gpt import PosthocGPTStanceClassifier
from sensemaking.themes.stationary_labeler import StationaryThemeLabeler, ThemeStore
from sensemaking.windows.rolling import generate_rolling_windows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the sensemaking pipeline for a named case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", required=True,
                   help="Case name, e.g. venezuela, ukraine, conflict_ie")
    p.add_argument("--window-hours", type=int, default=12,
                   help="Rolling window size in hours")
    p.add_argument("--step-hours", type=int, default=4,
                   help="Step size between windows in hours")
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--cluster-epsilon", type=float, default=0.0,
                   help="HDBSCAN cluster_selection_epsilon (raise to merge fragments)")
    p.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--device", default="cuda",
                   help="Compute device for embedding encoder (cuda or cpu)")
    p.add_argument("--gpt-model", default="gpt-4o-mini",
                   help="OpenAI model for theme generation and stance classification")
    p.add_argument("--n-representative", type=int, default=10,
                   help="Posts per cluster passed to GPT for theme generation")
    p.add_argument("--stance-batch-size", type=int, default=20,
                   help="Posts per GPT call for stance classification")
    p.add_argument("--align-threshold", type=float, default=0.5,
                   help="Minimum cosine similarity to accept a cluster alignment")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_posts(case_dir: Path, embed_model: str, device: str) -> list[Post]:
    """
    Load posts for a case. Uses pre-computed representations if available;
    otherwise computes and caches embeddings.
    """
    repr_path = case_dir / "posts_repr.parquet"

    if repr_path.exists():
        print(f"Loading pre-computed representations from {repr_path}")
        df = pd.read_parquet(repr_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return [
            Post(
                post_id=str(row.post_id),
                text=row.text,
                timestamp=row.timestamp,
                user_id=str(getattr(row, "user_id", "")) or None,
                embedding=row.embedding,
            )
            for row in df.itertuples(index=False)
        ]

    # Fall back to clean text parquet
    candidates = (
        list(case_dir.glob("*_en_clean.parquet")) +
        list(case_dir.glob("clean.parquet"))
    )
    if not candidates:
        sys.exit(
            f"No input data found in {case_dir}.\n"
            "Expected one of:\n"
            "  posts_repr.parquet   (pre-computed embeddings)\n"
            "  *_en_clean.parquet   (clean text, embeddings will be computed)\n"
            "  clean.parquet        (same)"
        )

    clean_path = candidates[0]
    print(f"Loading clean data from {clean_path}")
    df = pd.read_parquet(clean_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    posts = [
        Post(
            post_id=str(row.post_id),
            text=row.text,
            timestamp=row.timestamp,
            user_id=str(getattr(row, "user_id", "")) or None,
        )
        for row in df.itertuples(index=False)
    ]

    print(f"Computing embeddings for {len(posts):,} posts...")
    encoder = EmbeddingEncoder(model_name=embed_model, device=device, batch_size=64)
    posts = encoder(posts)

    # Cache to disk
    print(f"Caching representations to {repr_path}")
    pd.DataFrame({
        "post_id":   [p.post_id for p in posts],
        "user_id":   [p.user_id for p in posts],
        "timestamp": [p.timestamp for p in posts],
        "text":      [p.text for p in posts],
        "embedding": [p.embedding for p in posts],
    }).to_parquet(repr_path, index=False)

    return posts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    case_dir = Path("data/processed") / args.case
    eval_dir = Path("data/evaluated") / args.case
    eval_dir.mkdir(parents=True, exist_ok=True)

    theme_path   = eval_dir / "themes.json"
    results_path = eval_dir / "results.parquet"

    # ── Load posts ────────────────────────────────────────────────────────────
    posts = load_posts(case_dir, args.embed_model, args.device)
    print(f"Loaded {len(posts):,} posts spanning "
          f"{posts[0].timestamp.date()} – {posts[-1].timestamp.date()}")

    # ── Initialise components ─────────────────────────────────────────────────
    clusterer = HDBSCANClusterer(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_epsilon,
    )
    theme_labeler = StationaryThemeLabeler(
        model=args.gpt_model,
        n_representative=args.n_representative,
    )
    stance_classifier = PosthocGPTStanceClassifier(
        model=args.gpt_model,
        batch_size=args.stance_batch_size,
    )

    # Resume existing theme store so re-runs don't re-label known clusters
    store = ThemeStore.load(theme_path) if theme_path.exists() else ThemeStore()
    next_global_id = (max(store.to_dict().keys(), default=-1) + 1)

    # ── Rolling window loop ───────────────────────────────────────────────────
    prev_posts: list[Post] = []
    prev_local_to_global: dict[int, int] = {}
    all_rows: list[dict] = []

    windows = list(generate_rolling_windows(
        posts,
        window_size=timedelta(hours=args.window_hours),
        step_size=timedelta(hours=args.step_hours),
    ))
    print(f"Processing {len(windows)} windows "
          f"({args.window_hours}h window, {args.step_hours}h step)")

    for (win_start, win_end), win_posts in windows:
        if not win_posts:
            continue

        # Cluster
        win_posts = clusterer.fit_predict(win_posts)
        n_clusters = len({p.cluster_id for p in win_posts if p.cluster_id is not None})
        noise_frac = sum(p.is_noise for p in win_posts) / len(win_posts)
        print(f"  {win_start.strftime('%Y-%m-%d %H:%M')} | "
              f"posts={len(win_posts):4d} | clusters={n_clusters:2d} | noise={noise_frac:.2f}")

        # Align local cluster IDs → global IDs
        # align_clusters returns {prev_local_id: curr_local_id}
        alignment = (
            align_clusters(prev_posts, win_posts, args.align_threshold)
            if prev_posts else {}
        )
        curr_to_prev = {curr: prev for prev, curr in alignment.items()}

        local_to_global: dict[int, int] = {}
        for p in win_posts:
            if p.cluster_id is None or p.cluster_id in local_to_global:
                continue
            prev_local = curr_to_prev.get(p.cluster_id)
            if prev_local is not None and prev_local in prev_local_to_global:
                local_to_global[p.cluster_id] = prev_local_to_global[prev_local]
            else:
                local_to_global[p.cluster_id] = next_global_id
                next_global_id += 1

        # Assign stationary themes for newly-born clusters
        store = theme_labeler.assign_new_themes(win_posts, local_to_global, store)

        # Stance classification — one GPT call per cluster (grouped by theme)
        cluster_groups: dict[int, list[Post]] = defaultdict(list)
        for p in win_posts:
            if p.cluster_id is not None:
                cluster_groups[local_to_global[p.cluster_id]].append(p)

        for global_id, cluster_posts in cluster_groups.items():
            theme = store.get(global_id)
            if theme:
                stance_classifier.classify_posts(cluster_posts, theme)

        # Collect output rows
        for p in win_posts:
            global_id = (
                local_to_global.get(p.cluster_id)
                if p.cluster_id is not None else None
            )
            all_rows.append({
                "post_id":          p.post_id,
                "text":             p.text,
                "timestamp":        p.timestamp,
                "user_id":          p.user_id,
                "window_start":     win_start,
                "global_cluster_id": global_id,
                "cluster_theme":    store.get(global_id) if global_id is not None else None,
                "stance":           p.stance,
                "is_noise":         p.is_noise,
            })

        prev_posts = win_posts
        prev_local_to_global = local_to_global

    # ── Save outputs ──────────────────────────────────────────────────────────
    store.save(theme_path)
    print(f"\nTheme store saved → {theme_path} ({len(store)} themes)")

    df_results = pd.DataFrame(all_rows)
    df_results.to_parquet(results_path, index=False)
    print(f"Results saved → {results_path} ({len(df_results):,} rows)")
    print(df_results.head())


if __name__ == "__main__":
    run(parse_args())
