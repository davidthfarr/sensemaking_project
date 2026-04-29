"""
Assign stable global cluster IDs across all windows for a case.

Loads per-window clustering results, steps through adjacent window pairs using
align_clusters(), propagates cluster identities forward, and writes a single
consolidated parquet with global_cluster_id assigned to every post.

Usage
-----
python scripts/run_alignment.py --case iran
python scripts/run_alignment.py --case russia
python scripts/run_alignment.py --case venezuela

Input:
  data/evaluated/<case>/*.parquet          (per-window outputs from run_rolling_windows.py)
  data/processed/<case>/posts_repr.parquet (embeddings for centroid computation)

Output:
  data/evaluated/<case>/global_clusters.parquet
    post_id, window, local_cluster_id, global_cluster_id, is_noise
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sensemaking.clustering.alignment import align_clusters
from sensemaking.data.schemas import Post

SIMILARITY_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=["iran", "russia", "venezuela"])
    return p.parse_args()


def load_window_files(eval_dir: Path) -> list[Path]:
    """Return per-window parquets sorted by window timestamp, excluding output files."""
    files = [
        f for f in eval_dir.glob("*.parquet")
        if f.stem not in ("global_clusters", "results")
    ]
    # Filenames are %Y-%m-%d-%H — lexicographic sort is chronological
    return sorted(files)


def build_posts(window_df: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> list[Post]:
    """Reconstruct Post objects with embeddings for a single window."""
    posts = []
    for row in window_df.itertuples(index=False):
        emb = embeddings.get(str(row.post_id))
        p = Post(
            post_id=str(row.post_id),
            text="",
            timestamp=None,
            embedding=emb,
        )
        p.cluster_id = None if row.is_noise else row.cluster_id
        p.is_noise = bool(row.is_noise)
        posts.append(p)
    return posts


def main() -> None:
    args = parse_args()

    eval_dir  = Path("data/evaluated") / args.case
    repr_path = Path("data/processed") / args.case / "posts_repr.parquet"
    out_path  = eval_dir / "global_clusters.parquet"

    # Load embeddings once, keyed by post_id
    print(f"Loading embeddings from {repr_path}")
    repr_df = pd.read_parquet(repr_path, columns=["post_id", "embedding"])
    embeddings = {str(row.post_id): row.embedding for row in repr_df.itertuples(index=False)}
    print(f"  {len(embeddings):,} embeddings loaded")

    window_files = load_window_files(eval_dir)
    if not window_files:
        raise FileNotFoundError(f"No window parquets found in {eval_dir}")
    print(f"Processing {len(window_files)} windows for case '{args.case}'")

    next_global_id = 0
    prev_posts: list[Post] = []
    prev_local_to_global: dict[int, int] = {}
    all_rows: list[dict] = []

    for wf in window_files:
        window_name = wf.stem
        window_df = pd.read_parquet(wf)

        curr_posts = build_posts(window_df, embeddings)

        # Align current window against previous
        if prev_posts:
            # align_clusters returns {prev_local_id: curr_local_id}
            alignment = align_clusters(prev_posts, curr_posts, SIMILARITY_THRESHOLD)
            curr_to_prev = {curr: prev for prev, curr in alignment.items()}
        else:
            curr_to_prev = {}

        # Propagate or mint global IDs
        local_to_global: dict[int, int] = {}
        for p in curr_posts:
            if p.cluster_id is None or p.cluster_id in local_to_global:
                continue
            prev_local = curr_to_prev.get(p.cluster_id)
            if prev_local is not None and prev_local in prev_local_to_global:
                local_to_global[p.cluster_id] = prev_local_to_global[prev_local]
            else:
                local_to_global[p.cluster_id] = next_global_id
                next_global_id += 1

        n_new = sum(
            1 for lid, gid in local_to_global.items()
            if gid >= (next_global_id - len(local_to_global))
        )
        n_linked = len(local_to_global) - n_new
        noise = sum(p.is_noise for p in curr_posts)
        print(
            f"  {window_name} | posts={len(curr_posts):4d} | "
            f"clusters={len(local_to_global):2d} | "
            f"linked={n_linked} new={n_new} noise={noise}"
        )

        for p in curr_posts:
            global_id = local_to_global.get(p.cluster_id) if p.cluster_id is not None else None
            all_rows.append({
                "post_id":           p.post_id,
                "window":            window_name,
                "local_cluster_id":  p.cluster_id,
                "global_cluster_id": global_id,
                "is_noise":          p.is_noise,
            })

        prev_posts = curr_posts
        prev_local_to_global = local_to_global

    # Write output
    df_out = pd.DataFrame(all_rows)
    df_out.to_parquet(out_path, index=False)
    print(f"\nWritten → {out_path} ({len(df_out):,} rows)")

    # Summary
    total_global = next_global_id
    print(f"\nTotal global clusters: {total_global}")

    active = df_out[~df_out["is_noise"] & df_out["global_cluster_id"].notna()]
    lifespan = (
        active.groupby("global_cluster_id")["window"]
        .nunique()
        .rename("windows_alive")
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    lifespan["global_cluster_id"] = lifespan["global_cluster_id"].astype(int)

    print("\nTop 10 clusters by lifespan (windows alive):")
    print(f"  {'global_id':>10}  {'windows_alive':>13}")
    print(f"  {'-'*10}  {'-'*13}")
    for row in lifespan.itertuples(index=False):
        print(f"  {int(row.global_cluster_id):>10}  {int(row.windows_alive):>13}")


if __name__ == "__main__":
    main()
