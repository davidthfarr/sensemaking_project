"""
Generate stationary theme labels for all global clusters in a case.

Reads global_clusters.parquet, finds the 5 posts closest to each cluster
centroid, calls GPT (gpt-4o-mini) to produce a ≤10-word claim-like label,
and writes cluster_themes.parquet.

Usage
-----
python scripts/run_theme_labeling.py --case iran
python scripts/run_theme_labeling.py --case russia
python scripts/run_theme_labeling.py --case venezuela

Input:
  data/evaluated/<case>/global_clusters.parquet
  data/processed/<case>/posts_repr.parquet

Output:
  data/evaluated/<case>/cluster_themes.parquet
    global_cluster_id, theme, representative_posts (JSON list of post_ids)

Requires OPENAI_API_KEY in .env or the environment.
Already-labeled clusters are skipped (resume-safe).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sensemaking.data.schemas import Post
from sensemaking.themes.stationary_labeler import StationaryThemeLabeler

SLEEP_BETWEEN_CALLS = 0.5  # seconds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=["iran", "russia", "venezuela"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS,
                   help="Seconds to sleep between API calls")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    eval_dir  = Path("data/evaluated") / args.case
    repr_path = Path("data/processed") / args.case / "posts_repr.parquet"
    gc_path   = eval_dir / "global_clusters.parquet"
    out_path  = eval_dir / "cluster_themes.parquet"

    # Load embeddings + text keyed by post_id
    print(f"Loading embeddings from {repr_path}")
    repr_df = pd.read_parquet(repr_path, columns=["post_id", "text", "embedding"])
    repr_df["post_id"] = repr_df["post_id"].astype(str)
    id_to_text = dict(zip(repr_df["post_id"], repr_df["text"]))
    id_to_emb  = dict(zip(repr_df["post_id"], repr_df["embedding"]))
    print(f"  {len(repr_df):,} posts with embeddings")

    # Load global cluster assignments (take the birth window rows only)
    print(f"Loading global clusters from {gc_path}")
    gc_df = pd.read_parquet(gc_path)
    gc_df["post_id"] = gc_df["post_id"].astype(str)
    gc_df = gc_df[~gc_df["is_noise"] & gc_df["global_cluster_id"].notna()]
    gc_df["global_cluster_id"] = gc_df["global_cluster_id"].astype(int)

    cluster_ids = sorted(gc_df["global_cluster_id"].unique())
    print(f"  {len(cluster_ids)} non-noise clusters to label")

    # Load existing output for resume support
    already_done: set[int] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        already_done = set(existing_df["global_cluster_id"].astype(int).tolist())
        existing_rows = existing_df.to_dict("records")
        print(f"  Resuming: {len(already_done)} clusters already labeled")

    labeler = StationaryThemeLabeler(model=args.model)
    new_rows: list[dict] = []

    todo = [cid for cid in cluster_ids if cid not in already_done]
    print(f"Labeling {len(todo)} clusters...")

    for i, cid in enumerate(todo, 1):
        post_ids = gc_df[gc_df["global_cluster_id"] == cid]["post_id"].tolist()

        posts = []
        for pid in post_ids:
            emb = id_to_emb.get(pid)
            txt = id_to_text.get(pid, "")
            posts.append(Post(
                post_id=pid,
                text=txt,
                timestamp=None,
                embedding=emb,
            ))

        # Select 5 representative posts via centroid proximity
        representative = labeler._select_representative(posts)
        rep_ids = [p.post_id for p in representative]

        theme = labeler.label_cluster(posts)
        print(f"  [{i}/{len(todo)}] cluster {cid:4d} | posts={len(posts):4d} | {theme}")

        new_rows.append({
            "global_cluster_id":  cid,
            "theme":              theme,
            "representative_posts": json.dumps(rep_ids),
        })

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Merge with any previously written rows and save
    all_rows = existing_rows + new_rows
    df_out = pd.DataFrame(all_rows)
    df_out["global_cluster_id"] = df_out["global_cluster_id"].astype(int)
    df_out = df_out.sort_values("global_cluster_id").reset_index(drop=True)
    df_out.to_parquet(out_path, index=False)

    print(f"\nDone. {len(df_out)} clusters written → {out_path}")


if __name__ == "__main__":
    main()
