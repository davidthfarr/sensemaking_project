"""
Rolling-window HDBSCAN clustering for a named case.

Usage
-----
python scripts/run_rolling_windows.py --case iran
python scripts/run_rolling_windows.py --case russia
python scripts/run_rolling_windows.py --case venezuela --window-hours 24 --step-hours 6

Input:  data/processed/<case>/posts_repr.parquet
Output: data/evaluated/<case>/<window_start>.parquet  (one file per window)

Note: For the full pipeline including theme generation and stance classification,
use scripts/run_pipeline.py instead.
"""

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

from sensemaking.clustering.hdbscan import HDBSCANClusterer
from sensemaking.data.schemas import Post


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=["iran", "russia", "venezuela"],
                   help="Case name — resolves input/output paths automatically")
    p.add_argument("--window-hours", type=int, default=12,
                   help="Rolling window size in hours")
    p.add_argument("--step-hours", type=int, default=4,
                   help="Step between windows in hours")
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--cluster-epsilon", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path    = Path("data/processed") / args.case / "posts_repr.parquet"
    output_dir = Path("data/evaluated") / args.case
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading representations from {in_path}")
    df = pd.read_parquet(in_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    min_time = df["timestamp"].min().floor("h")
    max_time = df["timestamp"].max().floor("h")
    print(f"Time range: {min_time} → {max_time}")

    clusterer = HDBSCANClusterer(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_epsilon,
    )

    window_start = min_time
    while window_start <= max_time:
        window_end = window_start + timedelta(hours=args.window_hours)

        window_df = df[
            (df["timestamp"] >= window_start) &
            (df["timestamp"] < window_end)
        ]

        if len(window_df) == 0:
            window_start += timedelta(hours=args.step_hours)
            continue

        posts = [
            Post(
                post_id=row.post_id,
                timestamp=row.timestamp,
                text=row.text,
                embedding=row.embedding,
            )
            for _, row in window_df.iterrows()
        ]

        posts = clusterer.fit_predict(posts)

        n_clusters = len({p.cluster_id for p in posts if p.cluster_id is not None})
        noise_frac = sum(p.is_noise for p in posts) / len(posts)
        print(
            f"  {window_start.strftime('%Y-%m-%d %H:%M')} | "
            f"posts={len(posts):4d} | clusters={n_clusters:2d} | noise={noise_frac:.2f}"
        )

        out_df = pd.DataFrame({
            "post_id":    [p.post_id for p in posts],
            "window":     window_start.strftime("%Y-%m-%d-%H"),
            "cluster_id": [p.cluster_id if not p.is_noise else None for p in posts],
            "is_noise":   [p.is_noise for p in posts],
        })

        out_path = output_dir / f"{window_start.strftime('%Y-%m-%d-%H')}.parquet"
        out_df.to_parquet(out_path, index=False)

        window_start += timedelta(hours=args.step_hours)

    print(f"\nDone. Output written to {output_dir}/")


if __name__ == "__main__":
    main()
