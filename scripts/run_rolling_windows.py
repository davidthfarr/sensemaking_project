"""
Rolling-window HDBSCAN clustering for a named case.

Usage
-----
python scripts/run_rolling_windows.py --case iran
python scripts/run_rolling_windows.py --case russia
python scripts/run_rolling_windows.py --case venezuela

Per-case defaults are set in CASE_PARAMS below. Any parameter can be
overridden on the command line:

    python scripts/run_rolling_windows.py --case russia --min-cluster-size 8

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

# Per-case parameter defaults.
# window_hours / step_hours are in hours; Russia uses 7d/1d windows.
CASE_PARAMS = {
    "venezuela": dict(window_hours=12,  step_hours=4,  min_cluster_size=8, min_samples=2),
    "iran":      dict(window_hours=12,  step_hours=4,  min_cluster_size=8, min_samples=2),
    "russia":    dict(window_hours=168, step_hours=24, min_cluster_size=5, min_samples=2),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=list(CASE_PARAMS),
                   help="Case name — sets per-case defaults from CASE_PARAMS")
    # Use None as sentinel so we can detect when the user explicitly overrides
    p.add_argument("--window-hours",      type=int,   default=None,
                   help="Rolling window size in hours (overrides case default)")
    p.add_argument("--step-hours",        type=int,   default=None,
                   help="Step between windows in hours (overrides case default)")
    p.add_argument("--min-cluster-size",  type=int,   default=None,
                   help="HDBSCAN min_cluster_size (overrides case default)")
    p.add_argument("--min-samples",       type=int,   default=None,
                   help="HDBSCAN min_samples (overrides case default)")
    p.add_argument("--cluster-epsilon",   type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Merge case defaults with any explicit CLI overrides
    defaults = CASE_PARAMS[args.case]
    window_hours     = args.window_hours     if args.window_hours     is not None else defaults["window_hours"]
    step_hours       = args.step_hours       if args.step_hours       is not None else defaults["step_hours"]
    min_cluster_size = args.min_cluster_size if args.min_cluster_size is not None else defaults["min_cluster_size"]
    min_samples      = args.min_samples      if args.min_samples      is not None else defaults["min_samples"]

    print(
        f"Case: {args.case} | window={window_hours}h | step={step_hours}h | "
        f"min_cluster_size={min_cluster_size} | min_samples={min_samples}"
    )

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
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=args.cluster_epsilon,
    )

    window_start = min_time
    while window_start <= max_time:
        window_end = window_start + timedelta(hours=window_hours)

        window_df = df[
            (df["timestamp"] >= window_start) &
            (df["timestamp"] < window_end)
        ]

        if len(window_df) < min_cluster_size:
            window_start += timedelta(hours=step_hours)
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

        window_start += timedelta(hours=step_hours)

    print(f"\nDone. Output written to {output_dir}/")


if __name__ == "__main__":
    main()
