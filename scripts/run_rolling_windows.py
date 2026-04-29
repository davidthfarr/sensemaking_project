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

Windows are processed in parallel with joblib (n_jobs=-1).

Input:  data/processed/<case>/posts_repr.parquet
Output: data/evaluated/<case>/<window_start>.parquet  (one file per window)

Note: For the full pipeline including theme generation and stance classification,
use scripts/run_pipeline.py instead.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from sensemaking.clustering.hdbscan import HDBSCANClusterer
from sensemaking.data.schemas import Post

# Per-case parameter defaults.
# window_hours / step_hours are in hours; Russia uses 7d/1d windows.
CASE_PARAMS = {
    "venezuela": dict(window_hours=12,  step_hours=4,  min_cluster_size=8, min_samples=2),
    "iran":      dict(window_hours=12,  step_hours=4,  min_cluster_size=8, min_samples=2),
    "russia":    dict(window_hours=168, step_hours=24, min_cluster_size=5, min_samples=10,
                     cluster_selection_epsilon=0.15),
}

# If a window returns more clusters than this, double min_cluster_size and retry.
MAX_CLUSTERS = 50
MAX_RETRIES  = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=list(CASE_PARAMS),
                   help="Case name — sets per-case defaults from CASE_PARAMS")
    p.add_argument("--window-hours",      type=int,   default=None,
                   help="Rolling window size in hours (overrides case default)")
    p.add_argument("--step-hours",        type=int,   default=None,
                   help="Step between windows in hours (overrides case default)")
    p.add_argument("--min-cluster-size",  type=int,   default=None,
                   help="HDBSCAN min_cluster_size floor (overrides case default)")
    p.add_argument("--min-samples",       type=int,   default=None,
                   help="HDBSCAN min_samples (overrides case default)")
    p.add_argument("--cluster-epsilon",   type=float, default=None)
    p.add_argument("--n-jobs",            type=int,   default=-1,
                   help="joblib n_jobs (-1 = all cores)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-window worker (runs in parallel)
# ---------------------------------------------------------------------------

def process_window(
    window_start: datetime,
    df: pd.DataFrame,
    window_hours: int,
    min_cluster_size_floor: int,
    min_samples: int,
    cluster_epsilon: float,
    output_dir: Path,
) -> dict | None:
    """
    Cluster one window and write its parquet. Returns a summary dict, or None
    if the window was skipped (too few posts).
    """
    window_end = window_start + timedelta(hours=window_hours)
    window_df = df[
        (df["timestamp"] >= window_start) &
        (df["timestamp"] < window_end)
    ]

    if len(window_df) < min_cluster_size_floor:
        return None

    posts = [
        Post(
            post_id=row.post_id,
            timestamp=row.timestamp,
            text=row.text,
            embedding=row.embedding,
        )
        for _, row in window_df.iterrows()
    ]

    # Dynamic base: 1 cluster per 50 posts, floor at min_cluster_size_floor
    base_mcs = max(min_cluster_size_floor, len(posts) // 50)
    attempt_mcs = base_mcs
    warnings: list[str] = []

    for attempt in range(1, MAX_RETRIES + 1):
        clusterer = HDBSCANClusterer(
            min_cluster_size=attempt_mcs,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_epsilon,
        )
        posts = clusterer.fit_predict(posts)
        n_clusters = len({p.cluster_id for p in posts if p.cluster_id is not None})
        if n_clusters <= MAX_CLUSTERS or attempt == MAX_RETRIES:
            break
        warnings.append(
            f"  WARNING {window_start.strftime('%Y-%m-%d %H:%M')} | "
            f"{n_clusters} clusters > {MAX_CLUSTERS} cap — "
            f"retrying with min_cluster_size={attempt_mcs * 2} (attempt {attempt}/{MAX_RETRIES})"
        )
        attempt_mcs *= 2

    noise_frac = sum(p.is_noise for p in posts) / len(posts)

    out_df = pd.DataFrame({
        "post_id":    [p.post_id for p in posts],
        "window":     window_start.strftime("%Y-%m-%d-%H"),
        "cluster_id": [p.cluster_id if not p.is_noise else None for p in posts],
        "is_noise":   [p.is_noise for p in posts],
    })
    out_path = output_dir / f"{window_start.strftime('%Y-%m-%d-%H')}.parquet"
    out_df.to_parquet(out_path, index=False)

    return {
        "window_start": window_start,
        "n_posts":      len(posts),
        "n_clusters":   n_clusters,
        "noise_frac":   noise_frac,
        "attempt_mcs":  attempt_mcs,
        "base_mcs":     base_mcs,
        "warnings":     warnings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    defaults = CASE_PARAMS[args.case]
    window_hours           = args.window_hours     if args.window_hours     is not None else defaults["window_hours"]
    step_hours             = args.step_hours       if args.step_hours       is not None else defaults["step_hours"]
    min_cluster_size_floor = args.min_cluster_size if args.min_cluster_size is not None else defaults["min_cluster_size"]
    min_samples            = args.min_samples      if args.min_samples      is not None else defaults["min_samples"]
    cluster_epsilon        = args.cluster_epsilon  if args.cluster_epsilon  is not None else defaults.get("cluster_selection_epsilon", 0.0)

    print(
        f"Case: {args.case} | window={window_hours}h | step={step_hours}h | "
        f"mcs_floor={min_cluster_size_floor} | min_samples={min_samples} | "
        f"epsilon={cluster_epsilon} | n_jobs={args.n_jobs}"
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

    # Pre-generate all window start times
    window_starts = []
    ws = min_time
    while ws <= max_time:
        window_starts.append(ws)
        ws += timedelta(hours=step_hours)
    print(f"Dispatching {len(window_starts)} windows across {args.n_jobs} jobs...")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_window)(
            ws, df, window_hours,
            min_cluster_size_floor, min_samples, cluster_epsilon,
            output_dir,
        )
        for ws in window_starts
    )

    # Print results in chronological order
    for r in sorted((r for r in results if r is not None), key=lambda x: x["window_start"]):
        for w in r["warnings"]:
            print(w)
        flag = " [capped]" if r["attempt_mcs"] > r["base_mcs"] else ""
        print(
            f"  {r['window_start'].strftime('%Y-%m-%d %H:%M')} | "
            f"posts={r['n_posts']:4d} | clusters={r['n_clusters']:2d} | "
            f"noise={r['noise_frac']:.2f} | mcs={r['attempt_mcs']}{flag}"
        )

    n_written = sum(1 for r in results if r is not None)
    print(f"\nDone. {n_written} windows written to {output_dir}/")


if __name__ == "__main__":
    main()
