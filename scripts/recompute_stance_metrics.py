"""
Recompute controversy_score and stance_entropy from existing cluster_stance.parquet.

Deduplicates to one row per global_cluster_id (taking the mean of pct columns
across any duplicate rows) then overwrites the parquet in place — no API calls.

Usage
-----
python scripts/recompute_stance_metrics.py --case iran
python scripts/recompute_stance_metrics.py --case russia
python scripts/recompute_stance_metrics.py --case venezuela
python scripts/recompute_stance_metrics.py  # all cases

Metrics added / updated:
  controversy_score  = (1 - |support_pct - oppose_pct|) * (1 - neutral_pct)
      1.0 for a pure 50/50 split; penalised toward 0 as neutral dominates.
  stance_entropy     = scipy.stats.entropy([support_pct, oppose_pct, neutral_pct], base=2)
      Max ≈ 1.58 (log2 3) when all three stances are equal; 0 when unanimous.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import entropy as scipy_entropy

CASES = ["venezuela", "iran", "russia"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", choices=CASES, default=None,
                   help="Case to process (omit to run all cases)")
    return p.parse_args()


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate to one row per global_cluster_id and recompute metrics.

    If pct columns are present, aggregate by mean across duplicate rows.
    If absent, derive them from raw stance labels first.
    Returns one row per cluster.
    """
    PCT_COLS = ["support_pct", "oppose_pct", "neutral_pct"]

    if all(c in df.columns for c in PCT_COLS):
        # Aggregate in case there are duplicate cluster rows.
        # Exclude n_posts — it will be overwritten from global_clusters counts.
        skip = set(PCT_COLS) | {"global_cluster_id", "n_posts", "total_posts"}
        agg: dict[str, object] = {c: "mean" for c in PCT_COLS}
        for col in df.columns:
            if col not in skip:
                agg[col] = "first"
        df = df.groupby("global_cluster_id").agg(agg).reset_index()
    else:
        # Derive pct columns from raw stance labels, then collapse
        counts = (
            df.groupby(["global_cluster_id", "stance"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={"support": "n_support", "oppose": "n_oppose",
                             "neutral": "n_neutral"})
        )
        for col in ("n_support", "n_oppose", "n_neutral"):
            if col not in counts.columns:
                counts[col] = 0
        counts["support_pct"] = counts["n_support"] / counts[["n_support", "n_oppose", "n_neutral"]].sum(axis=1)
        counts["oppose_pct"]  = counts["n_oppose"]  / counts[["n_support", "n_oppose", "n_neutral"]].sum(axis=1)
        counts["neutral_pct"] = counts["n_neutral"] / counts[["n_support", "n_oppose", "n_neutral"]].sum(axis=1)
        # Merge back any non-stance columns (e.g. theme) from first row per cluster
        extra = df.groupby("global_cluster_id").first().drop(columns=["stance"], errors="ignore")
        df = counts.reset_index().merge(extra.reset_index(), on="global_cluster_id", how="left")

    s   = df["support_pct"]
    o   = df["oppose_pct"]
    neu = df["neutral_pct"]

    df["controversy_score"] = (1.0 - (s - o).abs()) * (1.0 - neu)
    df["stance_entropy"] = [
        float(scipy_entropy([si, oi, ni], base=2))
        for si, oi, ni in zip(s, o, neu)
    ]
    return df


def process_case(case: str) -> None:
    stance_path = Path("data/evaluated") / case / "cluster_stance.parquet"
    gc_path     = Path("data/processed")  / case / "global_clusters.parquet"

    if not stance_path.exists():
        print(f"  [skip] {case}: {stance_path} not found")
        return

    df = pd.read_parquet(stance_path)
    original_cols = list(df.columns)

    df = compute_metrics(df)

    # Recompute n_posts and total_posts from unique post_ids in global_clusters
    if gc_path.exists():
        global_df = pd.read_parquet(gc_path, columns=["global_cluster_id", "post_id"])
        cluster_sizes = (
            global_df.groupby("global_cluster_id")["post_id"]
            .nunique()
            .reset_index(name="total_posts")
        )
        cluster_sizes["global_cluster_id"] = cluster_sizes["global_cluster_id"].astype(
            df["global_cluster_id"].dtype
        )
        df = df.merge(cluster_sizes, on="global_cluster_id", how="left")
        df["n_posts"] = df["total_posts"]
    else:
        print(f"  [warn] {case}: {gc_path} not found — n_posts/total_posts not updated")

    df.to_parquet(stance_path, index=False)

    added   = [c for c in df.columns if c not in original_cols]
    updated = [c for c in ("controversy_score", "stance_entropy", "n_posts") if c in original_cols]
    print(
        f"  {case}: {len(df):,} clusters — "
        f"added={added or '—'}  updated={updated or '—'}  → {stance_path}"
    )


def main() -> None:
    args = parse_args()
    cases = [args.case] if args.case else CASES

    print(f"Recomputing stance metrics for: {cases}")
    for case in cases:
        process_case(case)
    print("Done.")


if __name__ == "__main__":
    main()
