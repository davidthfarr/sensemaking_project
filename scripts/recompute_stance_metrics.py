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
        # Aggregate in case there are duplicate cluster rows
        agg: dict[str, object] = {c: "mean" for c in PCT_COLS}
        # Carry non-metric columns from the first row
        for col in df.columns:
            if col not in PCT_COLS and col != "global_cluster_id":
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
        counts["n_posts"] = counts[["n_support", "n_oppose", "n_neutral"]].sum(axis=1)
        counts["support_pct"] = counts["n_support"] / counts["n_posts"]
        counts["oppose_pct"]  = counts["n_oppose"]  / counts["n_posts"]
        counts["neutral_pct"] = counts["n_neutral"] / counts["n_posts"]
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
    path = Path("data/evaluated") / case / "cluster_stance.parquet"
    if not path.exists():
        print(f"  [skip] {case}: {path} not found")
        return

    df = pd.read_parquet(path)
    original_cols = list(df.columns)

    df = compute_metrics(df)

    df.to_parquet(path, index=False)

    added   = [c for c in df.columns if c not in original_cols]
    updated = [c for c in ("controversy_score", "stance_entropy") if c in original_cols]
    print(
        f"  {case}: {len(df):,} clusters — "
        f"added={added or '—'}  updated={updated or '—'}  → {path}"
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
