"""
Recompute controversy_score and stance_entropy from existing cluster_stance.parquet.

Reads support_pct, oppose_pct, neutral_pct that were written during
run_stance_classification.py and derives the two metrics without any API calls,
then overwrites the parquet in place.

Usage
-----
python scripts/recompute_stance_metrics.py --case iran
python scripts/recompute_stance_metrics.py --case russia
python scripts/recompute_stance_metrics.py --case venezuela
python scripts/recompute_stance_metrics.py  # all cases

Metrics added / updated:
  controversy_score  = 1 - |support_pct - oppose_pct|
      1.0 when support and oppose are equal; 0.0 when one side dominates.
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
    Derive controversy_score and stance_entropy from per-cluster pct columns.

    If the pct columns are not present, recomputes them from raw stance labels.
    """
    # If pct columns are missing, derive them from raw stance labels
    if "support_pct" not in df.columns:
        n = df.groupby("global_cluster_id")["stance"].transform("count")
        df["support_pct"] = (df["stance"] == "support").astype(float) / n
        df["oppose_pct"]  = (df["stance"] == "oppose").astype(float)  / n
        df["neutral_pct"] = (df["stance"] == "neutral").astype(float) / n
        # Reduce to per-cluster values via groupby mean (all rows in a cluster
        # already hold the same pct value once set above)
        for col in ("support_pct", "oppose_pct", "neutral_pct"):
            df[col] = df.groupby("global_cluster_id")[col].transform("mean")

    df["controversy_score"] = 1.0 - (df["support_pct"] - df["oppose_pct"]).abs()
    df["stance_entropy"] = df.apply(
        lambda r: float(scipy_entropy(
            [r["support_pct"], r["oppose_pct"], r["neutral_pct"]], base=2
        )),
        axis=1,
    )
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

    added = [c for c in df.columns if c not in original_cols]
    updated = [c for c in ("controversy_score", "stance_entropy") if c in original_cols]
    print(
        f"  {case}: {len(df):,} rows — "
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
