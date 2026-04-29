"""
Stance distribution per cluster — stacked horizontal bars (support / oppose / neutral).

Usage
-----
python analysis/stance_distribution.py --case venezuela
python analysis/stance_distribution.py --case ukraine --top-n 20 --output stance.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STANCE_COLORS = {
    "support": "#2ecc71",
    "neutral": "#95a5a6",
    "oppose":  "#e74c3c",
}
STANCE_ORDER = ["support", "neutral", "oppose"]


def load_results(case: str) -> pd.DataFrame:
    path = Path("data/evaluated") / case / "results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No results for '{case}' at {path}")
    df = pd.read_parquet(path)
    return df[~df["is_noise"] & df["stance"].notna() & df["cluster_theme"].notna()]


def stance_counts(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Rank clusters by total post count, take top N
    top_clusters = (
        df.groupby("global_cluster_id")["post_id"]
        .count()
        .nlargest(top_n)
        .index
    )
    sub = df[df["global_cluster_id"].isin(top_clusters)]

    counts = (
        sub.groupby(["global_cluster_id", "cluster_theme", "stance"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("global_cluster_id")["count"].transform("sum")
    counts["pct"] = counts["count"] / totals

    # Pivot: rows=cluster, cols=stance
    pivot = counts.pivot_table(
        index=["global_cluster_id", "cluster_theme"],
        columns="stance",
        values="pct",
        fill_value=0,
    ).reset_index()

    for s in STANCE_ORDER:
        if s not in pivot.columns:
            pivot[s] = 0.0

    # Sort by support − oppose
    pivot["support_bias"] = pivot["support"] - pivot["oppose"]
    return pivot.sort_values("support_bias", ascending=True).reset_index(drop=True)


def plot(case: str, top_n: int, output: str | None) -> None:
    df = load_results(case)
    pivot = stance_counts(df, top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.45 + 1)))

    lefts = [0.0] * len(pivot)
    for stance in STANCE_ORDER:
        vals = pivot[stance].values
        ax.barh(range(len(pivot)), vals, left=lefts, height=0.6,
                color=STANCE_COLORS[stance], label=stance)
        lefts = [l + v for l, v in zip(lefts, vals)]

    labels = [
        f"G{int(r.global_cluster_id)}: {r.cluster_theme[:50]}…"
        if len(r.cluster_theme) > 50 else f"G{int(r.global_cluster_id)}: {r.cluster_theme}"
        for _, r in pivot.iterrows()
    ]
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Proportion")
    ax.set_xlim(0, 1)
    ax.set_title(f"Stance Distribution per Cluster — {case}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    plot(args.case, args.top_n, args.output)
