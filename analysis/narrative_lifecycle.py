"""
Narrative lifecycle plot — horizontal bars showing each cluster's active window span.

Usage
-----
python analysis/narrative_lifecycle.py --cases venezuela ukraine
python analysis/narrative_lifecycle.py --cases venezuela --top-n 20 --output lifecycle.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(case: str) -> pd.DataFrame:
    path = Path("data/evaluated") / case / "results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No results for '{case}' at {path}")
    df = pd.read_parquet(path)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    return df


def cluster_lifespan(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    active = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
    active["global_cluster_id"] = active["global_cluster_id"].astype(int)
    lifespan = (
        active.groupby(["global_cluster_id", "cluster_theme"])["window_start"]
        .agg(birth="min", death="max")
        .reset_index()
    )
    lifespan["duration_h"] = (
        (lifespan["death"] - lifespan["birth"]).dt.total_seconds() / 3600
    )
    return lifespan.sort_values("duration_h", ascending=False).head(top_n).reset_index(drop=True)


def plot(cases: list[str], top_n: int, output: str | None) -> None:
    fig, axes = plt.subplots(
        1, len(cases),
        figsize=(8 * len(cases), max(5, top_n * 0.5 + 1)),
        squeeze=False,
    )

    for ax, case in zip(axes[0], cases):
        df = load_results(case)
        spans = cluster_lifespan(df, top_n)

        colors = plt.cm.tab20.colors
        for i, row in spans.iterrows():
            ax.barh(i, row["duration_h"], height=0.6, color=colors[i % len(colors)])

        ax.set_yticks(range(len(spans)))
        ax.set_yticklabels(
            [f"G{int(r.global_cluster_id)}: {r.cluster_theme[:45]}…"
             if len(r.cluster_theme) > 45 else f"G{int(r.global_cluster_id)}: {r.cluster_theme}"
             for _, r in spans.iterrows()],
            fontsize=8,
        )
        ax.set_xlabel("Duration (hours)")
        ax.set_title(f"Narrative Lifecycle — {case}", fontsize=11, fontweight="bold")
        ax.invert_yaxis()

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    plot(args.cases, args.top_n, args.output)
