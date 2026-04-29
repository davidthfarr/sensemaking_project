"""
Cross-case overlay — side-by-side lifecycle and noise panels for multiple cases.

Produces a single figure with two rows:
  Row 1: Narrative lifecycle (top-N clusters per case, one column per case)
  Row 2: Noise fraction over time (all cases on one shared axis)

Usage
-----
python analysis/cross_case_overlay.py --cases venezuela ukraine
python analysis/cross_case_overlay.py --cases venezuela ukraine conflict_ie --output overlay.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def noise_by_window(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("window_start")
        .apply(lambda g: g["is_noise"].sum() / len(g), include_groups=False)
        .rename("noise_fraction")
        .reset_index()
        .sort_values("window_start")
    )


def plot(cases: list[str], top_n: int, output: str | None) -> None:
    n = len(cases)
    fig = plt.figure(figsize=(8 * n, max(5, top_n * 0.45 + 1) + 4))

    gs = gridspec.GridSpec(
        2, n,
        height_ratios=[max(4, top_n * 0.45), 3],
        hspace=0.45, wspace=0.35,
    )

    colors = plt.cm.tab10.colors

    # ── Row 0: lifecycle per case ─────────────────────────────────────────────
    for col, case in enumerate(cases):
        ax = fig.add_subplot(gs[0, col])
        df = load_results(case)
        spans = cluster_lifespan(df, top_n)

        for i, row in spans.iterrows():
            ax.barh(i, row["duration_h"], height=0.6, color=colors[i % len(colors)])

        ax.set_yticks(range(len(spans)))
        ax.set_yticklabels(
            [f"G{int(r.global_cluster_id)}: {r.cluster_theme[:40]}"
             for _, r in spans.iterrows()],
            fontsize=7,
        )
        ax.set_xlabel("Duration (hours)", fontsize=8)
        ax.set_title(f"Lifecycle — {case}", fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    # ── Row 1: noise over time, all cases overlaid ────────────────────────────
    ax_noise = fig.add_subplot(gs[1, :])
    for i, case in enumerate(cases):
        df = load_results(case)
        noise = noise_by_window(df)
        ax_noise.plot(
            noise["window_start"], noise["noise_fraction"],
            marker="o", markersize=3, linewidth=1.5,
            color=colors[i % len(colors)], label=case,
        )

    ax_noise.set_xlabel("Window start")
    ax_noise.set_ylabel("Noise fraction")
    ax_noise.set_title("Noise Fraction Over Time", fontsize=10, fontweight="bold")
    ax_noise.set_ylim(0, 1)
    ax_noise.legend()
    fig.autofmt_xdate()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    plot(args.cases, args.top_n, args.output)
