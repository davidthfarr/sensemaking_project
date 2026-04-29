"""
Centroid drift — trajectory of top narrative clusters in 2D PCA space over time.

Requires both results.parquet (cluster membership) and posts_repr.parquet
(embeddings) for the case. Computes per-window centroids, fits PCA across all
centroids, then plots each cluster's path through the projected space.

Usage
-----
python analysis/centroid_drift.py --case venezuela
python analysis/centroid_drift.py --case ukraine --top-n 8 --output drift.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_data(case: str):
    results_path = Path("data/evaluated") / case / "results.parquet"
    repr_path    = Path("data/processed") / case / "posts_repr.parquet"

    if not results_path.exists():
        raise FileNotFoundError(f"No results for '{case}' at {results_path}")
    if not repr_path.exists():
        raise FileNotFoundError(f"No representations for '{case}' at {repr_path}")

    results = pd.read_parquet(results_path)
    results = results[~results["is_noise"] & results["global_cluster_id"].notna()]
    results["window_start"] = pd.to_datetime(results["window_start"], utc=True, errors="coerce")

    reprs = pd.read_parquet(repr_path)[["post_id", "embedding"]]
    return results.merge(reprs, on="post_id", how="inner")


def compute_centroids(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_clusters = (
        df.groupby("global_cluster_id")["post_id"]
        .count()
        .nlargest(top_n)
        .index
    )
    sub = df[df["global_cluster_id"].isin(top_clusters)]

    rows = []
    for (gid, ws), grp in sub.groupby(["global_cluster_id", "window_start"]):
        embs = np.vstack(grp["embedding"].values)
        centroid = embs.mean(axis=0)
        theme = grp["cluster_theme"].iloc[0]
        rows.append({
            "global_cluster_id": gid,
            "window_start": ws,
            "centroid": centroid,
            "cluster_theme": theme,
        })
    return pd.DataFrame(rows)


def plot(case: str, top_n: int, output: str | None) -> None:
    df = load_data(case)
    centroids = compute_centroids(df, top_n)

    # PCA across all centroid snapshots
    all_centroids = np.vstack(centroids["centroid"].values)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_centroids)
    centroids["pc1"] = projected[:, 0]
    centroids["pc2"] = projected[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10.colors

    for i, (gid, grp) in enumerate(centroids.groupby("global_cluster_id")):
        grp = grp.sort_values("window_start")
        color = colors[i % len(colors)]
        theme_short = grp["cluster_theme"].iloc[0][:40]

        ax.plot(grp["pc1"], grp["pc2"], "-o", color=color,
                linewidth=1.5, markersize=5, label=f"G{int(gid)}: {theme_short}…")
        # Mark birth with a larger dot
        ax.scatter(grp["pc1"].iloc[0], grp["pc2"].iloc[0],
                   s=80, color=color, zorder=5, marker="^")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Centroid Drift — {case} (top {top_n} clusters)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    plot(args.case, args.top_n, args.output)
