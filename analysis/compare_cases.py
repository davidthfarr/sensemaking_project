"""
Cross-case comparison plots using global_clusters.parquet.

Produces four figures saved to analysis/figures/:
  1. narrative_lifecycles.png  — per-case subplot, top-20 clusters as horizontal bars
  2. noise_over_time.png       — noise fraction vs. normalized time, all cases overlaid
  3. cluster_count_over_time.png — active cluster count vs. normalized time, all cases
  4. lifespan_distribution.png — KDE/histogram of lifespan in windows, all cases

Usage
-----
python analysis/compare_cases.py

Missing global_clusters.parquet for any case is skipped gracefully.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

CASES = ["venezuela", "iran", "russia"]
COLORS = {"venezuela": "#2166ac", "iran": "#1a9641", "russia": "#d73027"}
FIGURES_DIR = Path("analysis/figures")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gc(case: str) -> pd.DataFrame | None:
    path = Path("data/evaluated") / case / "global_clusters.parquet"
    if not path.exists():
        print(f"  [skip] {case}: {path} not found")
        return None
    df = pd.read_parquet(path)
    df["post_id"] = df["post_id"].astype(str)
    # window column is a string like "2019-01-03-12"; parse to datetime
    df["window_dt"] = pd.to_datetime(df["window"], format="%Y-%m-%d-%H", utc=True, errors="coerce")
    df["is_noise"] = df["is_noise"].astype(bool)
    df["global_cluster_id"] = df["global_cluster_id"].where(~df["is_noise"])
    return df


def load_all() -> dict[str, pd.DataFrame]:
    loaded = {}
    for case in CASES:
        df = load_gc(case)
        if df is not None:
            loaded[case] = df
    return loaded


# ---------------------------------------------------------------------------
# Per-window summary helper
# ---------------------------------------------------------------------------

def window_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-window noise fraction and active cluster count."""
    rows = []
    for wdt, wdf in df.groupby("window_dt"):
        total = len(wdf)
        noise = wdf["is_noise"].sum()
        active = wdf.loc[~wdf["is_noise"], "global_cluster_id"].nunique()
        rows.append({"window_dt": wdt, "total": total, "noise": noise, "active_clusters": active})
    out = pd.DataFrame(rows).sort_values("window_dt").reset_index(drop=True)
    out["noise_frac"] = out["noise"] / out["total"].replace(0, np.nan)
    # Normalized time 0→1
    t0, t1 = out["window_dt"].min(), out["window_dt"].max()
    span = (t1 - t0).total_seconds()
    out["t_norm"] = (out["window_dt"] - t0).dt.total_seconds() / (span if span > 0 else 1)
    return out


# ---------------------------------------------------------------------------
# Plot 1: Narrative lifecycles
# ---------------------------------------------------------------------------

def plot_narrative_lifecycles(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    cases = list(data.keys())
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8), squeeze=False)

    for ax, case in zip(axes[0], cases):
        df = data[case]
        color = COLORS[case]
        active = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
        active["global_cluster_id"] = active["global_cluster_id"].astype(int)

        lifespan = (
            active.groupby("global_cluster_id")["window_dt"]
            .agg(birth="min", death="max")
            .reset_index()
        )
        nw = (
            active.groupby("global_cluster_id")["window_dt"]
            .nunique()
            .rename("n_windows")
            .reset_index()
        )
        lifespan = lifespan.merge(nw, on="global_cluster_id")
        lifespan = lifespan.sort_values("n_windows", ascending=False).head(20).reset_index(drop=True)

        # Normalize lifespan for color mapping
        vmin, vmax = lifespan["n_windows"].min(), lifespan["n_windows"].max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Blues if case == "venezuela" else (plt.cm.Greens if case == "iran" else plt.cm.Reds)

        for i, row in lifespan.iterrows():
            c = cmap(0.3 + 0.7 * norm(row["n_windows"]))
            ax.barh(
                y=i,
                width=(row["death"] - row["birth"]).total_seconds() / 3600,
                left=0,
                height=0.7,
                color=c,
            )
            ax.text(
                (row["death"] - row["birth"]).total_seconds() / 3600 + 1,
                i,
                f"G{int(row['global_cluster_id'])}",
                va="center", fontsize=7, color="0.3",
            )

        # Use actual datetimes on x axis — convert duration back to dates for labels
        # Actually plot with hours on x axis and label every N hours
        ax.set_yticks(range(len(lifespan)))
        ax.set_yticklabels(
            [f"G{int(r.global_cluster_id)}  ({int(r.n_windows)} w)" for r in lifespan.itertuples()],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Lifespan (hours)", fontsize=9)
        ax.set_title(f"{case.capitalize()} — Top 20 Clusters by Lifespan", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}h"))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Windows alive", shrink=0.6, pad=0.02)

    fig.suptitle("Narrative Lifecycles by Case", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Noise fraction over time
# ---------------------------------------------------------------------------

def plot_noise_over_time(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, df in data.items():
        ws = window_summary(df)
        color = COLORS[case]

        # Raw scatter (faint)
        ax.scatter(ws["t_norm"], ws["noise_frac"], s=8, alpha=0.25, color=color)

        # Smoothed trend — uniform_filter1d on sorted values
        if len(ws) >= 5:
            k = max(3, len(ws) // 15)
            smoothed = uniform_filter1d(ws["noise_frac"].fillna(0).values, size=k, mode="nearest")
            ax.plot(ws["t_norm"], smoothed, color=color, linewidth=2, label=case.capitalize())
        else:
            ax.plot(ws["t_norm"], ws["noise_frac"], color=color, linewidth=2, label=case.capitalize())

    ax.set_xlabel("Normalized time (0 = start, 1 = end of dataset)", fontsize=10)
    ax.set_ylabel("Noise fraction", fontsize=10)
    ax.set_title("Noise Fraction Over Time", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Active cluster count over time
# ---------------------------------------------------------------------------

def plot_cluster_count_over_time(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, df in data.items():
        ws = window_summary(df)
        color = COLORS[case]

        ax.scatter(ws["t_norm"], ws["active_clusters"], s=8, alpha=0.25, color=color)

        if len(ws) >= 5:
            k = max(3, len(ws) // 15)
            smoothed = uniform_filter1d(ws["active_clusters"].fillna(0).values, size=k, mode="nearest")
            ax.plot(ws["t_norm"], smoothed, color=color, linewidth=2, label=case.capitalize())
        else:
            ax.plot(ws["t_norm"], ws["active_clusters"], color=color, linewidth=2, label=case.capitalize())

    ax.set_xlabel("Normalized time (0 = start, 1 = end of dataset)", fontsize=10)
    ax.set_ylabel("Active clusters per window", fontsize=10)
    ax.set_title("Active Cluster Count Over Time", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Lifespan distribution
# ---------------------------------------------------------------------------

def plot_lifespan_distribution(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for case, df in data.items():
        color = COLORS[case]
        active = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
        active["global_cluster_id"] = active["global_cluster_id"].astype(int)

        lifespans = (
            active.groupby("global_cluster_id")["window_dt"]
            .nunique()
            .rename("n_windows")
        )
        vals = lifespans.values
        if len(vals) == 0:
            continue

        # Log-spaced bins from 1 to max
        log_max = np.log10(max(vals.max(), 2))
        bins = np.logspace(0, log_max, 30)

        ax.hist(
            vals,
            bins=bins,
            density=True,
            alpha=0.45,
            color=color,
            label=f"{case.capitalize()} (n={len(vals)})",
        )

        # Overlay KDE
        from scipy.stats import gaussian_kde
        if len(vals) >= 3:
            log_vals = np.log10(vals.clip(1))
            kde = gaussian_kde(log_vals, bw_method=0.4)
            x_log = np.linspace(0, log_max, 300)
            y_kde = kde(x_log)
            # Scale KDE to roughly match histogram density in log space
            ax.plot(10 ** x_log, y_kde / (10 ** x_log * np.log(10)),
                    color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Cluster lifespan (number of windows)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Cluster Lifespan Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading global_clusters.parquet for all cases...")
    data = load_all()

    if not data:
        print("No data found for any case — nothing to plot.")
        return

    print(f"\nLoaded cases: {list(data.keys())}\n")

    plot_narrative_lifecycles(data, FIGURES_DIR / "narrative_lifecycles.png")
    plot_noise_over_time(data, FIGURES_DIR / "noise_over_time.png")
    plot_cluster_count_over_time(data, FIGURES_DIR / "cluster_count_over_time.png")
    plot_lifespan_distribution(data, FIGURES_DIR / "lifespan_distribution.png")

    print("\nAll figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
