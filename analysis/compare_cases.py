"""
Cross-case comparison plots using global_clusters.parquet + cluster_themes.parquet.

Produces six figures saved to analysis/figures/:
  1. narrative_lifecycles.png       — per-case subplot, top-20 clusters as horizontal bars, Y axis = theme label
  2. noise_over_time.png            — noise fraction vs. normalized time, all cases overlaid
  3. cluster_count_over_time.png    — active cluster count vs. normalized time, all cases overlaid
  4. lifespan_distribution.png      — KDE/histogram of lifespan in windows, log x axis
  5. top_narratives_per_case.png    — bar chart top-15 clusters by lifespan, Y axis = theme label
  6. narrative_birth_death_rate.png — birth and death event counts per window, normalized time

Usage
-----
python analysis/compare_cases.py

Missing global_clusters.parquet or cluster_themes.parquet for any case is handled gracefully.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde

CASES = ["venezuela", "iran", "russia"]
COLORS = {"venezuela": "#2166ac", "iran": "#1a9641", "russia": "#d73027"}
FIGURES_DIR = Path("analysis/figures")

LABEL_MAX_CHARS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trunc(s: str, n: int = LABEL_MAX_CHARS) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _theme_label(row) -> str:
    """Return theme string if present, else 'Cluster <id>'."""
    if pd.notna(row.get("theme")):
        return _trunc(str(row["theme"]))
    return f"Cluster {int(row['global_cluster_id'])}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_themes(case: str) -> pd.DataFrame:
    """Load cluster_themes.parquet; return empty DataFrame if missing."""
    path = Path("data/evaluated") / case / "cluster_themes.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["global_cluster_id", "theme"])
    df = pd.read_parquet(path, columns=["global_cluster_id", "theme"])
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    return df


def load_gc(case: str) -> pd.DataFrame | None:
    path = Path("data/evaluated") / case / "global_clusters.parquet"
    if not path.exists():
        print(f"  [skip] {case}: {path} not found")
        return None
    df = pd.read_parquet(path)
    df["post_id"] = df["post_id"].astype(str)
    df["window_dt"] = pd.to_datetime(df["window"], format="%Y-%m-%d-%H", utc=True, errors="coerce")
    df["is_noise"] = df["is_noise"].astype(bool)
    df["global_cluster_id"] = df["global_cluster_id"].where(~df["is_noise"])

    # Left-join themes; fall back to None if file absent
    themes = load_themes(case)
    if not themes.empty:
        df["global_cluster_id_int"] = df["global_cluster_id"].astype("Int64")
        df = df.merge(
            themes.rename(columns={"global_cluster_id": "global_cluster_id_int"}),
            on="global_cluster_id_int",
            how="left",
        )
        df = df.drop(columns=["global_cluster_id_int"])
    else:
        df["theme"] = pd.NA

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
    rows = []
    for wdt, wdf in df.groupby("window_dt"):
        total = len(wdf)
        noise = wdf["is_noise"].sum()
        active = wdf.loc[~wdf["is_noise"], "global_cluster_id"].nunique()
        rows.append({"window_dt": wdt, "total": total, "noise": noise, "active_clusters": active})
    out = pd.DataFrame(rows).sort_values("window_dt").reset_index(drop=True)
    out["noise_frac"] = out["noise"] / out["total"].replace(0, np.nan)
    t0, t1 = out["window_dt"].min(), out["window_dt"].max()
    span = (t1 - t0).total_seconds()
    out["t_norm"] = (out["window_dt"] - t0).dt.total_seconds() / (span if span > 0 else 1)
    return out


def cluster_lifespan_df(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return top_n clusters sorted by lifespan with theme labels resolved."""
    active = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
    active["global_cluster_id"] = active["global_cluster_id"].astype(int)

    span = (
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
    lifespan = span.merge(nw, on="global_cluster_id")

    # Attach theme (one row per cluster; take first non-null theme per cluster)
    theme_col = active.groupby("global_cluster_id")["theme"].first().reset_index() if "theme" in active.columns else pd.DataFrame(columns=["global_cluster_id", "theme"])
    lifespan = lifespan.merge(theme_col, on="global_cluster_id", how="left")
    lifespan["label"] = lifespan.apply(_theme_label, axis=1)

    return lifespan.sort_values("n_windows", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot 1: Narrative lifecycles
# ---------------------------------------------------------------------------

def plot_narrative_lifecycles(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    cases = list(data.keys())
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 9), squeeze=False)

    for ax, case in zip(axes[0], cases):
        df = data[case]
        lifespan = cluster_lifespan_df(df, top_n=20)
        cmap = plt.cm.Blues if case == "venezuela" else (plt.cm.Greens if case == "iran" else plt.cm.Reds)
        vmin, vmax = lifespan["n_windows"].min(), lifespan["n_windows"].max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        for i, row in lifespan.iterrows():
            c = cmap(0.3 + 0.7 * norm(row["n_windows"]))
            width_h = (row["death"] - row["birth"]).total_seconds() / 3600
            ax.barh(y=i, width=width_h, height=0.7, color=c)

        ax.set_yticks(range(len(lifespan)))
        ax.set_yticklabels(lifespan["label"].tolist(), fontsize=8)
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
        ax.scatter(ws["t_norm"], ws["noise_frac"], s=8, alpha=0.25, color=color)
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
        vals = active.groupby("global_cluster_id")["window_dt"].nunique().values
        if len(vals) == 0:
            continue

        log_max = np.log10(max(vals.max(), 2))
        bins = np.logspace(0, log_max, 30)
        ax.hist(vals, bins=bins, density=True, alpha=0.45, color=color,
                label=f"{case.capitalize()} (n={len(vals)})")

        if len(vals) >= 3:
            log_vals = np.log10(vals.clip(1))
            kde = gaussian_kde(log_vals, bw_method=0.4)
            x_log = np.linspace(0, log_max, 300)
            y_kde = kde(x_log)
            ax.plot(10 ** x_log, y_kde / (10 ** x_log * np.log(10)), color=color, linewidth=2)

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
# Plot 5: Top narratives per case
# ---------------------------------------------------------------------------

def plot_top_narratives(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    cases = list(data.keys())
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 8), squeeze=False)

    for ax, case in zip(axes[0], cases):
        color = COLORS[case]
        lifespan = cluster_lifespan_df(data[case], top_n=15)
        # Plot in ascending order so longest bar is at top
        lifespan_plot = lifespan.iloc[::-1].reset_index(drop=True)

        bars = ax.barh(
            y=range(len(lifespan_plot)),
            width=lifespan_plot["n_windows"],
            height=0.7,
            color=color,
            alpha=0.85,
        )

        # Value labels at end of each bar
        for bar, nw in zip(bars, lifespan_plot["n_windows"]):
            ax.text(
                bar.get_width() + 0.15,
                bar.get_y() + bar.get_height() / 2,
                str(int(nw)),
                va="center", fontsize=8, color="0.3",
            )

        ax.set_yticks(range(len(lifespan_plot)))
        ax.set_yticklabels(lifespan_plot["label"].tolist(), fontsize=8)
        ax.set_xlabel("Lifespan (windows)", fontsize=9)
        ax.set_title(f"{case.capitalize()} — Top 15 Narratives", fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, lifespan_plot["n_windows"].max() * 1.15)

    fig.suptitle("Top Narratives by Lifespan", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Narrative birth and death rates over normalized time
# ---------------------------------------------------------------------------

def plot_birth_death_rate(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    For each case, compute per-window counts of cluster births (first appearance)
    and deaths (last appearance), smooth them, and overlay on normalized time.
    """
    cases = list(data.keys())
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), squeeze=False, sharex=True)

    for ax, case in zip(axes[:, 0], cases):
        df = data[case]
        color = COLORS[case]

        active = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
        active["global_cluster_id"] = active["global_cluster_id"].astype(int)

        # Birth = first window for each global cluster
        births = (
            active.groupby("global_cluster_id")["window_dt"]
            .min()
            .rename("window_dt")
            .reset_index()
        )
        # Death = last window for each global cluster
        deaths = (
            active.groupby("global_cluster_id")["window_dt"]
            .max()
            .rename("window_dt")
            .reset_index()
        )

        # All unique window times for this case, normalized
        all_windows = df["window_dt"].dropna().sort_values().unique()
        t0, t1 = all_windows.min(), all_windows.max()
        span = (t1 - t0).total_seconds()

        def to_norm(ts):
            return (ts - t0).total_seconds() / (span if span > 0 else 1)

        win_norm = np.array([to_norm(w) for w in all_windows])

        birth_counts = births["window_dt"].value_counts().reindex(all_windows, fill_value=0).values
        death_counts = deaths["window_dt"].value_counts().reindex(all_windows, fill_value=0).values

        if len(win_norm) >= 5:
            k = max(3, len(win_norm) // 15)
            birth_smooth = uniform_filter1d(birth_counts.astype(float), size=k, mode="nearest")
            death_smooth = uniform_filter1d(death_counts.astype(float), size=k, mode="nearest")
        else:
            birth_smooth = birth_counts.astype(float)
            death_smooth = death_counts.astype(float)

        ax.bar(win_norm, birth_counts, width=1 / max(len(win_norm), 1),
               alpha=0.2, color=color)
        ax.bar(win_norm, -death_counts, width=1 / max(len(win_norm), 1),
               alpha=0.2, color="0.4")

        ax.plot(win_norm, birth_smooth, color=color, linewidth=2, label="Births")
        ax.plot(win_norm, -death_smooth, color="0.4", linewidth=2, linestyle="--", label="Deaths")

        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_ylabel("Clusters / window", fontsize=9)
        ax.set_title(f"{case.capitalize()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Annotate y-axis: positive = births, negative = deaths
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: str(int(abs(y))))
        )

    axes[-1, 0].set_xlabel("Normalized time (0 = start, 1 = end of dataset)", fontsize=10)
    fig.suptitle("Narrative Birth and Death Rates Over Time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading global_clusters.parquet (+ cluster_themes.parquet) for all cases...")
    data = load_all()

    if not data:
        print("No data found for any case — nothing to plot.")
        return

    print(f"\nLoaded cases: {list(data.keys())}\n")

    plot_narrative_lifecycles(data, FIGURES_DIR / "narrative_lifecycles.png")
    plot_noise_over_time(data, FIGURES_DIR / "noise_over_time.png")
    plot_cluster_count_over_time(data, FIGURES_DIR / "cluster_count_over_time.png")
    plot_lifespan_distribution(data, FIGURES_DIR / "lifespan_distribution.png")
    plot_top_narratives(data, FIGURES_DIR / "top_narratives_per_case.png")
    plot_birth_death_rate(data, FIGURES_DIR / "narrative_birth_death_rate.png")

    print("\nAll figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
