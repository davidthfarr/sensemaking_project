"""
Cross-case comparison plots using global_clusters.parquet + cluster_themes.parquet.

Produces figures saved to analysis/figures/:
  1. narrative_lifecycles.pdf/png       — per-case subplot, top-20 clusters as horizontal bars
  2. noise_over_time.pdf/png            — noise fraction vs. normalized time, all cases overlaid
  3. cluster_count_over_time.pdf/png    — active cluster count vs. normalized time
  4. lifespan_distribution.pdf/png      — KDE/histogram of lifespan in windows (log x)
  5. top_narratives_per_case.pdf/png    — bar chart top-15 clusters by lifespan
  6. narrative_birth_death_rate.pdf/png — birth/death event counts per window
  7. stance_over_time.pdf/png           — stacked area support/neutral/oppose vs. real dates
  8. drift_normalized.pdf/png           — drift_rate vs. net_displacement, 1×3 faceted scatter
  9. persistence_normalized.pdf/png     — persistence_frac vs. log10(n_posts), 1×3 with OLS fit
     cluster_metrics.csv               — per-cluster metrics joining drift, persistence, stance

Usage
-----
python analysis/compare_cases.py

Missing parquets for any case are handled gracefully (per-case skip with warning).
"""

import matplotlib.dates as mdates
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde, linregress

# ── Case config ────────────────────────────────────────────────────────────────
CASES  = ["venezuela", "iran", "russia"]
COLORS = {"venezuela": "#2166ac", "iran": "#1a9641", "russia": "#d73027"}
OPPOSE_COLORS = {"venezuela": "#053061", "iran": "#00441b", "russia": "#67000d"}

FIGURES_DIR   = Path("analysis/figures")
LABEL_MAX_CHARS = 50

TOPIC_CLAIMS = {
    "venezuela": "The U.S. capture of Maduro was justified.",
    "iran":      "U.S. military action against Iran is justified.",
    "russia":    "Russia's invasion of Ukraine is justified.",
}

# ── Temporal calibration ───────────────────────────────────────────────────────
# window_size_days / step_size_days: from CASE_PARAMS in run_pipeline.py.
# n_windows_total: set to None to derive from data with no assertion, or fill in
# after a first run to guard against unexpected data changes (assertion tolerance ±12%).
CASE_WINDOWS: dict[str, dict] = {
    "venezuela": {"window_size_days": 8 / 24, "step_size_days": 4 / 24, "n_windows_total": None},
    "iran":      {"window_size_days": 1.0,    "step_size_days": 8 / 24, "n_windows_total": None},
    "russia":    {"window_size_days": 7.0,    "step_size_days": 1.0,    "n_windows_total": None},
}

# ── Typography (paper-matched, serif, 7-8 pt) ──────────────────────────────────
FONT_FAMILY = "serif"
TICK_SIZE   = 7
LABEL_SIZE  = 8
TITLE_SIZE  = 8
ANNOT_SIZE  = 7

# Region boundaries (mirrors stance_geometry.py)
_C_THRESH  = 0.40
_PN_THRESH = 0.50

# Post-id aliases for posts_repr.parquet
_POST_ID_ALIASES = ("Resource Id", "tweet_id", "tweetid", "post id", "postid", "id")

CASE_DISPLAY = {
    "venezuela": "Venezuela",
    "iran":      "Iran",
    "russia":    "Russia–Ukraine",
}


# ── Generic helpers ────────────────────────────────────────────────────────────

def _trunc(s: str, n: int = LABEL_MAX_CHARS) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _theme_label(row) -> str:
    if pd.notna(row.get("theme")):
        return _trunc(str(row["theme"]))
    return f"Cluster {int(row['global_cluster_id'])}"


def _norm_post_id(df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns:
        for alias in _POST_ID_ALIASES:
            if alias in df.columns:
                return df.rename(columns={alias: "post_id"})
    return df


def _setup_rcparams() -> None:
    matplotlib.rcParams.update({
        "font.family":       FONT_FAMILY,
        "font.size":         TICK_SIZE,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })


def _savefig_dual(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    """Write vector PDF + 300 dpi PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved → {out_dir / stem}.{{pdf,png}}")


def _style_ax(ax: plt.Axes, is_left: bool) -> None:
    """Apply hairline-spine / light-horizontal-grid style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.93", lw=0.35, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_SIZE, length=2.5, pad=2)


# ── Data loading (original figures) ───────────────────────────────────────────

def load_themes(case: str) -> pd.DataFrame:
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
    df = _norm_post_id(df)
    df["post_id"]    = df["post_id"].astype(str)
    df["window_dt"]  = pd.to_datetime(df["window"], format="%Y-%m-%d-%H", utc=True, errors="coerce")
    df["is_noise"]   = df["is_noise"].astype(bool)
    df["global_cluster_id"] = df["global_cluster_id"].where(~df["is_noise"])

    themes = load_themes(case)
    if not themes.empty:
        df["global_cluster_id_int"] = df["global_cluster_id"].astype("Int64")
        df = df.merge(
            themes.rename(columns={"global_cluster_id": "global_cluster_id_int"}),
            on="global_cluster_id_int", how="left",
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


# ── Data loading (drift / persistence figures) ─────────────────────────────────

def load_cluster_stance(case: str) -> pd.DataFrame:
    """
    Load cluster_stance.parquet. Returns columns including n_posts, proportions,
    C, region, and theme if present.
    """
    path = Path("data/evaluated") / case / "cluster_stance.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["global_cluster_id"])
    df = pd.read_parquet(path)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    df = df.rename(columns={
        "support_pct": "p_support",
        "oppose_pct":  "p_oppose",
        "neutral_pct": "p_neutral",
    })
    if all(c in df.columns for c in ("p_support", "p_oppose", "p_neutral")):
        df["C"] = (2.0 * np.minimum(df["p_support"], df["p_oppose"])).clip(lower=0.0)
        df["region"] = "contested"
        df.loc[(df["C"] < _C_THRESH) & (df["p_neutral"] < _PN_THRESH), "region"] = "consolidated"
        df.loc[(df["C"] < _C_THRESH) & (df["p_neutral"] >= _PN_THRESH), "region"] = "fact-relaying"
    return df


def load_embeddings(case: str) -> dict[str, np.ndarray]:
    """Return {post_id: L2-normalised 768-d embedding} from posts_repr.parquet."""
    path = Path("data/processed") / case / "posts_repr.parquet"
    if not path.exists():
        print(f"  [{case}] posts_repr.parquet not found — drift metrics skipped")
        return {}
    df = pd.read_parquet(path, columns=["post_id", "embedding"])
    df = _norm_post_id(df)
    emb_map: dict[str, np.ndarray] = {}
    for pid, emb in zip(df["post_id"].astype(str), df["embedding"]):
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        if n > 1e-9:
            emb_map[pid] = arr / n
    print(f"  [{case}] {len(emb_map):,} embeddings loaded")
    return emb_map


# ── Geometry ───────────────────────────────────────────────────────────────────

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _angular_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Angular distance in radians ∈ [0, π] between two L2-normalised vectors."""
    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


# ── Per-cluster metric computation ─────────────────────────────────────────────

def compute_cluster_metrics(case: str, gc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster drift and persistence metrics for one case.

    Drift (requires posts_repr.parquet embeddings):
      cumulative_path   — sum of angular distances between consecutive per-window centroids
      net_displacement  — angular distance from first to last centroid
      n_active_windows  — windows with a valid centroid (≥1 mapped post)
      drift_rate        — cumulative_path / n_active_windows (per-window mean displacement)
      directedness      — net_displacement / cumulative_path, bounded [0, 1]

    Persistence (from global_clusters.parquet):
      persistence_windows — unique windows the cluster is active
      persistence_frac    — persistence_windows / n_windows_total (calendar-comparable)
      persistence_days    — persistence_windows × step_size_days

    Returns a DataFrame keyed on global_cluster_id, also including n_posts, stance
    proportions, C, region from cluster_stance.parquet if available.
    """
    cfg         = CASE_WINDOWS.get(case, {})
    step_days   = cfg.get("step_size_days", 1.0)
    n_configured = cfg.get("n_windows_total")

    all_windows     = gc_df["window"].dropna().unique()
    n_windows_total = int(len(all_windows))

    if n_configured is not None:
        tol = max(2, int(round(0.12 * n_configured)))
        if abs(n_windows_total - n_configured) > tol:
            raise AssertionError(
                f"[{case}] n_windows_total mismatch: data has {n_windows_total}, "
                f"config says {n_configured} (tolerance ±{tol}). "
                f"Update CASE_WINDOWS['n_windows_total'] or investigate the data."
            )
    print(f"  [{case}] {n_windows_total} windows  step={step_days:.4f} days "
          f"({step_days * 24:.1f} h)")

    active = gc_df[~gc_df["is_noise"] & gc_df["global_cluster_id"].notna()].copy()
    active["global_cluster_id"] = active["global_cluster_id"].astype(int)
    active["post_id"]           = active["post_id"].astype(str)

    # ── Persistence ────────────────────────────────────────────────────────────
    pers = (
        active.groupby("global_cluster_id")["window"]
        .nunique()
        .reset_index(name="persistence_windows")
    )
    pers["persistence_frac"] = pers["persistence_windows"] / n_windows_total
    pers["persistence_days"] = pers["persistence_windows"] * step_days

    # ── Drift ──────────────────────────────────────────────────────────────────
    emb_map = load_embeddings(case)

    drift_rows: list[dict] = []
    if emb_map:
        n_clusters = active["global_cluster_id"].nunique()
        for i, (cid, cdf) in enumerate(active.groupby("global_cluster_id")):
            windows = sorted(cdf["window"].unique())
            if len(windows) < 2:
                continue

            centroids: list[np.ndarray] = []
            for w in windows:
                post_ids = cdf[cdf["window"] == w]["post_id"].tolist()
                embs = [emb_map[p] for p in post_ids if p in emb_map]
                if not embs:
                    continue
                centroids.append(_l2norm(np.mean(np.vstack(embs), axis=0)))

            if len(centroids) < 2:
                continue

            cum_path = float(sum(
                _angular_dist(centroids[j], centroids[j + 1])
                for j in range(len(centroids) - 1)
            ))
            net_disp     = _angular_dist(centroids[0], centroids[-1])
            n_active     = len(centroids)
            drift_rate   = cum_path / n_active
            directedness = float(min(1.0, net_disp / cum_path)) if cum_path > 1e-9 else 0.0

            drift_rows.append({
                "global_cluster_id": cid,
                "n_active_windows":  n_active,
                "cumulative_path":   cum_path,
                "net_displacement":  net_disp,
                "drift_rate":        drift_rate,
                "directedness":      directedness,
            })

        print(f"  [{case}] drift computed for {len(drift_rows)}/{n_clusters} clusters")

    del emb_map  # release memory

    drift_df = pd.DataFrame(drift_rows) if drift_rows else pd.DataFrame(columns=[
        "global_cluster_id", "n_active_windows", "cumulative_path",
        "net_displacement", "drift_rate", "directedness",
    ])

    metrics = pers.merge(drift_df, on="global_cluster_id", how="left")

    # ── Stance, n_posts, C, region ─────────────────────────────────────────────
    stance = load_cluster_stance(case)
    if not stance.empty:
        keep = ["global_cluster_id"] + [
            c for c in ("n_posts", "p_support", "p_oppose", "p_neutral", "C", "region", "theme")
            if c in stance.columns
        ]
        metrics = metrics.merge(stance[keep], on="global_cluster_id", how="left")

    if "n_posts" not in metrics.columns:
        # Fall back to counting unique posts in global_clusters
        n_posts_gc = (
            active.groupby("global_cluster_id")["post_id"]
            .nunique()
            .reset_index(name="n_posts")
        )
        metrics = metrics.merge(n_posts_gc, on="global_cluster_id", how="left")

    # Theme: if not in stance parquet, pull from gc_df
    if "theme" not in metrics.columns and "theme" in active.columns:
        t = active.groupby("global_cluster_id")["theme"].first().reset_index()
        metrics = metrics.merge(t, on="global_cluster_id", how="left")

    metrics["case"]            = case
    metrics["n_windows_total"] = n_windows_total
    return metrics


# ── Displacement artifact investigation ────────────────────────────────────────

def investigate_displacement(metrics_df: pd.DataFrame, case: str) -> None:
    """
    Report on the distribution of net_displacement (angular distance in radians).

    Flags suspicious mass at exactly 1.0 rad: in a positive-orthant embedding
    space cosine similarity is bounded to [0, 1], making cosine distance = 1 - cos ∈ [0, 1]
    with 1.0 meaning perfectly orthogonal centroids. If the script's arccos formula
    is correct and embeddings are truly L2-normalised the ceiling is π ≈ 3.14, so
    mass at 1.0 rad (≈57°) is a genuine value, not a clipping artifact. Mass at
    exactly π ≈ 3.14 (antipodal centroids) would be suspicious.
    """
    nd = metrics_df["net_displacement"].dropna()
    if nd.empty:
        print(f"  [{case}] No drift data — displacement investigation skipped")
        return

    print(f"\n  [{case}] net_displacement (radians):")
    print(f"    min={nd.min():.4f}  max={nd.max():.4f}  "
          f"mean={nd.mean():.4f}  median={nd.median():.4f}")

    n_at_zero    = int((nd < 0.01).sum())
    n_near_one   = int(((nd - 1.0).abs() < 0.02).sum())
    n_near_half_pi = int(((nd - np.pi / 2).abs() < 0.02).sum())
    n_near_pi    = int((nd > np.pi - 0.02).sum())
    frac_above_2 = float((nd > 2.0).sum()) / max(len(nd), 1)

    print(f"    ≈0.00 (static):          {n_at_zero}")
    print(f"    ≈1.00 rad (57.3°):       {n_near_one}   ← primary concern")
    print(f"    ≈π/2  rad (90.0°):       {n_near_half_pi}")
    print(f"    ≈π    rad (anti-podal):  {n_near_pi}")
    print(f"    >2.0 rad ({frac_above_2:.1%} of clusters)")

    if n_near_pi > 0:
        print(f"    ⚠  {n_near_pi} clusters have near-antipodal centroids. "
              f"This suggests an embedding sign-flip or corpus split artefact.")
    if n_near_one > 5:
        print(f"    NOTE: mass at ~1.0 rad is geometrically valid if embeddings "
              f"span a positive orthant (many transformer models clip to [0,·] "
              f"via pooling). It is NOT a clipping artefact from the arccos formula "
              f"(max of that formula is π when both norms are truly unit).")

    # Drift regime statistics (also printed here per spec)
    dr = metrics_df["drift_rate"].dropna()
    di = metrics_df["directedness"].dropna()
    if not dr.empty:
        n_osc  = int((di < 0.3).sum())
        n_dir  = int((di > 0.7).sum())
        print(f"\n  [{case}] drift_rate: median={dr.median():.4f} rad/window  "
              f"directedness: median={di.median():.4f}")
        print(f"    oscillating (directedness<0.3): {n_osc}   "
              f"directed (directedness>0.7): {n_dir}")


# ── Persistence residuals ──────────────────────────────────────────────────────

def add_persistence_residuals(metrics_by_case: dict[str, pd.DataFrame]) -> None:
    """
    Add standardised OLS residuals of persistence_frac ~ log10(n_posts) to each
    metrics DataFrame. Modifies in place. Used by both the persistence figure and
    the CSV output.
    """
    for case, mdf in metrics_by_case.items():
        mdf["persistence_residual"] = np.nan
        valid = mdf.dropna(subset=["n_posts", "persistence_frac"])
        valid = valid[valid["n_posts"] >= 1]
        if len(valid) < 3:
            continue
        log_n = np.log10(valid["n_posts"].clip(lower=1).values.astype(float))
        pf    = valid["persistence_frac"].values.astype(float)
        slope, intercept, r, *_ = linregress(log_n, pf)
        y_pred    = slope * log_n + intercept
        residuals = pf - y_pred
        std_r     = residuals.std()
        std_res   = (residuals - residuals.mean()) / std_r if std_r > 1e-9 else np.zeros_like(residuals)
        mdf.loc[valid.index, "persistence_residual"] = std_res


# ── Per-window summary helper ──────────────────────────────────────────────────

def window_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wdt, wdf in df.groupby("window_dt"):
        total  = len(wdf)
        noise  = wdf["is_noise"].sum()
        active = wdf.loc[~wdf["is_noise"], "global_cluster_id"].nunique()
        rows.append({"window_dt": wdt, "total": total, "noise": noise, "active_clusters": active})
    out = pd.DataFrame(rows).sort_values("window_dt").reset_index(drop=True)
    out["noise_frac"] = out["noise"] / out["total"].replace(0, np.nan)
    t0, t1 = out["window_dt"].min(), out["window_dt"].max()
    span   = (t1 - t0).total_seconds()
    out["t_norm"] = (out["window_dt"] - t0).dt.total_seconds() / (span if span > 0 else 1)
    return out


def cluster_lifespan_df(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
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

    theme_col = (
        active.groupby("global_cluster_id")["theme"].first().reset_index()
        if "theme" in active.columns
        else pd.DataFrame(columns=["global_cluster_id", "theme"])
    )
    lifespan = lifespan.merge(theme_col, on="global_cluster_id", how="left")
    lifespan["label"] = lifespan.apply(_theme_label, axis=1)
    return lifespan.sort_values("n_windows", ascending=False).head(top_n).reset_index(drop=True)


# ── Original figure functions (unchanged logic, updated to save PDF+PNG) ───────

def plot_narrative_lifecycles(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
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
        ax.set_title(f"{case.capitalize()} — top 20 clusters by lifespan", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}h"))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Windows alive", shrink=0.6, pad=0.02)

    fig.suptitle("Narrative lifecycles by case", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _savefig_dual(fig, "narrative_lifecycles", out_dir)
    plt.close(fig)


def plot_noise_over_time(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, df in data.items():
        ws    = window_summary(df)
        color = COLORS[case]
        ax.scatter(ws["t_norm"], ws["noise_frac"], s=8, alpha=0.25, color=color)
        if len(ws) >= 5:
            k = max(3, len(ws) // 15)
            smoothed = uniform_filter1d(ws["noise_frac"].fillna(0).values, size=k, mode="nearest")
            ax.plot(ws["t_norm"], smoothed, color=color, linewidth=2, label=CASE_DISPLAY.get(case, case))
        else:
            ax.plot(ws["t_norm"], ws["noise_frac"], color=color, linewidth=2, label=CASE_DISPLAY.get(case, case))

    ax.set_xlabel("Normalised time (0 = start, 1 = end)", fontsize=10)
    ax.set_ylabel("Noise fraction", fontsize=10)
    ax.set_title("Noise fraction over time", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig_dual(fig, "noise_over_time", out_dir)
    plt.close(fig)


def plot_cluster_count_over_time(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, df in data.items():
        ws    = window_summary(df)
        color = COLORS[case]
        ax.scatter(ws["t_norm"], ws["active_clusters"], s=8, alpha=0.25, color=color)
        if len(ws) >= 5:
            k = max(3, len(ws) // 15)
            smoothed = uniform_filter1d(ws["active_clusters"].fillna(0).values, size=k, mode="nearest")
            ax.plot(ws["t_norm"], smoothed, color=color, linewidth=2, label=CASE_DISPLAY.get(case, case))
        else:
            ax.plot(ws["t_norm"], ws["active_clusters"], color=color, linewidth=2, label=CASE_DISPLAY.get(case, case))

    ax.set_xlabel("Normalised time (0 = start, 1 = end)", fontsize=10)
    ax.set_ylabel("Active clusters per window", fontsize=10)
    ax.set_title("Active cluster count over time", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig_dual(fig, "cluster_count_over_time", out_dir)
    plt.close(fig)


def plot_lifespan_distribution(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for case, df in data.items():
        color  = COLORS[case]
        active = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
        active["global_cluster_id"] = active["global_cluster_id"].astype(int)
        vals = active.groupby("global_cluster_id")["window_dt"].nunique().values
        if len(vals) == 0:
            continue

        log_max = np.log10(max(vals.max(), 2))
        bins    = np.logspace(0, log_max, 30)
        ax.hist(vals, bins=bins, density=True, alpha=0.45, color=color,
                label=f"{CASE_DISPLAY.get(case, case)} (n={len(vals)})")

        if len(vals) >= 3:
            log_vals = np.log10(vals.clip(1))
            kde      = gaussian_kde(log_vals, bw_method=0.4)
            x_log   = np.linspace(0, log_max, 300)
            y_kde   = kde(x_log)
            ax.plot(10 ** x_log, y_kde / (10 ** x_log * np.log(10)), color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Cluster lifespan (windows)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Cluster lifespan distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    _savefig_dual(fig, "lifespan_distribution", out_dir)
    plt.close(fig)


def plot_top_narratives(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    cases = list(data.keys())
    n     = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 8), squeeze=False)

    for ax, case in zip(axes[0], cases):
        color    = COLORS[case]
        lifespan = cluster_lifespan_df(data[case], top_n=15)
        lifespan_plot = lifespan.iloc[::-1].reset_index(drop=True)

        bars = ax.barh(
            y=range(len(lifespan_plot)),
            width=lifespan_plot["n_windows"],
            height=0.7, color=color, alpha=0.85,
        )
        for bar, nw in zip(bars, lifespan_plot["n_windows"]):
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    str(int(nw)), va="center", fontsize=8, color="0.3")

        ax.set_yticks(range(len(lifespan_plot)))
        ax.set_yticklabels(lifespan_plot["label"].tolist(), fontsize=8)
        ax.set_xlabel("Lifespan (windows)", fontsize=9)
        ax.set_title(f"{CASE_DISPLAY.get(case, case)} — top 15 narratives", fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, lifespan_plot["n_windows"].max() * 1.15)

    fig.suptitle("Top narratives by lifespan", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig_dual(fig, "top_narratives_per_case", out_dir)
    plt.close(fig)


def plot_birth_death_rate(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    cases = list(data.keys())
    n     = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), squeeze=False, sharex=True)

    for ax, case in zip(axes[:, 0], cases):
        df    = data[case]
        color = COLORS[case]

        active = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
        active["global_cluster_id"] = active["global_cluster_id"].astype(int)

        births = active.groupby("global_cluster_id")["window_dt"].min().rename("window_dt").reset_index()
        deaths = active.groupby("global_cluster_id")["window_dt"].max().rename("window_dt").reset_index()

        all_windows = df["window_dt"].dropna().sort_values().unique()
        t0, t1 = all_windows.min(), all_windows.max()
        span   = (t1 - t0).total_seconds()

        def to_norm(ts):
            return (ts - t0).total_seconds() / (span if span > 0 else 1)

        win_norm     = np.array([to_norm(w) for w in all_windows])
        birth_counts = births["window_dt"].value_counts().reindex(all_windows, fill_value=0).values
        death_counts = deaths["window_dt"].value_counts().reindex(all_windows, fill_value=0).values

        if len(win_norm) >= 5:
            k = max(3, len(win_norm) // 15)
            birth_smooth = uniform_filter1d(birth_counts.astype(float), size=k, mode="nearest")
            death_smooth = uniform_filter1d(death_counts.astype(float), size=k, mode="nearest")
        else:
            birth_smooth = birth_counts.astype(float)
            death_smooth = death_counts.astype(float)

        ax.bar(win_norm, birth_counts, width=1 / max(len(win_norm), 1), alpha=0.2, color=color)
        ax.bar(win_norm, -death_counts, width=1 / max(len(win_norm), 1), alpha=0.2, color="0.4")
        ax.plot(win_norm, birth_smooth, color=color, linewidth=2, label="Births")
        ax.plot(win_norm, -death_smooth, color="0.4", linewidth=2, linestyle="--", label="Deaths")
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_ylabel("Clusters / window", fontsize=9)
        ax.set_title(CASE_DISPLAY.get(case, case), fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: str(int(abs(y)))))

    axes[-1, 0].set_xlabel("Normalised time (0 = start, 1 = end)", fontsize=10)
    fig.suptitle("Narrative birth and death rates over time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig_dual(fig, "narrative_birth_death_rate", out_dir)
    plt.close(fig)


def plot_stance_over_time(out_dir: Path) -> None:
    available = []
    for case in CASES:
        p = Path("data/evaluated") / case / "topic_stance_by_window.parquet"
        if p.exists():
            available.append(case)
        else:
            print(f"  [stance_over_time] skip {case}: topic_stance_by_window.parquet not found")

    if not available:
        print("  [stance_over_time] no data — skipping")
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n), squeeze=False)

    for ax, case in zip(axes[:, 0], available):
        color        = COLORS[case]
        oppose_color = OPPOSE_COLORS[case]
        claim        = TOPIC_CLAIMS.get(case, "")

        df = pd.read_parquet(Path("data/evaluated") / case / "topic_stance_by_window.parquet")
        df["window_dt"] = pd.to_datetime(df["window"], format="%Y-%m-%d-%H", utc=True, errors="coerce")
        df = df.dropna(subset=["window_dt"]).sort_values("window_dt").reset_index(drop=True)

        if df.empty:
            ax.set_visible(False)
            continue

        smooth  = df[["support_pct","neutral_pct","oppose_pct"]].rolling(3, min_periods=1, center=True).mean()
        # to_pydatetime() returns ndarray; convert to list to avoid pandas 2.x
        # label-based indexing issues (series[-1] raises KeyError since 2.0).
        dates   = list(df["window_dt"].dt.to_pydatetime())

        ax.stackplot(
            dates,
            smooth["support_pct"].values * 100,
            smooth["neutral_pct"].values * 100,
            smooth["oppose_pct"].values  * 100,
            labels=["Support", "Neutral", "Oppose"],
            colors=[color, "#aaaaaa", oppose_color],
            alpha=0.85,
        )
        ax.axhline(50, color="white", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylabel("Share of posts (%)", fontsize=9)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax.set_xlim(dates[0], dates[-1])
        title = f"{CASE_DISPLAY.get(case, case)}"
        if claim:
            title += f'\n"{claim}"'
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6, ncol=3)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1, 0].set_xlabel("Date", fontsize=10)
    fig.suptitle("Stance toward topic claim over time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig_dual(fig, "stance_over_time", out_dir)
    plt.close(fig)


# ── Figure 8: Normalised drift scatter ────────────────────────────────────────

def plot_normalized_drift(
    metrics_by_case: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    1×3 faceted scatter: drift_rate (x) vs net_displacement (y) per case.
    Both axes in radians. Shared limits. Bubble area ∝ sqrt(n_posts).

    Three regimes separated by the y=x reference line:
      · Lower-right (high drift_rate, low net_displacement): oscillating — wanders and returns
      · Upper-left  (low drift_rate, high net_displacement): improbable (net ≤ path, always)
      · Near diagonal:                                        directed semantic transformation
    The y=x diagonal marks where net_displacement = drift_rate (directedness=1 for 1-step clusters).
    Points above the diagonal have high net displacement relative to per-window rate,
    indicating consistent directionality across windows.
    """
    _setup_rcparams()

    # Determine shared axis bounds from all available data (99.5th pct, capped at π)
    all_dr = pd.concat([
        m["drift_rate"].dropna() for m in metrics_by_case.values() if "drift_rate" in m
    ], ignore_index=True)
    all_nd = pd.concat([
        m["net_displacement"].dropna() for m in metrics_by_case.values() if "net_displacement" in m
    ], ignore_index=True)

    if all_dr.empty or all_nd.empty:
        print("  [drift] No drift data across cases — skipping figure")
        return

    xy_max = min(
        max(all_dr.quantile(0.995), all_nd.quantile(0.995)) * 1.12,
        np.pi,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.8),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(wspace=0.06, left=0.11, right=0.97, top=0.87, bottom=0.18)

    for col_idx, (ax, case) in enumerate(zip(axes, CASES)):
        is_left  = col_idx == 0
        is_right = col_idx == 2

        if case not in metrics_by_case:
            ax.set_visible(False)
            continue

        mdf = metrics_by_case[case].dropna(subset=["drift_rate", "net_displacement"]).copy()
        if mdf.empty:
            ax.text(0.5, 0.5, "no drift data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=ANNOT_SIZE, color="0.5")
            ax.set_title(CASE_DISPLAY.get(case, case), fontsize=TITLE_SIZE, pad=3)
            _style_ax(ax, is_left)
            continue

        # ── Reference diagonal (directedness=1, n_steps=1 case) ──────────────
        diag_x = np.array([0, xy_max])
        ax.plot(diag_x, diag_x, "--", color="0.55", lw=0.8, zorder=1)

        # Region annotations on leftmost panel only
        if is_left:
            mid = xy_max * 0.5
            ax.text(mid * 1.25, mid * 0.45, "oscillating",
                    fontsize=ANNOT_SIZE, color="0.50", style="italic",
                    ha="center", va="center", rotation=0, zorder=2)
            ax.text(mid * 0.55, mid * 1.1, "directed",
                    fontsize=ANNOT_SIZE, color="0.50", style="italic",
                    ha="center", va="center", rotation=0, zorder=2)

        # ── Bubble sizes ∝ sqrt(n_posts) ──────────────────────────────────────
        n_posts = mdf.get("n_posts", pd.Series(10, index=mdf.index)).fillna(10).clip(lower=1)
        sizes   = np.clip(2.5 * np.sqrt(n_posts.values), 4, 200)

        ax.scatter(
            mdf["drift_rate"], mdf["net_displacement"],
            s=sizes,
            c=COLORS.get(case, "#4878CF"),
            alpha=0.50, linewidths=0.3,
            edgecolors=tuple(np.array(matplotlib.colors.to_rgb(COLORS.get(case, "#4878CF"))) * 0.60),
            zorder=3,
        )

        ax.set_xlim(0, xy_max)
        ax.set_ylim(0, xy_max)
        _style_ax(ax, is_left)

        ax.set_title(CASE_DISPLAY.get(case, case), fontsize=TITLE_SIZE, pad=3)
        ax.set_xlabel("Drift rate (rad / window)", fontsize=LABEL_SIZE)
        if is_left:
            ax.set_ylabel("Net displacement (rad)", fontsize=LABEL_SIZE)

    fig.suptitle("Normalised centroid drift: rate vs. net displacement",
                 fontsize=TITLE_SIZE + 0.5, y=0.99)
    _savefig_dual(fig, "drift_normalized", out_dir)
    plt.close(fig)


# ── Figure 9: Normalised persistence scatter ───────────────────────────────────

def plot_normalized_persistence(
    metrics_by_case: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    1×3 scatter: log10(n_posts) (x) vs persistence_frac (y), y ∈ [0, 1] shared.
    OLS fit line with slope and R² annotated. Top-3 positive and negative
    standardised residuals labeled per panel (long-lived/short-lived relative to size).
    Secondary y-axis in days on rightmost panel.
    """
    _setup_rcparams()

    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.8),
        sharey=True, sharex=False,
    )
    fig.subplots_adjust(wspace=0.10, left=0.11, right=0.94, top=0.87, bottom=0.18)

    for col_idx, (ax, case) in enumerate(zip(axes, CASES)):
        is_left  = col_idx == 0
        is_right = col_idx == 2

        if case not in metrics_by_case:
            ax.set_visible(False)
            continue

        mdf = metrics_by_case[case].copy()
        mdf = mdf[mdf["n_posts"].notna() & mdf["persistence_frac"].notna()].copy()
        mdf = mdf[mdf["n_posts"] >= 1].copy()
        mdf["log_n"] = np.log10(mdf["n_posts"].clip(lower=1))

        # ── Scatter ───────────────────────────────────────────────────────────
        ax.scatter(
            mdf["log_n"], mdf["persistence_frac"],
            s=9, c=COLORS.get(case, "#4878CF"),
            alpha=0.45, linewidths=0.25,
            edgecolors=tuple(np.array(matplotlib.colors.to_rgb(COLORS.get(case, "#4878CF"))) * 0.60),
            zorder=3,
        )

        # ── OLS fit ───────────────────────────────────────────────────────────
        valid = mdf.dropna(subset=["log_n", "persistence_frac"])
        if len(valid) >= 3:
            slope, intercept, r, *_ = linregress(
                valid["log_n"].values, valid["persistence_frac"].values
            )
            r2    = r ** 2
            x_fit = np.linspace(valid["log_n"].min(), valid["log_n"].max(), 200)
            y_fit = np.clip(slope * x_fit + intercept, 0, 1)
            ax.plot(x_fit, y_fit, color="0.20", lw=1.0, zorder=4, alpha=0.85)
            ax.annotate(
                f"$b$ = {slope:+.3f}\n$R^2$ = {r2:.2f}",
                xy=(0.97, 0.05), xycoords="axes fraction",
                fontsize=ANNOT_SIZE, ha="right", va="bottom", color="0.30",
            )

            # ── Residual labels (top 3 positive + top 3 negative) ─────────────
            if "persistence_residual" in mdf.columns:
                res_col = "persistence_residual"
            else:
                # Compute inline if add_persistence_residuals wasn't called
                y_pred = slope * mdf["log_n"] + intercept
                mdf["_resid"] = mdf["persistence_frac"] - y_pred
                std_r = mdf["_resid"].std()
                mdf["persistence_residual"] = (mdf["_resid"] - mdf["_resid"].mean()) / std_r if std_r > 1e-9 else 0.0
                res_col = "persistence_residual"

            top_pos = mdf.nlargest(3, res_col)
            top_neg = mdf.nsmallest(3, res_col)
            to_label = pd.concat([top_pos, top_neg], ignore_index=True)

            theme_col = "theme" if "theme" in mdf.columns else None
            texts = []
            for _, row in to_label.iterrows():
                if theme_col and pd.notna(row.get(theme_col)):
                    lbl = str(row[theme_col])[:28]
                else:
                    lbl = f"C{int(row['global_cluster_id'])}"
                is_pos = float(row[res_col]) > 0
                t = ax.text(
                    float(row["log_n"]), float(row["persistence_frac"]),
                    lbl,
                    fontsize=max(ANNOT_SIZE - 1, 5),
                    ha="center", va="bottom",
                    color="0.15" if is_pos else "0.50",
                    zorder=5,
                )
                texts.append(t)

            try:
                from adjustText import adjust_text
                adjust_text(
                    texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.65", lw=0.35),
                    only_move={"texts": "xy"},
                    force_text=(0.6, 1.0),
                    expand_points=(1.2, 1.4),
                )
            except ImportError:
                pass  # labels may overlap; pip install adjustText to fix

        # ── Panel style ───────────────────────────────────────────────────────
        ax.set_ylim(0, 1)
        _style_ax(ax, is_left)

        ax.set_title(CASE_DISPLAY.get(case, case), fontsize=TITLE_SIZE, pad=3)
        ax.set_xlabel("log₁₀(posts)", fontsize=LABEL_SIZE)
        if is_left:
            ax.set_ylabel("Persistence fraction", fontsize=LABEL_SIZE)

        # ── Secondary y-axis (days) on rightmost panel ────────────────────────
        if is_right and case in metrics_by_case:
            n_wt      = int(metrics_by_case[case]["n_windows_total"].iloc[0])
            step_days = float(CASE_WINDOWS.get(case, {}).get("step_size_days", 1.0))
            max_days  = n_wt * step_days

            ax2 = ax.twinx()
            ax2.set_ylim(0, max_days)
            ax2.set_ylabel("Days", fontsize=LABEL_SIZE)
            ax2.tick_params(labelsize=TICK_SIZE, length=2.5, pad=2)
            ax2.spines["top"].set_visible(False)

    fig.suptitle("Normalised persistence: fraction of case span vs. cluster size",
                 fontsize=TITLE_SIZE + 0.5, y=0.99)
    _savefig_dual(fig, "persistence_normalized", out_dir)
    plt.close(fig)


# ── CSV output ─────────────────────────────────────────────────────────────────

def write_metrics_csv(metrics_by_case: dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Write cluster_metrics.csv joining drift, persistence, stance, and region per cluster.
    Keyed on (case, global_cluster_id) for cross-case joins.
    """
    all_frames = list(metrics_by_case.values())
    if not all_frames:
        return
    out = pd.concat(all_frames, ignore_index=True)

    ordered = [
        "case", "global_cluster_id",
        "persistence_windows", "persistence_frac", "persistence_days", "persistence_residual",
        "n_active_windows", "cumulative_path", "net_displacement", "drift_rate", "directedness",
        "n_posts",
        "p_support", "p_oppose", "p_neutral", "C", "region",
        "theme",
        "n_windows_total",
    ]
    cols = [c for c in ordered if c in out.columns]
    out  = out[cols].sort_values(["case", "global_cluster_id"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Metrics CSV → {out_path}  ({len(out):,} rows, {len(out.columns)} cols)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading global_clusters.parquet (+ cluster_themes.parquet) for all cases...")
    data = load_all()

    if not data:
        print("No data found for any case — nothing to plot.")
        return
    print(f"Loaded cases: {list(data.keys())}\n")

    # ── Original figures (1–7) ─────────────────────────────────────────────────
    plot_narrative_lifecycles(data,  FIGURES_DIR)
    plot_noise_over_time(data,       FIGURES_DIR)
    plot_cluster_count_over_time(data, FIGURES_DIR)
    plot_lifespan_distribution(data, FIGURES_DIR)
    plot_top_narratives(data,        FIGURES_DIR)
    plot_birth_death_rate(data,      FIGURES_DIR)
    plot_stance_over_time(          FIGURES_DIR)

    # ── Per-cluster metrics (drift + persistence) ──────────────────────────────
    print("\nComputing per-cluster drift and persistence metrics...")
    metrics_by_case: dict[str, pd.DataFrame] = {}
    for case, df in data.items():
        if case not in CASE_WINDOWS:
            print(f"  [{case}] not in CASE_WINDOWS — skipped")
            continue
        try:
            mdf = compute_cluster_metrics(case, df)
            if mdf is not None and len(mdf):
                metrics_by_case[case] = mdf
        except AssertionError as e:
            print(f"  [{case}] ASSERTION ERROR: {e}")
        except Exception as e:
            print(f"  [{case}] ERROR: {e}")

    # ── Displacement investigation ─────────────────────────────────────────────
    if metrics_by_case:
        print("\n── Displacement artifact investigation ──────────────────────────")
        for case, mdf in metrics_by_case.items():
            investigate_displacement(mdf, case)

        # Add standardised persistence residuals to all DataFrames (used in figure + CSV)
        add_persistence_residuals(metrics_by_case)

        # ── Figures 8 and 9 ───────────────────────────────────────────────────
        print()
        plot_normalized_drift(metrics_by_case,       FIGURES_DIR)
        plot_normalized_persistence(metrics_by_case, FIGURES_DIR)
        write_metrics_csv(metrics_by_case,           FIGURES_DIR / "cluster_metrics.csv")

    print("\nAll figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
