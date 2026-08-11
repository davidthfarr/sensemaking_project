"""
Faceted bubble scatter — stance geometry of global clusters across conflict cases.

Reads cluster_stance.parquet (support/oppose/neutral proportions) and
global_clusters.parquet (for persistence) from the pipeline's evaluated output.

Produces
--------
  <output_dir>/stance_geometry.pdf        vector PDF for paper (use \\begin{figure*})
  <output_dir>/stance_geometry.png        300 dpi draft PNG
  <output_dir>/stance_geometry_stats.csv  per-case/region cluster & post-volume counts

Usage
-----
  python analysis/stance_geometry.py
  python analysis/stance_geometry.py --color-by persistence
  python analysis/stance_geometry.py --annotate-top-k 5
  python analysis/stance_geometry.py --output-dir paper/figures
  python analysis/stance_geometry.py --c-thresh 0.35 --pn-thresh 0.55 --min-posts 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Module-level constants (tune here before sweeping via CLI) ────────────────
S_SCALE   = 50      # pt² per log10 unit; adjust so min ≈4 pt², max ≈200 pt²
C_THRESH  = 0.40    # horizontal boundary: C >= C_THRESH → contested
PN_THRESH = 0.50    # vertical boundary: p_neutral >= PN_THRESH → fact-relaying
MIN_POSTS = 10      # clusters below this render dimly as background texture

FILL_COLOR = "#4878CF"   # single fill color across all panels
FIG_WIDTH  = 7.0         # inches (full two-column text width for figure*)
FIG_HEIGHT = 2.6         # inches

# Case display order and panel titles
CASES = ["venezuela", "iran", "russia"]
CASE_LABELS = {
    "venezuela": "Venezuela",
    "iran":      "Iran",
    "russia":    "Russia–Ukraine",
}

REGION_COLORS = {
    "contested":    "#d62728",
    "consolidated": "#1f77b4",
    "fact-relaying":"#2ca02c",
}
REGION_ALPHA  = 0.07   # very-low-alpha fill so bubbles dominate

# Typography — match AAAI/ICWSM serif body
FONT_FAMILY = "serif"
TICK_SIZE   = 7
LABEL_SIZE  = 8
TITLE_SIZE  = 8
ANNOT_SIZE  = 7

# ── Data root (overridable via --data-root CLI flag) ──────────────────────────
DATA_ROOT = Path("data/evaluated")

_POST_ID_ALIASES = ("Resource Id", "tweet_id", "tweetid", "post id", "postid", "id")


def _norm_post_id(df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns:
        for alias in _POST_ID_ALIASES:
            if alias in df.columns:
                return df.rename(columns={alias: "post_id"})
    return df


# ── Data loading ───────────────────────────────────────────────────────────────

def load_case_data(case: str) -> pd.DataFrame:
    """
    Load one case. Reads cluster_stance.parquet for proportions, global_clusters.parquet
    for persistence, and cluster_themes.parquet for theme labels.

    Falls back to aggregating post-level stance labels if pct columns are absent.
    """
    case_dir = DATA_ROOT / case

    # ── Stance proportions ─────────────────────────────────────────────────────
    stance_path = case_dir / "cluster_stance.parquet"
    if not stance_path.exists():
        raise FileNotFoundError(f"Missing {stance_path}")
    df = pd.read_parquet(stance_path)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)

    # Rename pipeline column names to generic spec names
    df = df.rename(columns={
        "support_pct": "p_support",
        "oppose_pct":  "p_oppose",
        "neutral_pct": "p_neutral",
    })

    # Fallback: aggregate from post-level stance labels if proportions are missing
    if not all(c in df.columns for c in ("p_support", "p_oppose", "p_neutral")):
        if "stance" in df.columns:
            counts = (
                df.groupby(["global_cluster_id", "stance"])
                .size().unstack(fill_value=0)
            )
            for col in ("support", "oppose", "neutral"):
                if col not in counts.columns:
                    counts[col] = 0
            totals = counts.sum(axis=1).clip(lower=1)
            df = (
                df.drop(columns=["stance"], errors="ignore")
                .drop_duplicates("global_cluster_id")
                .merge(
                    (counts[["support", "oppose", "neutral"]]
                     .div(totals, axis=0)
                     .rename(columns={"support":"p_support","oppose":"p_oppose","neutral":"p_neutral"})
                     .reset_index()),
                    on="global_cluster_id", how="left",
                )
            )
        else:
            raise ValueError(
                f"[{case}] No p_support/p_oppose/p_neutral columns and no raw 'stance' column."
            )

    # ── Persistence (unique windows per global cluster) ───────────────────────
    gc_path = case_dir / "global_clusters.parquet"
    if gc_path.exists():
        gc = _norm_post_id(pd.read_parquet(gc_path))
        gc = gc[gc["global_cluster_id"].notna() & ~gc["is_noise"]].copy()
        gc["global_cluster_id"] = gc["global_cluster_id"].astype(float).astype(int)
        pers = (
            gc.groupby("global_cluster_id")["window"]
            .nunique()
            .reset_index(name="persistence")
        )
        df = df.merge(pers, on="global_cluster_id", how="left")
    else:
        df["persistence"] = np.nan

    # ── Theme labels ──────────────────────────────────────────────────────────
    themes_path = case_dir / "cluster_themes.parquet"
    if themes_path.exists():
        themes = pd.read_parquet(themes_path)
        themes["global_cluster_id"] = themes["global_cluster_id"].astype(int)
        df = df.merge(themes[["global_cluster_id", "theme"]], on="global_cluster_id", how="left")
        df["theme_label"] = df["theme"].fillna("")
    elif "theme" in df.columns:
        df["theme_label"] = df["theme"].fillna("")
    else:
        df["theme_label"] = ""

    df["case"]       = case
    df["cluster_id"] = df["global_cluster_id"]
    df["n_posts"]    = df.get("n_posts", pd.Series(1, index=df.index)).fillna(1).astype(int)

    return df


def load_all(cases: list = None) -> pd.DataFrame:
    cases = cases or CASES
    frames = []
    for case in cases:
        try:
            frames.append(load_case_data(case))
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)
    if not frames:
        raise RuntimeError("No data loaded — check DATA_ROOT / data/evaluated/<case>/ paths.")
    return pd.concat(frames, ignore_index=True)


# ── Validation & derived quantities ───────────────────────────────────────────

def validate_and_compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assert stance proportions sum to 1 and C <= 1 - p_neutral.
    Fail loudly — a violation indicates broken stance aggregation upstream.
    Computes C = 2 * min(p_support, p_oppose).
    """
    df = df.copy()

    psum = df["p_support"] + df["p_oppose"] + df["p_neutral"]
    bad  = (psum - 1.0).abs() > 1e-4
    if bad.any():
        raise ValueError(
            f"Stance proportions do not sum to 1 (tol 1e-4) for "
            f"{bad.sum()} clusters:\n"
            + df.loc[bad, ["case","cluster_id","p_support","p_oppose","p_neutral"]].to_string()
        )

    df["C"] = (2.0 * np.minimum(df["p_support"], df["p_oppose"])).clip(lower=0.0)

    bad_C = df["C"] > (1.0 - df["p_neutral"]) + 1e-9
    if bad_C.any():
        raise ValueError(
            f"C > 1 - p_neutral for {bad_C.sum()} clusters — stance aggregation is wrong:\n"
            + df.loc[bad_C, ["case","cluster_id","p_support","p_oppose","p_neutral","C"]].to_string()
        )

    return df


def assign_regions(df: pd.DataFrame, c_thresh: float, pn_thresh: float) -> pd.DataFrame:
    df = df.copy()
    df["region"] = "contested"
    mask_low_C     = df["C"] < c_thresh
    mask_high_neu  = df["p_neutral"] >= pn_thresh
    df.loc[mask_low_C & ~mask_high_neu, "region"] = "consolidated"
    df.loc[mask_low_C &  mask_high_neu, "region"] = "fact-relaying"
    return df


# ── Region polygon geometry ────────────────────────────────────────────────────

def _region_polygons(c_thresh: float, pn_thresh: float) -> dict:
    """
    Return {region: np.ndarray of (p_neutral, C) polygon vertices}.
    All polygons lie within the feasible triangle C <= 1 - p_neutral.
    """
    pn_at_thresh = 1.0 - c_thresh   # p_neutral where feasible boundary = c_thresh

    # Contested: triangle above C_THRESH, below feasible boundary
    contested = np.array([
        [0.0,          c_thresh],
        [pn_at_thresh, c_thresh],
        [0.0,          1.0],
    ])

    # Consolidated: rectangle in lower-left (feasible for pn_thresh < pn_at_thresh)
    c_top_right = min(c_thresh, max(0.0, 1.0 - pn_thresh))
    consolidated = np.array([
        [0.0,          0.0],
        [pn_thresh,    0.0],
        [pn_thresh,    c_top_right],
        [0.0,          c_thresh],
    ])

    # Fact-relaying: right of pn_thresh, below c_thresh, within feasible region
    if pn_thresh <= pn_at_thresh:
        fact_relaying = np.array([
            [pn_thresh,      0.0],
            [1.0,            0.0],
            [pn_at_thresh,   c_thresh],
            [pn_thresh,      c_thresh],
        ])
    else:
        fact_relaying = np.array([
            [pn_thresh,              0.0],
            [1.0,                    0.0],
            [pn_thresh, max(0.0, 1.0 - pn_thresh)],
        ])

    return {
        "contested":    contested,
        "consolidated": consolidated,
        "fact-relaying": fact_relaying,
    }


# ── Visual helpers ─────────────────────────────────────────────────────────────

def _bubble_area(n_posts, s_scale: float = S_SCALE) -> np.ndarray:
    """pt² area for scatter s= parameter."""
    return s_scale * np.log10(np.asarray(n_posts, dtype=float).clip(min=0) + 1.0)


def _darken(hex_color: str, factor: float = 0.65) -> tuple:
    """Return a darkened RGBA tuple for edge colors."""
    return tuple(np.array(mcolors.to_rgb(hex_color)) * factor) + (1.0,)


def _region_label_positions(c_thresh: float, pn_thresh: float) -> dict:
    """Hand-tuned label anchor points for the three region polygons."""
    pn_at = 1.0 - c_thresh
    return {
        "contested":    (max(0.03, (pn_at) / 3),   (1.0 + 2*c_thresh) / 3 + 0.04),
        "consolidated": (pn_thresh / 2,              c_thresh / 2 - 0.04),
        "fact-relaying":((pn_thresh + pn_at) / 2,   c_thresh / 2 - 0.04),
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_region_stats(df: pd.DataFrame, min_posts: int) -> pd.DataFrame:
    rows = []
    for case in CASES:
        cdf    = df[df["case"] == case]
        n_tot  = len(cdf)
        v_tot  = cdf["n_posts"].sum()
        for region in ("contested", "consolidated", "fact-relaying"):
            rdf = cdf[cdf["region"] == region]
            v_r = rdf["n_posts"].sum()
            rows.append({
                "case":          case,
                "region":        region,
                "n_clusters":    len(rdf),
                "pct_clusters":  round(len(rdf) / n_tot, 4) if n_tot else 0,
                "n_posts":       int(v_r),
                "pct_posts":     round(v_r / v_tot, 4)      if v_tot else 0,
                "n_below_min":   int((cdf["n_posts"] < min_posts).sum()),
            })
    return pd.DataFrame(rows)


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(
    df: pd.DataFrame,
    color_by_persistence: bool = False,
    annotate_top_k: int = 0,
    c_thresh: float = C_THRESH,
    pn_thresh: float = PN_THRESH,
    min_posts: int = MIN_POSTS,
    s_scale: float = S_SCALE,
    fill_color: str = FILL_COLOR,
) -> plt.Figure:

    matplotlib.rcParams.update({
        "font.family":     FONT_FAMILY,
        "font.size":       TICK_SIZE,
        "axes.linewidth":  0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    edge_rgba = _darken(fill_color, 0.65)
    polys     = _region_polygons(c_thresh, pn_thresh)
    lbl_pos   = _region_label_positions(c_thresh, pn_thresh)

    # Persistence colormap (shared across panels)
    pers_all = df["persistence"].dropna()
    if color_by_persistence and len(pers_all) > 0:
        cmap = plt.cm.Blues
        norm = mcolors.Normalize(vmin=pers_all.min(), vmax=max(pers_all.max(), 1))
    else:
        cmap = norm = None

    fig, axes = plt.subplots(
        1, 3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(wspace=0.06, left=0.08, right=0.97, top=0.90, bottom=0.18)

    print()
    for col_idx, (ax, case) in enumerate(zip(axes, CASES)):
        is_left  = col_idx == 0
        is_right = col_idx == 2
        cdf = df[df["case"] == case].copy()

        # ── Region shading ─────────────────────────────────────────────────────
        for region, verts in polys.items():
            ax.add_patch(mpatches.Polygon(
                verts, closed=True,
                facecolor=REGION_COLORS[region], alpha=REGION_ALPHA,
                edgecolor="none", zorder=1,
            ))
            if is_left:
                px, py = lbl_pos[region]
                ax.text(
                    px, py,
                    region.replace("-", "-\n") if region == "fact-relaying" else region.capitalize(),
                    fontsize=ANNOT_SIZE, color=tuple(np.array(mcolors.to_rgb(REGION_COLORS[region])) * 0.55),
                    ha="center", va="center", style="italic", zorder=3,
                )

        # ── Feasible boundary C = 1 - p_neutral ───────────────────────────────
        ax.plot([0, 1], [1, 0], "--", color="0.52", lw=0.9, zorder=2)
        if is_left:
            ax.text(0.03, 0.95, "max $C$", fontsize=ANNOT_SIZE, color="0.45",
                    va="top", ha="left", style="italic", zorder=3)

        # ── Threshold grid lines ───────────────────────────────────────────────
        ax.axhline(c_thresh,  color="0.78", lw=0.5, ls=":", zorder=2)
        ax.axvline(pn_thresh, color="0.78", lw=0.5, ls=":", zorder=2)

        # ── Scatter ────────────────────────────────────────────────────────────
        above_min = cdf["n_posts"] >= min_posts
        below_min = ~above_min

        # Dim background texture for tiny clusters
        if below_min.any():
            sub = cdf[below_min]
            ax.scatter(
                sub["p_neutral"], sub["C"],
                s=_bubble_area(sub["n_posts"], s_scale),
                c=fill_color, alpha=0.18,
                linewidths=0, edgecolors="none",
                zorder=3,
            )

        # Primary scatter — clusters with sufficient evidence
        if above_min.any():
            sub   = cdf[above_min]
            sizes = _bubble_area(sub["n_posts"].values, s_scale)
            if cmap is not None:
                c_vals = cmap(norm(sub["persistence"].fillna(pers_all.min()).values))
            else:
                c_vals = fill_color
            ax.scatter(
                sub["p_neutral"], sub["C"],
                s=sizes,
                c=c_vals,
                alpha=0.55,
                linewidths=0.4,
                edgecolors=[edge_rgba] * len(sub),
                zorder=4,
            )

        # ── Optional theme-label annotation ───────────────────────────────────
        if annotate_top_k > 0 and above_min.any():
            topk = cdf[above_min].nlargest(annotate_top_k, "n_posts")
            try:
                from adjustText import adjust_text
                texts = [
                    ax.text(
                        row["p_neutral"], row["C"],
                        str(row.get("theme_label", ""))[:28],
                        fontsize=max(ANNOT_SIZE - 1, 5), ha="center", va="bottom", zorder=5,
                    )
                    for _, row in topk.iterrows()
                    if str(row.get("theme_label", "")).strip()
                ]
                adjust_text(texts, ax=ax,
                            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4))
            except ImportError:
                for _, row in topk.iterrows():
                    lbl = str(row.get("theme_label",""))[:25].strip()
                    if lbl:
                        ax.annotate(
                            lbl, xy=(row["p_neutral"], row["C"]),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=max(ANNOT_SIZE - 1, 5),
                            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4),
                            zorder=5,
                        )

        # ── Panel decoration ───────────────────────────────────────────────────
        ax.set_title(CASE_LABELS.get(case, case), fontsize=TITLE_SIZE, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=TICK_SIZE, length=2.5, pad=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Very light horizontal grid only
        ax.yaxis.grid(True, color="0.93", lw=0.35, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        # Axis labels on outer axes only
        ax.set_xlabel("P(neutral)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Controversy score $C$" if is_left else "", fontsize=LABEL_SIZE)

        # ── Small-cluster stdout report ────────────────────────────────────────
        n_below = int(below_min.sum())
        n_total = len(cdf)
        v_below = int(cdf.loc[below_min, "n_posts"].sum())
        v_total = int(cdf["n_posts"].sum())
        print(
            f"  {case}: {n_below}/{n_total} clusters below MIN_POSTS={min_posts} "
            f"({100*n_below/max(n_total,1):.1f}%);  "
            f"{v_below:,}/{v_total:,} posts ({100*v_below/max(v_total,1):.1f}% of volume)"
        )

    # ── Size legend in rightmost panel ────────────────────────────────────────
    ax_r = axes[2]
    n_all = df["n_posts"].clip(lower=1)

    def _round_to_sig(x, sig=1):
        """Round x to sig significant figures."""
        if x <= 0:
            return 1
        mag = 10 ** int(np.floor(np.log10(x)))
        return int(round(x / mag) * mag)

    n_lo  = max(1,  _round_to_sig(n_all.quantile(0.05)))
    n_mid = max(n_lo + 1, _round_to_sig(n_all.median()))
    n_hi  = max(n_mid + 1, _round_to_sig(n_all.quantile(0.95)))
    # Snap to round numbers for readability
    for v, snap in [(n_lo, 10), (n_mid, 100), (n_hi, 1000)]:
        pass  # override: use canonical values if they fall within observed range
    n_lo  = max(1,  min(n_lo,  10))
    n_mid = max(10, min(n_mid, 100))
    n_hi  = max(100, n_hi)
    legend_ns = [n_lo, n_mid, n_hi]

    legend_handles = [
        ax_r.scatter([], [],
                     s=float(_bubble_area(n, s_scale)),
                     c=fill_color,
                     alpha=0.55,
                     linewidths=0.4,
                     edgecolors=[edge_rgba],
                     label=f"{n:,}")
        for n in legend_ns
    ]
    ax_r.legend(
        handles=legend_handles,
        title="Posts",
        title_fontsize=ANNOT_SIZE,
        fontsize=ANNOT_SIZE,
        loc="upper right",
        frameon=True,
        framealpha=0.85,
        edgecolor="0.80",
        borderpad=0.6,
        handletextpad=0.5,
        labelspacing=0.9,
    )

    # ── Shared persistence colorbar ────────────────────────────────────────────
    if cmap is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes), shrink=0.80, pad=0.02, aspect=22)
        cbar.set_label("Persistence (windows)", fontsize=ANNOT_SIZE)
        cbar.ax.tick_params(labelsize=ANNOT_SIZE)

    return fig


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir",      default="outputs/figures",
                   help="Directory to write PDF, PNG, and stats CSV")
    p.add_argument("--data-root",       default="data/evaluated",
                   help="Root directory for evaluated case outputs")
    p.add_argument("--color-by",        choices=["persistence"], default=None,
                   help="Color bubbles by a continuous variable")
    p.add_argument("--annotate-top-k",  type=int, default=0, metavar="K",
                   help="Annotate the top-K clusters by post volume per panel")
    p.add_argument("--c-thresh",        type=float, default=C_THRESH,
                   help=f"Horizontal C boundary (default {C_THRESH})")
    p.add_argument("--pn-thresh",       type=float, default=PN_THRESH,
                   help=f"Vertical p_neutral boundary (default {PN_THRESH})")
    p.add_argument("--min-posts",       type=int,   default=MIN_POSTS,
                   help=f"Min posts for full-alpha rendering (default {MIN_POSTS})")
    p.add_argument("--s-scale",         type=float, default=S_SCALE,
                   help=f"Bubble area scale factor in pt² (default {S_SCALE})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data_root)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_all()
    for case in CASES:
        cdf = df[df["case"] == case]
        if len(cdf):
            print(f"  {case}: {len(cdf):,} clusters | "
                  f"n_posts median={cdf['n_posts'].median():.1f} "
                  f"mean={cdf['n_posts'].mean():.1f} "
                  f"max={cdf['n_posts'].max()}")

    print("\nValidating proportions and computing C...")
    df = validate_and_compute(df)
    df = assign_regions(df, c_thresh=args.c_thresh, pn_thresh=args.pn_thresh)

    print(f"\nSmall-cluster report (MIN_POSTS={args.min_posts}):")
    fig = make_figure(
        df,
        color_by_persistence=(args.color_by == "persistence"),
        annotate_top_k=args.annotate_top_k,
        c_thresh=args.c_thresh,
        pn_thresh=args.pn_thresh,
        min_posts=args.min_posts,
        s_scale=args.s_scale,
    )

    stem = "stance_geometry"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    csv_path = output_dir / f"{stem}_stats.csv"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure → {pdf_path}")
    print(f"Draft  → {png_path}")

    stats = compute_region_stats(df, min_posts=args.min_posts)
    stats.to_csv(csv_path, index=False)
    print(f"Stats  → {csv_path}\n")
    print(stats.to_string(index=False))
    print(
        "\nNote: use \\begin{figure*}...\\end{figure*} in LaTeX "
        "for this 7.0-inch full-width figure."
    )


if __name__ == "__main__":
    main()
