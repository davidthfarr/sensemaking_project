"""
Generate validation_analysis.ipynb from cell source strings.

Run once on the analysis server (where nbformat is available):
    python generate_notebook.py
"""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

cells = []
md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Narrative Monitoring — Validation & Metric-Correction Analysis

**Pipeline:** embed → HDBSCAN rolling windows → Hungarian alignment → GPT theme + stance
**Cases:** Venezuela · Iran · Russia-Ukraine
**Paper:** arXiv 2603.17617

Run sections in order. Expensive operations (centroids, τ sweep) are cached under `outputs/.cache/`.

| Phase | Description |
|-------|-------------|
| 0 | Data discovery & normalization checks |
| 1 | Parameter audit |
| 2.1 | Controversy score formula correction |
| 2.2 | Drift recomputed on angular distance |
| 2.3 | Drift-event alignment test |
| 3.1 | Cluster coherence vs. null |
| 3.2 | Silhouette coefficients |
| 3.3 | Linkage threshold sensitivity (τ sweep) |
| 4 | Annotation export |
| 5 | Stance scoring (post-annotation) |
| 6 | Theme classification export |
"""))


# ── Setup ─────────────────────────────────────────────────────────────────────
cells.append(code("""\
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Repo root must be on the path for src.validation imports
sys.path.insert(0, str(Path.cwd()))

from src.validation import io, centroids as ctr, metrics as mtr, tau_sweep as ts
from src.validation import annotation as ann, scoring as sc, latex as ltx

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "venezuela": "#0072B2",
    "iran":      "#009E73",
    "russia":    "#D55E00",
}
CASES = io.CASES
SEED  = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

%matplotlib inline
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (6.5, 3.5),
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

print("Setup complete. Cases:", CASES)
"""))

cells.append(code("""\
# Create output directories
FIGURES_DIR = Path("outputs/figures")
TABLES_DIR  = Path("outputs/tables")
ANNOT_DIR   = Path("outputs/annotation")
CACHE_DIR   = Path("outputs/.cache")

for d in [FIGURES_DIR, TABLES_DIR, ANNOT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print("Output directories ready.")
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: Data Discovery
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Phase 0: Data Discovery & Normalization Checks"))

cells.append(code("""\
discovery = []
for case in CASES:
    row = {"case": case}
    try:
        repr_df = io.load_posts_repr(case)
        gc_df   = io.load_global_clusters(case)
        stance  = io.load_cluster_stance(case)
        wfiles  = io.load_window_files(case)

        nonnoise = gc_df[~gc_df["is_noise"] & gc_df["global_cluster_id"].notna()]
        noise    = gc_df[gc_df["is_noise"]]

        ts_vals = pd.to_datetime(
            gc_df["window"].str.replace(r"(\\d{4}-\\d{2}-\\d{2})-(\\d{2})", r"\\1 \\2:00", regex=True),
            utc=True, errors="coerce",
        ).dropna()

        row.update({
            "n_posts_repr":      len(repr_df),
            "n_posts_clustered": nonnoise["post_id"].nunique(),
            "noise_pct":         f"{100 * len(noise) / len(gc_df):.1f}%",
            "n_global_clusters": int(gc_df["global_cluster_id"].dropna().nunique()),
            "n_windows":         len(wfiles),
            "date_range":        f"{ts_vals.min().date()} – {ts_vals.max().date()}" if len(ts_vals) else "N/A",
            "status":            "OK",
        })
    except FileNotFoundError as e:
        row.update({"status": f"MISSING: {e}"})

    discovery.append(row)

disc_df = pd.DataFrame(discovery).set_index("case")
display(disc_df)
"""))

cells.append(code("""\
# Verify L2 normalization of stored embeddings (sample 500 posts per case)
print("Embedding L2-norm check (should be 1.0 ± 1e-5 for all posts):\\n")
for case in CASES:
    try:
        repr_df = io.load_posts_repr(case)
        valid   = repr_df["embedding"].dropna()
        sample  = valid.sample(min(500, len(valid)), random_state=SEED)
        norms   = np.array([np.linalg.norm(e) for e in sample])
        print(f"  {case}: mean={norms.mean():.6f}  std={norms.std():.2e}  "
              f"min={norms.min():.6f}  max={norms.max():.6f}")
        assert norms.min() > 0.99 and norms.max() < 1.01, \
            f"Non-unit norms detected in {case} — embeddings may not be L2-normalized"
    except FileNotFoundError:
        print(f"  {case}: SKIP (no posts_repr.parquet)")
print("\\nAll checks passed.")
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Parameter Audit
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Phase 1: Parameter Audit"))

cells.append(code("""\
# Hard-coded from code inspection of the pipeline scripts.
# Sources: run_rolling_windows.py CASE_PARAMS, run_pipeline.py defaults,
#          sensemaking/embeddings/encoder.py, sensemaking/clustering/hdbscan.py,
#          sensemaking/clustering/alignment.py, sensemaking/themes/stationary_labeler.py,
#          sensemaking/stance/posthoc_gpt.py

PARAM_TABLE = [
    # ── Embeddings ──────────────────────────────────────────────
    ("Embedding model",       "all-mpnet-base-v2 (sentence-transformers)",    "encoder.py"),
    ("Embedding dim",         "768",                                           "encoder.py"),
    ("L2 normalization",      "Yes (normalize=True at encode time)",           "encoder.py"),
    ("Batch size",            "64",                                            "encoder.py"),

    # ── HDBSCAN ─────────────────────────────────────────────────
    ("HDBSCAN metric",        "euclidean (= cosine on L2-norm vectors)",       "hdbscan.py"),
    ("Cluster selection",     "eom (library default)",                         "hdbscan.py"),
    ("min_cluster_size (VEN)","10 (floor); cap=20; dynamic=True",             "run_rolling_windows.py"),
    ("min_cluster_size (IRN)","20 (floor); cap=30; dynamic=True",             "run_rolling_windows.py"),
    ("min_cluster_size (RUS)","5 (floor);  cap=50; dynamic=True",             "run_rolling_windows.py"),
    ("min_samples",           "3 (VEN/IRN) / 10 (RUS)",                       "run_rolling_windows.py"),
    ("epsilon",               "0.20 (VEN) / 0.30 (IRN) / 0.15 (RUS)",        "run_rolling_windows.py"),
    ("Window / step (VEN)",   "8 h / 4 h",                                    "run_rolling_windows.py"),
    ("Window / step (IRN)",   "24 h / 8 h",                                   "run_rolling_windows.py"),
    ("Window / step (RUS)",   "168 h / 24 h",                                 "run_rolling_windows.py"),
    ("Dynamic MCS formula",   "min(cap, max(floor, len(posts)//50))",          "run_rolling_windows.py"),

    # ── Alignment ───────────────────────────────────────────────
    ("Alignment method",      "Hungarian matching on cosine-similarity matrix","alignment.py"),
    ("τ (similarity thresh)", "0.70 (--align-threshold passed to run_pipeline)","run_pipeline.py"),
    ("τ default in script",   "0.50 (run_pipeline.py) / 0.85 (run_alignment.py standalone)","scripts"),

    # ── Theme labeling ──────────────────────────────────────────
    ("Theme model",           "gpt-4o-mini",                                  "run_theme_labeling.py"),
    ("Theme temperature",     "0.3",                                           "stationary_labeler.py"),
    ("Theme max words",       "10",                                            "stationary_labeler.py"),
    ("Representative posts",  "5 nearest centroid",                            "stationary_labeler.py"),

    # ── Stance classification ───────────────────────────────────
    ("Stance model",          "gpt-4o-mini",                                  "run_stance_classification.py"),
    ("Stance temperature",    "0.0",                                           "posthoc_gpt.py"),
    ("Stance mode",           "cluster (one label per cluster per window)",    "run_stance_classification.py"),
    ("Controversy formula",   "C = 1 - neu - |s - o|  (= 2·min(s,o))",       "CORRECTED in this analysis"),
]

params_df = pd.DataFrame(PARAM_TABLE, columns=["Parameter", "Value", "Source"])
display(params_df)
"""))

cells.append(code("""\
# Export parameter audit table
params_df.to_csv(TABLES_DIR / "parameter_audit.csv", index=False)
ltx.save_table(
    params_df, TABLES_DIR / "parameter_audit.tex",
    caption="Pipeline parameter audit. Values confirmed from source code inspection.",
    label="param_audit",
    column_format="lll",
)
print("Exported parameter_audit.{csv,tex}")
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.1: Controversy Score
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 2.1: Controversy Score — Formula Correction

The stored `controversy_score` was computed as `(1 - |s - o|) × (1 - neu)`.
The manuscript formula is **C = 1 - neu - |s - o| = 2 · min(s, o)**.
These differ: the stored formula conflates two terms multiplicatively rather than additively.

This section recomputes C using the correct formula, verifies the closed-form identity,
and tests whether high-volume clusters inflate the neutrality baseline.
"""))

cells.append(code("""\
all_stance = {}
for case in CASES:
    try:
        df = io.load_cluster_stance(case)
        df = mtr.recompute_controversy(df)
        df["case"] = case
        all_stance[case] = df
        print(f"  {case}: {len(df)} clusters loaded")
    except FileNotFoundError as e:
        print(f"  SKIP {case}: {e}")
"""))

cells.append(code("""\
# Closed-form identity check: C_correct should equal 2*min(s,o)
print("Closed-form identity  C = 2·min(s,o):")
for case, df in all_stance.items():
    n_bad = (~df["C_closed_form_match"]).sum()
    print(f"  {case}: {'FAIL ' + str(n_bad) + ' mismatches' if n_bad else 'OK'}")

print()
# Stored vs correct discrepancy check
print("Stored controversy_score vs recomputed C_correct:")
for case, df in all_stance.items():
    if "C_stored_matches_correct" not in df.columns:
        print(f"  {case}: no stored values to compare")
        continue
    n_diff = (~df["C_stored_matches_correct"]).sum()
    c_mean_old = df.get("C_stored", pd.Series(dtype=float)).mean()
    c_mean_new = df["C_correct"].mean()
    print(f"  {case}: {n_diff}/{len(df)} clusters differ | "
          f"mean C_stored={c_mean_old:.4f}  mean C_correct={c_mean_new:.4f}")
"""))

cells.append(code("""\
# Distribution of C_correct per case
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=False)
for ax, (case, df) in zip(axes, all_stance.items()):
    ax.hist(df["C_correct"], bins=20, color=PALETTE[case], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(df["C_correct"].median(), color="k", lw=1, ls="--", label=f'median={df["C_correct"].median():.2f}')
    ax.set_title(case.capitalize())
    ax.set_xlabel("C")
    if ax is axes[0]:
        ax.set_ylabel("Clusters")
    ax.legend(fontsize=7)

plt.suptitle("Controversy Score Distribution (corrected formula)", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "controversy_distribution.pdf", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR}/controversy_distribution.pdf")
"""))

cells.append(code("""\
# Scatter: C vs p_neutral; size = sqrt(n_posts)
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
for ax, (case, df) in zip(axes, all_stance.items()):
    sizes = np.sqrt(df["n_posts"].clip(lower=1)) * 3
    ax.scatter(df["neutral_pct"], df["C_correct"],
               s=sizes, color=PALETTE[case], alpha=0.45, linewidths=0)
    ax.set_xlabel("p_neutral")
    ax.set_title(case.capitalize())
    if ax is axes[0]:
        ax.set_ylabel("C")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.suptitle("C vs. p_neutral  (marker size ∝ √n_posts)", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "c_vs_neutral.pdf", bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# Volume-neutrality test: do high-volume clusters skew more neutral?
print("Volume–neutrality test (Mann-Whitney U, one-sided: high-vol > low-vol)\\n")
vn_rows = []
for case, df in all_stance.items():
    res = mtr.volume_neutrality_test(df)
    if "error" in res:
        print(f"  {case}: {res['error']}")
        continue
    vn_rows.append({
        "case":         case,
        "n_high_vol":   res["n_high_volume"],
        "n_low_vol":    res["n_low_volume"],
        "mean_neu_hi":  round(res["mean_neutral_high_vol"], 3),
        "mean_neu_lo":  round(res["mean_neutral_low_vol"], 3),
        "p_value":      round(res["p_value"], 4),
        "r (rank-biserial)": round(res["rank_biserial_r"], 3),
        "effect":       res["effect_size_label"],
    })

vn_df = pd.DataFrame(vn_rows)
display(vn_df)

vn_df.to_csv(TABLES_DIR / "volume_neutrality_test.csv", index=False)
ltx.save_table(vn_df, TABLES_DIR / "volume_neutrality_test.tex",
    caption="Mann-Whitney U test: high-volume clusters vs. rest on p_neutral. "
            r"Rank-biserial $r$ is the effect size.",
    label="vol_neu")
"""))

cells.append(md("""\
**Manuscript implications (2.1):** Replace all stored `controversy_score` values
with C = 1 − neu − |s − o| before any ranking or figure in the paper. Report
`C_tilde = C / (1 − neu)` only for clusters with ≥ 20 non-neutral posts, noting
the floor in the text. If the volume–neutrality test is significant, add a caveat
that large clusters' C values are downward-biased by neutrality inflation.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.2: Drift (Recomputed on Angular Distance)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 2.2: Drift — Recomputed on Angular Distance

Cluster centroids are means of L2-normalized post embeddings. Because the mean of
unit vectors is not itself unit-length, Euclidean distance between centroids conflates
semantic displacement with cluster-dispersion changes. Angular distance (arccos of
cosine similarity between L2-normalized centroids) is scale-invariant and matches
the τ threshold used during alignment.
"""))

cells.append(code("""\
# Load posts_repr and global_clusters; compute / cache per-window centroids
centroid_cache = {}
gc_cache = {}
emb_cache = {}

for case in CASES:
    try:
        repr_df  = io.load_posts_repr(case)
        gc_df    = io.load_global_clusters(case)
        emb_map  = io.build_emb_map(repr_df)
        cdf      = ctr.load_or_compute_centroids(case, gc_df, emb_map, CACHE_DIR)
        centroid_cache[case] = cdf
        gc_cache[case]       = gc_df
        emb_cache[case]      = emb_map
        print(f"  {case}: {len(cdf):,} (cluster, window) pairs")
    except FileNotFoundError as e:
        print(f"  SKIP {case}: {e}")
"""))

cells.append(code("""\
# Compute drift statistics per cluster per case
drift_all = {}
for case, cdf in centroid_cache.items():
    drift_df = mtr.compute_drift_stats(cdf)
    drift_df["case"] = case
    drift_all[case] = drift_df
    print(f"  {case}: {len(drift_df)} clusters | "
          f"median cumulative drift={drift_df['cumulative_path_angular'].median():.3f} rad")

combined_drift = pd.concat(drift_all.values(), ignore_index=True)
combined_drift.drop(columns="centroid", errors="ignore").to_csv(
    TABLES_DIR / "drift_stats.csv", index=False)
"""))

cells.append(code("""\
# Drift distributions: cumulative path length and net displacement
fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5), sharey="row")
metrics_plot = [
    ("cumulative_path_angular", "Cumulative path (rad)"),
    ("net_displacement_angular","Net displacement (rad)"),
]
for row_i, (col, ylabel) in enumerate(metrics_plot):
    for ax, (case, df) in zip(axes[row_i], drift_all.items()):
        multi = df[df["n_windows"] > 1]
        ax.hist(multi[col], bins=20, color=PALETTE[case], alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_xlabel(ylabel)
        if ax is axes[row_i, 0]:
            ax.set_ylabel("Clusters")
        ax.set_title(case.capitalize() if row_i == 0 else "")

plt.suptitle("Drift statistics (clusters with ≥ 2 windows)", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "drift_distributions.pdf", bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# Correlation: cumulative drift vs persistence; angular vs cosine drift
print("Spearman ρ: cumulative angular drift vs persistence\\n")
for case, df in drift_all.items():
    multi = df[df["n_windows"] > 1]
    if len(multi) < 5:
        continue
    rho, p = stats.spearmanr(multi["cumulative_path_angular"], multi["n_windows"])
    print(f"  {case}: ρ={rho:.3f}  p={p:.4f}  n={len(multi)}")

print("\\nSpearman ρ: angular cumulative vs cosine cumulative (sanity check)\\n")
for case, df in drift_all.items():
    multi = df[df["n_windows"] > 1]
    if len(multi) < 5:
        continue
    rho, p = stats.spearmanr(multi["cumulative_path_angular"], multi["cumulative_path_cosine"])
    print(f"  {case}: ρ={rho:.3f}  p={p:.4f}")
"""))

cells.append(md("""\
**Manuscript implications (2.2):** Report centroid drift in radians (angular distance),
not Euclidean distance. Cite the monotone relationship with cosine distance as justification.
High ρ between cumulative drift and persistence confirms that long-lived narratives are not
stationary — consistent with an information-environment hypothesis.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.3: Drift-Event Alignment
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 2.3: Drift-Event Alignment

Test whether windows immediately following a known real-world event show elevated
narrative drift relative to other windows. Requires the `data/event_timelines/` CSVs
to be populated (currently stubs — populate before running this cell).

Permutation test: compare mean drift in ±1 window around events vs. all other windows.
"""))

cells.append(code("""\
# Load event timelines and compute per-window mean drift time series
for case in CASES:
    events = io.load_event_timeline(case)
    if events.empty:
        print(f"  {case}: no events in timeline — populate data/event_timelines/{case}.csv first")
        continue

    if case not in centroid_cache:
        print(f"  {case}: centroids not available, skipping")
        continue

    # Per-window mean drift across all clusters active in that window
    cdf = centroid_cache[case]
    # Compute step drift (from centroid_df, per window boundary)
    drift_df = drift_all[case]
    # Map window → timestamp
    cdf["timestamp"] = pd.to_datetime(
        cdf["window"].str.replace(r"(\\d{4}-\\d{2}-\\d{2})-(\\d{2})", r"\\1 \\2:00", regex=True),
        utc=True, errors="coerce",
    )
    # Per-window mean drift step: mean of per-cluster mean_step_angular
    per_window = (
        drift_df.dropna(subset=["mean_step_angular"])
        .groupby("case", as_index=False)["mean_step_angular"].mean()
    )

    fig, ax = plt.subplots(figsize=(6, 2.5))
    # Plot a note since we need per-window granularity (not per-cluster summary)
    ax.set_title(f"{case.capitalize()} — populate event timeline to enable this plot")
    ax.text(0.5, 0.5, "Event timeline not yet populated.\\n"
            f"Edit data/event_timelines/{case}.csv",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="grey")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"drift_events_{case}.pdf", bbox_inches="tight")
    plt.show()
"""))

cells.append(md("""\
**Manuscript implications (2.3):** Once event timelines are populated, a permutation test
(1,000 draws) should produce a p-value for the claim that narrative drift spikes around
real-world events. Report effect size (Cohen's d) alongside the p-value.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.1: Cluster Coherence vs. Null
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 3.1: Cluster Coherence vs. Null

For each cluster, compute the mean cosine similarity of member post embeddings to the
cluster centroid. Compare against a null distribution: 100 size-matched random draws
from posts in the same window(s). Reports Cohen's d effect sizes.
"""))

cells.append(code("""\
NULL_DRAWS = 100
coherence_results = {}

for case in CASES:
    if case not in centroid_cache:
        print(f"  SKIP {case}: centroids not available")
        continue

    cdf    = centroid_cache[case]
    gc_df  = gc_cache[case]
    emb_map = emb_cache[case]

    df_nn = gc_df[~gc_df["is_noise"] & gc_df["global_cluster_id"].notna()].copy()
    df_nn["global_cluster_id"] = df_nn["global_cluster_id"].astype(int)

    # All posts per window (for null pool)
    window_pool = {}
    for window, wgrp in gc_df.groupby("window"):
        embs = [emb_map[pid] for pid in wgrp["post_id"]
                if pid in emb_map and emb_map[pid] is not None
                and hasattr(emb_map[pid], "shape") and emb_map[pid].shape == (768,)]
        if embs:
            window_pool[window] = np.vstack(embs).astype(np.float32)

    rng = np.random.default_rng(SEED)
    observed_sims, null_means = [], []

    for gcid, cgrp in cdf.groupby("global_cluster_id"):
        # Aggregate centroid over all windows for this cluster
        centroid = np.vstack(cgrp["centroid"].tolist()).mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        # Member embeddings
        member_pids = df_nn[df_nn["global_cluster_id"] == gcid]["post_id"].unique()
        member_embs = np.vstack([
            emb_map[pid] for pid in member_pids
            if pid in emb_map and emb_map[pid] is not None
            and hasattr(emb_map[pid], "shape") and emb_map[pid].shape == (768,)
        ]).astype(np.float32) if len(member_pids) > 0 else None

        if member_embs is None or len(member_embs) < 2:
            continue

        obs_sim = float(np.dot(member_embs, centroid).mean())
        observed_sims.append(obs_sim)

        # Null: random size-matched draws from same windows
        windows_used = cgrp["window"].tolist()
        null_pool_parts = [window_pool[w] for w in windows_used if w in window_pool]
        if not null_pool_parts:
            continue
        null_pool = np.vstack(null_pool_parts)

        draw_means = []
        for _ in range(NULL_DRAWS):
            idx = rng.choice(len(null_pool), size=min(len(member_embs), len(null_pool)), replace=False)
            draw_means.append(float(np.dot(null_pool[idx], centroid).mean()))
        null_means.append(np.mean(draw_means))

    observed_sims = np.array(observed_sims)
    null_means    = np.array(null_means[:len(observed_sims)])
    diff          = observed_sims - null_means
    cohens_d      = float(diff.mean() / (diff.std() + 1e-9))

    coherence_results[case] = {
        "n_clusters": len(observed_sims),
        "mean_observed": float(observed_sims.mean()),
        "mean_null":     float(null_means.mean()),
        "mean_diff":     float(diff.mean()),
        "cohens_d":      cohens_d,
        "pct_above_null": float((observed_sims > null_means).mean()),
    }
    print(f"  {case}: d={cohens_d:.3f}  obs={observed_sims.mean():.4f}  "
          f"null={null_means.mean():.4f}  pct_above={coherence_results[case]['pct_above_null']:.2%}")
"""))

cells.append(code("""\
coh_df = pd.DataFrame(coherence_results).T
display(coh_df.round(4))
coh_df.to_csv(TABLES_DIR / "coherence_vs_null.csv")
ltx.save_table(
    coh_df.reset_index().rename(columns={"index": "case"}),
    TABLES_DIR / "coherence_vs_null.tex",
    caption="Cluster coherence vs. size-matched random null. "
            "Cohen's $d$ is the effect size of the observed−null difference.",
    label="coherence")

# Distribution plot
fig, axes = plt.subplots(1, len(coherence_results), figsize=(6.5, 2.5))
if len(coherence_results) == 1:
    axes = [axes]
for ax, (case, res) in zip(axes, coherence_results.items()):
    ax.bar(["Observed", "Null"], [res["mean_observed"], res["mean_null"]],
           color=[PALETTE[case], "lightgrey"], edgecolor="white")
    ax.set_title(f"{case.capitalize()}\\nd={res['cohens_d']:.2f}")
    ax.set_ylabel("Mean cosine sim to centroid" if ax is axes[0] else "")

plt.suptitle("Cluster Coherence vs. Random Null", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "coherence_vs_null.pdf", bbox_inches="tight")
plt.show()
"""))

cells.append(md("""\
**Manuscript implications (3.1):** Cite Cohen's d and the percentage of clusters
with observed similarity above null as evidence that HDBSCAN clusters are
semantically coherent — validating the clustering as topically meaningful rather
than arbitrary. A Cohen's d > 0.8 is considered "large" by conventional thresholds.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.2: Silhouette Coefficients
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Phase 3.2: Silhouette Coefficients"))

cells.append(code("""\
# Compute per-window silhouette for each case
silhouette_series = {}

for case in CASES:
    if case not in emb_cache:
        print(f"  SKIP {case}: embeddings not loaded")
        continue

    emb_map = emb_cache[case]
    wfiles  = io.load_window_files(case)
    scores  = []

    for wf in wfiles:
        wdf = pd.read_parquet(wf)
        for alias in ("Resource Id", "tweet_id", "id"):
            if alias in wdf.columns and "post_id" not in wdf.columns:
                wdf = wdf.rename(columns={alias: "post_id"})
                break
        wdf["post_id"] = wdf["post_id"].astype(str)

        s = mtr.compute_silhouette_per_window(wdf, emb_map)
        scores.append({"window": wf.stem, "silhouette": s})

    sdf = pd.DataFrame(scores)
    sdf["timestamp"] = pd.to_datetime(
        sdf["window"].str.replace(r"(\\d{4}-\\d{2}-\\d{2})-(\\d{2})", r"\\1 \\2:00", regex=True),
        utc=True, errors="coerce",
    )
    silhouette_series[case] = sdf
    valid = sdf["silhouette"].dropna()
    print(f"  {case}: {len(valid)}/{len(sdf)} windows scored | "
          f"mean={valid.mean():.4f}  median={valid.median():.4f}")
"""))

cells.append(code("""\
# Plot silhouette over time per case
fig, axes = plt.subplots(len(silhouette_series), 1, figsize=(6.5, 2 * len(silhouette_series)),
                         sharex=False)
if len(silhouette_series) == 1:
    axes = [axes]

for ax, (case, sdf) in zip(axes, silhouette_series.items()):
    valid = sdf.dropna(subset=["silhouette"])
    ax.plot(valid["timestamp"], valid["silhouette"], color=PALETTE[case], lw=1.2)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axhline(valid["silhouette"].mean(), color=PALETTE[case], lw=0.8, ls=":",
               label=f'mean={valid["silhouette"].mean():.3f}')
    ax.set_ylabel("Silhouette")
    ax.set_title(case.capitalize())
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)

plt.suptitle("Per-Window Silhouette Coefficient (cosine distance)", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "silhouette_over_time.pdf", bbox_inches="tight")
plt.show()

# Summary table
sil_summary = pd.DataFrame({
    case: {
        "mean":   sdf["silhouette"].mean(),
        "median": sdf["silhouette"].median(),
        "std":    sdf["silhouette"].std(),
        "pct_positive": (sdf["silhouette"] > 0).mean(),
    }
    for case, sdf in silhouette_series.items()
}).T
display(sil_summary.round(4))
sil_summary.to_csv(TABLES_DIR / "silhouette_summary.csv")
"""))

cells.append(md("""\
**Manuscript implications (3.2):** Report per-case mean silhouette scores.
A positive mean (> 0) indicates within-cluster similarity exceeds between-cluster
similarity on average — supporting the quality of HDBSCAN clustering at each window.
Windows with negative silhouette likely correspond to periods with low narrative
differentiation or high noise fraction.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.3: Linkage Threshold Sensitivity (τ sweep)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 3.3: Linkage Threshold Sensitivity (τ sweep)

Re-run alignment for τ ∈ {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}.
τ = 0.70 is the value actually used; the sweep characterises sensitivity of
cluster count, persistence, and fragmentation to this choice.

**Warning:** this section re-processes all window parquets and may take
10–30 minutes per case on first run. Results are cached under `outputs/.cache/`.
"""))

cells.append(code("""\
TAUS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]   # 0.70 = value used in production

sweep_results = {}
for case in CASES:
    if case not in emb_cache:
        print(f"  SKIP {case}: embeddings not loaded")
        continue
    print(f"\\n── τ sweep for '{case}' ──────────────────────────────────────")
    sweep_df = ts.sweep_case(
        case=case,
        window_files=io.load_window_files(case),
        emb_map=emb_cache[case],
        taus=TAUS,
        cache_dir=CACHE_DIR,
    )
    sweep_results[case] = sweep_df
"""))

cells.append(code("""\
# Display sensitivity table per case
for case, df in sweep_results.items():
    print(f"\\n{case.upper()}")
    disp = df[["n_global_clusters","mean_persistence","median_persistence",
               "frac_one_window","frac_posts_3plus_windows","n_reactivations"]].copy()
    disp.index.name = "τ"
    display(disp.round(3))
"""))

cells.append(code("""\
# Plot: n_global_clusters and mean_persistence vs τ
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
ax_n, ax_p = axes

for case, df in sweep_results.items():
    ax_n.plot(df.index, df["n_global_clusters"],
              marker="o", ms=4, lw=1.5, label=case.capitalize(), color=PALETTE[case])
    ax_p.plot(df.index, df["mean_persistence"],
              marker="o", ms=4, lw=1.5, label=case.capitalize(), color=PALETTE[case])

# Mark actual τ used
for ax in axes:
    ax.axvline(0.70, color="k", lw=0.8, ls="--", label="τ=0.70 (used)")
    ax.set_xlabel("τ (alignment threshold)")
    ax.legend(fontsize=7)

ax_n.set_ylabel("Global clusters")
ax_p.set_ylabel("Mean persistence (windows)")
ax_n.set_title("Cluster fragmentation")
ax_p.set_title("Narrative persistence")

plt.suptitle("Alignment Threshold Sensitivity (τ sweep)", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "tau_sensitivity.pdf", bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# Export combined sweep table
all_sweep = pd.concat(
    {case: df for case, df in sweep_results.items()},
    axis=0,
).reset_index(names=["case", "tau"])
all_sweep.to_csv(TABLES_DIR / "tau_sensitivity.csv", index=False)
ltx.save_table(
    all_sweep,
    TABLES_DIR / "tau_sensitivity.tex",
    caption=r"Alignment threshold (\\tau) sensitivity across cases. "
            "Values at \\tau=0.70 are the production results.",
    label="tau_sens",
    float_fmt=".3f",
)
print(f"Exported to {TABLES_DIR}/tau_sensitivity.{{csv,tex}}")
"""))

cells.append(md("""\
**Manuscript implications (3.3):** Characterise the plateau region — the range of τ where
`n_global_clusters` and `mean_persistence` stabilise. If the plateau is wide and includes
τ = 0.70, argue that the results are not sensitive to the exact threshold. If τ = 0.70
sits at an inflection point, add a robustness note. Report 1-window fraction as a measure
of fragmentation; high values at large τ indicate over-splitting.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Annotation Export
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 4: Annotation Export

Draws a stratified sample of 400 posts for human annotation.

**Stratification:**
1. Proportional to cluster-assigned volume per case.
2. Within each case, balanced across the 3 stance classes (cluster modal stance).
3. Fixed seed = 42.

Files written to `outputs/annotation/`:
- `annotator_A.csv`, `annotator_B.csv` — blind (no model label)
- `_key.csv` — model label, for scoring after annotation is complete
- `INSTRUCTIONS.md` — complete annotation protocol with case claims and label definitions
"""))

cells.append(code("""\
# Load all required data for pool construction
gc_dfs, stance_dfs, repr_dfs, themes_dfs = {}, {}, {}, {}
for case in CASES:
    try:
        gc_dfs[case]     = io.load_global_clusters(case)
        stance_dfs[case] = io.load_cluster_stance(case)
        repr_dfs[case]   = io.load_posts_repr(case)
        try:
            themes_dfs[case] = io.load_cluster_themes(case)
        except FileNotFoundError:
            print(f"  [{case}] No cluster_themes.parquet — theme column will be blank")
    except FileNotFoundError as e:
        print(f"  SKIP {case}: {e}")

if not gc_dfs:
    print("No data loaded — cannot export annotation sample")
else:
    pool = ann.build_annotation_pool(CASES, gc_dfs, stance_dfs, repr_dfs, themes_dfs)
    print(f"\\nAnnotation pool: {len(pool):,} cluster-assigned posts")
    print(pool.groupby(["case", "modal_stance"]).size().unstack(fill_value=0).to_string())
"""))

cells.append(code("""\
if gc_dfs:
    sample = ann.stratified_sample(pool, n=ann.ANNOTATION_N, seed=ann.ANNOTATION_SEED)
    print(f"\\nSample ({len(sample)} posts):")
    print(sample.groupby(["case", "modal_stance"]).size().unstack(fill_value=0).to_string())
    print()
    ann.export_annotation_files(sample, ANNOT_DIR)
    print(f"\\nAnnotation files → {ANNOT_DIR}/")
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Stance Scoring (post-annotation)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 5: Stance Scoring (post-annotation)

**Run after annotation is complete.** This section is a no-op if the annotation
files have not been filled in yet.

Required files (created in Phase 4):
- `outputs/annotation/annotator_A.csv` — filled in with `stance_label` by Annotator A
- `outputs/annotation/annotator_B.csv` — filled in by Annotator B
- `outputs/annotation/_key.csv` — model labels (auto-generated, do not edit)

Workflow:
1. Compute IAA between A and B; review disagreements.
2. Adjudicate disagreements to produce a gold standard.
3. Score model against gold; export confusion matrices.
4. Simulate bias in C induced by model errors.
"""))

cells.append(code("""\
# Load annotation files; skip gracefully if incomplete
a_path  = ANNOT_DIR / "annotator_A.csv"
b_path  = ANNOT_DIR / "annotator_B.csv"
key_path = ANNOT_DIR / "_key.csv"

ann_ready = all(p.exists() for p in [a_path, b_path, key_path])
if not ann_ready:
    print("Annotation files not found. Run Phase 4 first, then complete annotation.")
else:
    a_df   = pd.read_csv(a_path)
    b_df   = pd.read_csv(b_path)
    key_df = pd.read_csv(key_path)

    # Check completion
    a_filled = a_df["stance_label"].isin(sc.LABELS).sum()
    b_filled = b_df["stance_label"].isin(sc.LABELS).sum()
    print(f"Annotator A: {a_filled}/{len(a_df)} rows filled")
    print(f"Annotator B: {b_filled}/{len(b_df)} rows filled")
    ann_complete = (a_filled == len(a_df)) and (b_filled == len(b_df))
    if not ann_complete:
        print("\\nAnnotation incomplete — fill in stance_label columns and re-run.")
"""))

cells.append(code("""\
if ann_ready and ann_complete:
    # Inter-annotator agreement
    iaa = sc.compute_iaa(a_df, b_df)
    print(f"IAA: κ={iaa['overall_kappa']:.3f}  agreement={iaa['agreement_rate']:.1%}  "
          f"n_disagreements={iaa['n_disagreements']}")
    print("Per-class κ:", {k: round(v, 3) for k, v in iaa["per_class_kappa"].items()})

    iaa_df = pd.DataFrame({
        "metric": ["overall_kappa", "agreement_rate"] + [f"kappa_{l}" for l in sc.LABELS],
        "value":  [iaa["overall_kappa"], iaa["agreement_rate"]]
                  + [iaa["per_class_kappa"][l] for l in sc.LABELS],
    })
    iaa_df.to_csv(TABLES_DIR / "iaa.csv", index=False)
    ltx.save_table(iaa_df, TABLES_DIR / "iaa.tex",
        caption=r"Inter-annotator agreement. Cohen's $\\kappa$ overall and per-class.",
        label="iaa")
    display(iaa["disagreements"].head(10))
"""))

cells.append(code("""\
if ann_ready and ann_complete:
    # ── Produce gold standard from adjudicated disagreements ───────────────
    # Strategy: use A's label for agreements; for disagreements, require a
    # 3rd adjudicator or majority vote. Here we use A as gold if A==B,
    # otherwise mark as NEEDS_REVIEW.
    merged = a_df[["post_id","stance_label"]].merge(
        b_df[["post_id","stance_label"]], on="post_id", suffixes=("_a","_b"))
    merged["gold"] = np.where(
        merged["stance_label_a"] == merged["stance_label_b"],
        merged["stance_label_a"], "DISAGREE")
    gold_df = merged[merged["gold"] != "DISAGREE"].rename(columns={"gold": "stance_label"})

    n_agree = len(gold_df)
    n_total = len(merged)
    print(f"Agreement-only gold standard: {n_agree}/{n_total} posts "
          f"({n_agree/n_total:.1%})")

    # Score model
    perf = sc.compute_model_performance(gold_df, key_df)
    print(f"\\nModel performance:")
    print(f"  accuracy={perf['accuracy']:.3f}  macro_F1={perf['macro_f1']:.3f}")

    # Confusion matrix
    display(perf["confusion_matrix"])
    ltx.save_table(
        perf["confusion_matrix"].reset_index().rename(columns={"index": "true \\\\ pred"}),
        TABLES_DIR / "confusion_matrix.tex",
        caption="Confusion matrix: GPT stance vs. human gold standard.",
        label="cm")

    # Normalized confusion matrix (for bias simulation)
    display(perf["confusion_matrix_normalized"].round(3))
"""))

cells.append(code("""\
if ann_ready and ann_complete:
    # Bias simulation
    bias_df = sc.simulate_c_bias(perf["confusion_matrix_normalized"])
    threshold = bias_df.attrs.get("recommended_threshold", float("nan"))
    print(f"\\nRecommended C threshold (|bias| < 0.05): {threshold:.2f}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(bias_df["C_true"], bias_df["induced_bias"], "k-", lw=1.5)
    ax.axhline(0,    color="grey", lw=0.8, ls="--")
    ax.axhline(0.05, color="red",  lw=0.8, ls=":", label="|bias|=0.05")
    ax.axhline(-0.05,color="red",  lw=0.8, ls=":")
    if not np.isnan(threshold):
        ax.axvline(threshold, color="orange", lw=0.8, ls="-.", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("C (true)")
    ax.set_ylabel("Induced bias in C")
    ax.set_title("C bias from model classification errors")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "c_bias_simulation.pdf", bbox_inches="tight")
    plt.show()

    bias_df.to_csv(TABLES_DIR / "c_bias_simulation.csv", index=False)
"""))

cells.append(md("""\
**Manuscript implications (5):** Report κ ≥ 0.6 as acceptable IAA for political-framing
stance annotation. Model macro-F1 of 0.7+ supports using GPT-derived stance as a proxy
for human judgment. The bias simulation provides a data-driven lower bound on
interpretable C values — cite the recommended threshold when reporting top-ranked clusters.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Theme Classification
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Phase 6: Theme Classification Export

Export a spreadsheet of cluster themes for manual taxonomy coding
(e.g., "accusation", "legitimation", "counter-narrative", "factual report").
Includes drift, persistence, and C to prioritise clusters worth reading.

Fill in the `theme_type` column in `outputs/tables/theme_labels_for_tagging.csv`
and re-run the follow-on cell to compare coded groups.
"""))

cells.append(code("""\
theme_export_rows = []
for case in CASES:
    try:
        themes = io.load_cluster_themes(case)
        stance = io.load_cluster_stance(case)
        if case in drift_all:
            drift  = drift_all[case][["global_cluster_id","n_windows",
                                      "cumulative_path_angular","net_displacement_angular"]]
        else:
            drift = pd.DataFrame(columns=["global_cluster_id","n_windows",
                                           "cumulative_path_angular","net_displacement_angular"])

        merged = themes.merge(
            stance[["global_cluster_id","n_posts","support_pct","oppose_pct",
                    "neutral_pct","controversy_score"]],
            on="global_cluster_id", how="left"
        ).merge(drift, on="global_cluster_id", how="left")
        merged["case"] = case
        theme_export_rows.append(merged)
    except FileNotFoundError as e:
        print(f"  SKIP {case}: {e}")

if theme_export_rows:
    te_df = pd.concat(theme_export_rows, ignore_index=True)
    te_df["theme_type"] = ""   # filled by researcher
    te_df = te_df.sort_values(["case", "n_posts"], ascending=[True, False])
    export_path = TABLES_DIR / "theme_labels_for_tagging.csv"
    te_df.to_csv(export_path, index=False)
    print(f"Exported {len(te_df)} themes → {export_path}")
    display(te_df.head(10))
"""))

cells.append(code("""\
# Follow-on: load back after tagging and compare groups
tag_path = TABLES_DIR / "theme_labels_for_tagging.csv"
if tag_path.exists():
    tagged = pd.read_csv(tag_path)
    tagged_complete = tagged["theme_type"].notna() & (tagged["theme_type"].str.strip() != "")
    n_tagged = tagged_complete.sum()
    print(f"{n_tagged}/{len(tagged)} clusters tagged")

    if n_tagged > 0:
        # Mean C and drift by theme type
        summary = (
            tagged[tagged_complete]
            .groupby("theme_type")
            .agg(
                n_clusters=("global_cluster_id","count"),
                mean_C=("controversy_score","mean"),
                mean_drift=("cumulative_path_angular","mean"),
                mean_n_posts=("n_posts","mean"),
            )
            .sort_values("mean_C", ascending=False)
        )
        display(summary.round(3))
        ltx.save_table(summary.reset_index(), TABLES_DIR / "theme_type_comparison.tex",
            caption="Theme type comparison: mean controversy and drift by narrative category.",
            label="theme_types")
"""))

cells.append(md("""\
**Manuscript implications (6):** Use the theme taxonomy to characterise which narrative
*types* (legitimation, accusation, counter-narrative, etc.) are most contested (high C)
and most volatile (high drift). These are the primary qualitative descriptors in the
results section.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Results Summary
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Results Summary"))

cells.append(code("""\
summary_lines = ["# Validation Analysis Results Summary\\n"]

for case in CASES:
    summary_lines.append(f"## {case.capitalize()}\\n")
    if case in all_stance:
        df = all_stance[case]
        c_med  = df["C_correct"].median()
        c_mean = df["C_correct"].mean()
        reliable_df = df[df["C_tilde_reliable"]]
        summary_lines.append(
            f"- Controversy: median C={c_med:.3f}  mean C={c_mean:.3f}  "
            f"({len(reliable_df)}/{len(df)} clusters have reliable C_tilde)\\n"
        )
    if case in drift_all:
        d = drift_all[case][drift_all[case]["n_windows"] > 1]
        summary_lines.append(
            f"- Drift: median cumulative={d['cumulative_path_angular'].median():.3f} rad  "
            f"median net={d['net_displacement_angular'].median():.3f} rad\\n"
        )
    if case in coherence_results:
        r = coherence_results[case]
        summary_lines.append(
            f"- Coherence: Cohen's d={r['cohens_d']:.3f}  "
            f"{r['pct_above_null']:.1%} of clusters above null\\n"
        )
    summary_lines.append("\\n")

summary_text = "".join(summary_lines)
summary_path = Path("outputs/RESULTS_SUMMARY.md")
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(summary_text, encoding="utf-8")
print(f"Results summary → {summary_path}")
print("\\n" + summary_text)
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Assemble and write the notebook
# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells

output_path = Path("validation_analysis.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written → {output_path}  ({len(cells)} cells)")
