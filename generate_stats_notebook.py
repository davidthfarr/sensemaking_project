"""
Generate analysis_stats.ipynb — the paper's single source of truth for statistics.

Run once on the analysis server:
    python generate_stats_notebook.py

The notebook reads cluster_metrics.csv produced by compare_cases.py, applies
off-topic filtering, corrects directedness, and emits analysis_stats.json plus
final figures for the paper.
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


# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Analysis Statistics — Conflict IE Paper
*Single source of truth for every number that appears in the paper.*

**Pipeline:** embed → HDBSCAN rolling windows → Hungarian alignment → GPT theme + stance
**Cases:** Venezuela · Iran · Russia-Ukraine
**Paper:** arXiv 2603.17617

---

| Section | Description |
|---------|-------------|
| 1 | Load & reconcile cluster_metrics.csv |
| 2 | Off-topic filtering |
| 3 | Directedness corrections |
| 4 | Region fragility sweep |
| 5 | Cross-case statistics |
| 6 | Near-duplicate theme detection |
| 7 | Stratified annotation sample |
| 8 | Regenerate figures (filtered data) |
| 9 | Export stats JSON + final CSV |

Run sections in order. `on_topic` is a hard gate — sections 3–9 operate only on
`df[df.on_topic]`.
"""))


# ── Section 1: Load & reconcile ───────────────────────────────────────────────
cells.append(md("## 1  Load & Reconcile `cluster_metrics.csv`"))

cells.append(code("""\
import sys, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 30)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(".")
METRICS_CSV = ROOT / "analysis" / "figures" / "cluster_metrics.csv"
OUT_DIR     = ROOT / "outputs" / "figures"
STATS_JSON  = ROOT / "analysis_stats.json"
SAMPLE_CSV  = ROOT / "annotation_sample.csv"
FINAL_CSV   = ROOT / "cluster_metrics_final.csv"
OFFTOPIC_CSV = ROOT / "offtopic_flagged.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Central stats accumulator — every citable number goes here
STATS = {}

assert METRICS_CSV.exists(), f"Run compare_cases.py first to generate {METRICS_CSV}"
df_raw = pd.read_parquet if METRICS_CSV.suffix == ".parquet" else pd.read_csv
df = pd.read_csv(METRICS_CSV)
print(f"Loaded {len(df):,} cluster-case rows from {METRICS_CSV}")
print(df.dtypes)
"""))

cells.append(code("""\
# ── Per-case summary ──────────────────────────────────────────────────────────
CASE_WINDOWS_CONFIG = {
    "venezuela": {"window_h": 8,   "step_h": 4,   "n_days": 22},
    "iran":      {"window_h": 24,  "step_h": 8,   "n_days": 76},
    "russia":    {"window_h": 168, "step_h": 24,  "n_days": 621},
}

for case, cfg in CASE_WINDOWS_CONFIG.items():
    sub = df[df["case"] == case]
    n_clusters    = len(sub)
    n_posts_total = sub["n_posts"].sum()
    n_windows     = sub["n_windows_total"].iloc[0] if "n_windows_total" in sub.columns else "?"
    print(f"\\n── {case} ──────────────────────────────")
    print(f"  clusters       : {n_clusters}")
    print(f"  total window-assignments: {n_posts_total:,}")
    print(f"  n_windows_total: {n_windows}")
    print(f"  config         : {cfg['window_h']}h window / {cfg['step_h']}h step → "
          f"{cfg['n_days']} days")
    # Venezuela reconciliation: 8h window, 4h step → 6 windows/day × 22 days = 132 windows
    if case == "venezuela":
        expected = int((cfg["n_days"] * 24 / cfg["step_h"]) - (cfg["window_h"] / cfg["step_h"]) + 1)
        print(f"  expected windows (formula): {expected}")
    STATS[f"{case}_n_clusters_raw"] = n_clusters
    STATS[f"{case}_n_posts_raw"]    = int(n_posts_total)
"""))


# ── Section 2: Off-topic filtering ────────────────────────────────────────────
cells.append(md("""\
## 2  Off-Topic Filtering

Rules live in `analysis/offtopic_rules.py` (version-controlled, not inline regex).
Expected counts are advisory — the cell below prints actual vs expected and warns
when actual count is below 50 % of expected.

**Russia categories**
- `tigray_eritrea_ethiopia` (~41 expected): Horn of Africa conflict
- `spam_recovery` (~29 expected): @spikeqr, account-recovery bots
- `domestic_political` (~20 expected, **switchable**): Jan 6, Tucker Carlson,
  Griner, Ohio derailment, US border policy, gas prices, Dobbs/abortion,
  DeSantis, school shootings, student loans, Bigg Boss, My Chemical Romance

**Venezuela categories**
- `crypto_promo` (~14 expected): PURK ecosystem (×2), XRP airdrop (×3),
  Tether/USDT, WhatsApp/Telegram stock tips (×2), AI model creation promotion,
  "Free crypto guidance", "Crypto trading advice", "Crypto market insights",
  "Market observations / stocks and crypto", "Stock market insights linked to Venezuela"
- **Manual override — cluster 73** ("Prosecution of dissenting voices against Russian
  government actions"): content is about Russia, off-topic for Venezuela. Prepopulated
  in `offtopic_overrides.csv`.

**Iran categories**
- `entertainment_sports` (~0–3 expected): Indian cricket/IPL, Bollywood, K-pop —
  Iran corpus appears genuinely cleaner; low count is expected but must be *measured*
- `markets_crypto` (~0–5 expected): crypto trading signals, investment spam
  (excludes Iran-sanctions crypto evasion content, which IS on-topic)

This cell runs off-topic classification **twice** — with and without domestic_political —
and prints both. The `domestic_political=False` version is used as the hard gate for
sections 3–9. The paper must state this choice and cite both numbers.
"""))

cells.append(code("""\
import sys
sys.path.insert(0, str(ROOT))
from analysis.offtopic_rules import apply_to_dataframe, RULES, classify_cluster

# ── Helper: apply rules and return annotated copy ────────────────────────────
def _filter(df_in, include_dp):
    dfs = []
    for case in df_in["case"].unique():
        sub = df_in[df_in["case"] == case].copy()
        apply_to_dataframe(sub, case=case, theme_col="theme",
                           include_domestic_political=include_dp)
        dfs.append(sub)
    return pd.concat(dfs, ignore_index=True)

# ── Expected counts (from manual inspection; warn if actual < 50% of expected) ─
EXPECTED_COUNTS = {r["case"] + "/" + r["category"]: r.get("expected_count", 0) for r in RULES}
WARN_THRESHOLD  = 0.50   # actual < threshold × expected → print warning

def _check_expected(df_ann, label=""):
    counts = df_ann[~df_ann.on_topic].groupby(["case","offtopic_category"]).size()
    print(f"\\n{'─'*55}")
    print(f"Off-topic match counts ({label}):")
    print(f"  {'case/category':<42} {'actual':>7} {'expected':>9} {'status':>7}")
    all_ok = True
    for key, expected in EXPECTED_COUNTS.items():
        case, cat = key.split("/")
        actual = counts.get((case, cat), 0)
        if expected > 0 and actual < WARN_THRESHOLD * expected:
            status = "WARN"
            all_ok = False
        else:
            status = "ok"
        print(f"  {key:<42} {actual:>7} {expected:>9} {status:>7}")
    if not all_ok:
        print("  ** WARN: some categories are well below expected — check patterns **")

# ── Run 1: domestic_political EXCLUDED (paper default) ───────────────────────
df_excl = _filter(df, include_dp=False)
_check_expected(df_excl, label="domestic_political=False [paper default]")

# ── Run 2: domestic_political INCLUDED (sensitivity) ─────────────────────────
df_incl = _filter(df, include_dp=True)
_check_expected(df_incl, label="domestic_political=True [sensitivity]")

# ── Sensitivity summary: how many clusters / posts differ ────────────────────
n_dp_clusters = (df_excl.on_topic.astype(int) - df_incl.on_topic.astype(int)).abs().sum()
n_dp_posts    = df_excl.loc[~df_excl.on_topic & df_excl.offtopic_category.eq("domestic_political"),
                             "n_posts"].sum()
print(f"\\nDomestic-political sensitivity: {n_dp_clusters} clusters, {n_dp_posts:,} posts")
print("  Paper must report both versions and justify the choice.")
STATS["russia_domestic_political_n_clusters"] = int(n_dp_clusters)
STATS["russia_domestic_political_n_posts"]    = int(n_dp_posts)

# ── Use exclusion version as hard gate for all downstream sections ────────────
df = df_excl.copy()

# Write full flagged list for hand review
offtopic_df = df[~df.on_topic][
    ["case", "global_cluster_id", "theme", "offtopic_category", "n_posts"]
].copy().sort_values(["case", "offtopic_category", "n_posts"], ascending=[True, True, False])
offtopic_df.to_csv(OFFTOPIC_CSV, index=False)
print(f"\\nWrote {len(offtopic_df)} off-topic rows to {OFFTOPIC_CSV} for hand review")
"""))

cells.append(code("""\
# ── Manual overrides ─────────────────────────────────────────────────────────
# Edit analysis/offtopic_overrides.csv to flip individual clusters:
#   case, global_cluster_id, on_topic_override (True/False), offtopic_category
OVERRIDES_CSV = ROOT / "analysis" / "offtopic_overrides.csv"

# Known manual overrides prepopulated at stub creation time
INITIAL_OVERRIDES = [
    # Venezuela cluster 73: theme is about "Prosecution of dissenting voices against
    # Russian government actions" — content concerns Russia, off-topic for Venezuela.
    {"case": "venezuela", "global_cluster_id": 73,
     "on_topic_override": False, "offtopic_category": "manual_russia_content"},
]

if not OVERRIDES_CSV.exists():
    pd.DataFrame(INITIAL_OVERRIDES).to_csv(OVERRIDES_CSV, index=False)
    print(f"Created {OVERRIDES_CSV} with {len(INITIAL_OVERRIDES)} prepopulated override(s).")
    print("  Edit this file to add or reverse individual cluster decisions.")

overrides = pd.read_csv(OVERRIDES_CSV)
overrides["on_topic_override"] = overrides["on_topic_override"].astype(bool)
applied = 0
for _, row in overrides.iterrows():
    mask = (df["case"] == row["case"]) & (df["global_cluster_id"] == row["global_cluster_id"])
    if mask.sum() == 0:
        print(f"  WARNING: override for {row['case']} cluster {row['global_cluster_id']} not found in df")
        continue
    df.loc[mask, "on_topic"] = row["on_topic_override"]
    if not row["on_topic_override"]:
        df.loc[mask, "offtopic_category"] = row.get("offtopic_category", "manual_override")
    applied += 1
print(f"Applied {applied} manual override(s) from {OVERRIDES_CSV}")

# ── Hard gate: all downstream work uses df_on ────────────────────────────────
df_on = df[df.on_topic].copy()

print()
print(f"  {'case':<12} {'raw':>6} {'filtered':>9} {'on-topic':>9} {'posts_removed':>14}")
for case in sorted(df["case"].unique()):
    n_raw  = (df["case"] == case).sum()
    n_on   = (df_on["case"] == case).sum()
    n_off  = n_raw - n_on
    p_raw  = df.loc[df["case"] == case, "n_posts"].sum()
    p_on   = df_on.loc[df_on["case"] == case, "n_posts"].sum()
    STATS[f"{case}_n_clusters_offtopic"] = int(n_off)
    STATS[f"{case}_n_clusters_ontopic"]  = int(n_on)
    STATS[f"{case}_n_posts_offtopic"]    = int(p_raw - p_on)
    print(f"  {case:<12} {n_raw:>6} {n_off:>9} {n_on:>9} {p_raw-p_on:>14,}")

print()
print("Per-case, per-category breakdown:")
print(df[~df.on_topic].groupby(["case","offtopic_category"])
      .agg(n_clusters=("global_cluster_id","count"), n_posts=("n_posts","sum"))
      .to_string())
"""))


# ── Section 3: Directedness corrections ───────────────────────────────────────
cells.append(md("""\
## 3  Directedness Corrections

- Clusters with `n_active_windows < 3` have directedness set to NaN
  (with only 1 step, cumulative_path = net_displacement → directedness = 1 by definition)
- `directedness_normalized = directedness × √n_active_windows`
  - 1.0 = consistent with a random walk
  - < 1 = sub-diffusive / stable semantic anchor
  - > 1 = directed systematic drift
"""))

cells.append(code("""\
MIN_WINDOWS_DIRECTEDNESS = 3  # require at least this many active windows

# Apply to full df (before on_topic gate) so the column exists everywhere
df["directedness"] = df["directedness"].where(df["n_active_windows"] >= MIN_WINDOWS_DIRECTEDNESS)
df["directedness_normalized"] = (
    df["directedness"] * np.sqrt(df["n_active_windows"])
)
# Propagate to df_on
df_on = df[df.on_topic].copy()

# ── Spot checks ───────────────────────────────────────────────────────────────
spot_checks = [
    ("venezuela", 33, "~0.59"),
    ("russia",   170, "~0.35"),
    ("iran",      70, "~0.38"),
]
print("Directedness spot checks (directedness_normalized):")
for case, gid, expected in spot_checks:
    row = df_on[(df_on.case == case) & (df_on.global_cluster_id == gid)]
    if len(row):
        val = row["directedness_normalized"].iloc[0]
        n_w = row["n_active_windows"].iloc[0]
        print(f"  {case} cluster {gid}: {val:.3f} (expected {expected}, n_active_windows={n_w})")
    else:
        print(f"  {case} cluster {gid}: NOT FOUND in on-topic set")

print()
for case in df_on["case"].unique():
    sub = df_on[df_on.case == case]
    n_degenerate = (sub.n_active_windows < MIN_WINDOWS_DIRECTEDNESS).sum()
    STATS[f"{case}_n_degenerate_directedness"] = int(n_degenerate)
    print(f"{case}: {n_degenerate} clusters with directedness=NaN (n_active_windows < {MIN_WINDOWS_DIRECTEDNESS})")
"""))


# ── Section 4: Region fragility sweep ─────────────────────────────────────────
cells.append(md("""\
## 4  Region Fragility Sweep

Sweeps `C_THRESH × PN_THRESH` to assess how sensitive region assignments are to
boundary choice. Flags clusters near the threshold boundaries as fragile.

**Known boundary cases:**
- Russia cluster 49 (C ≈ 0.388, ~27 k posts) — straddles C_THRESH = 0.40
- Russia cluster 170 (C ≈ 0.496, ~17 k posts) — straddles C_THRESH = 0.50
"""))

cells.append(code("""\
import itertools

C_THRESHOLDS  = [0.30, 0.35, 0.40, 0.45, 0.50]
PN_THRESHOLDS = [0.40, 0.50, 0.60]
DEFAULT_C_THRESH  = 0.40
DEFAULT_PN_THRESH = 0.50

# ── CORRECTED region assignment: contested tested BEFORE fact-relaying ─────────
# Paper definition: a cluster is contested if C >= C_THRESH, regardless of
# p_neutral; only non-contested clusters can be fact-relaying.
# Bug (old): p_neutral >= pn_thresh was tested first, so high-C / high-neutral
# clusters were misclassified as fact-relaying instead of contested.
def assign_region(c, p_neutral, c_thresh, pn_thresh):
    if c >= c_thresh:
        return "contested"
    if p_neutral >= pn_thresh:
        return "fact-relaying"
    return "consolidated"

# Old (wrong) ordering — kept to measure impact
def _assign_region_old(c, p_neutral, c_thresh, pn_thresh):
    if p_neutral >= pn_thresh:
        return "fact-relaying"
    if c >= c_thresh:
        return "contested"
    return "consolidated"

# ── Measure impact of precedence fix at default thresholds ────────────────────
n_ordering_diff = sum(
    _assign_region_old(r["C"], r["p_neutral"], DEFAULT_C_THRESH, DEFAULT_PN_THRESH) !=
    assign_region(r["C"], r["p_neutral"], DEFAULT_C_THRESH, DEFAULT_PN_THRESH)
    for _, r in df_on.iterrows()
)
print(f"Clusters reassigned by precedence fix (p_neutral-first -> C-first): "
      f"{n_ordering_diff} / {len(df_on)}")
STATS["n_clusters_region_precedence_fix"] = int(n_ordering_diff)

# ── Fragility sweep ───────────────────────────────────────────────────────────
records = []
for c_thresh, pn_thresh in itertools.product(C_THRESHOLDS, PN_THRESHOLDS):
    for _, row in df_on.iterrows():
        reg = assign_region(row["C"], row["p_neutral"], c_thresh, pn_thresh)
        records.append({
            "case": row["case"],
            "global_cluster_id": row["global_cluster_id"],
            "c_thresh": c_thresh,
            "pn_thresh": pn_thresh,
            "region": reg,
        })

sweep_df = pd.DataFrame(records)

n_configs = len(C_THRESHOLDS) * len(PN_THRESHOLDS)
fragility = (
    sweep_df.groupby(["case", "global_cluster_id"])["region"]
    .nunique()
    .reset_index()
    .rename(columns={"region": "n_region_assignments"})
)
fragile = fragility[fragility.n_region_assignments > 1].merge(
    df_on[["case", "global_cluster_id", "C", "p_neutral", "n_posts", "theme"]],
    on=["case", "global_cluster_id"],
)
fragile = fragile.sort_values(["case", "n_posts"], ascending=[True, False])

print(f"Fragile clusters (vary across {n_configs} threshold combos): {len(fragile)}")
print(fragile[["case","global_cluster_id","C","p_neutral","n_posts","theme",
               "n_region_assignments"]].to_string(index=False))
STATS["n_fragile_clusters"] = int(len(fragile))

# ── Extract recomputed default assignment ─────────────────────────────────────
ref_sweep = sweep_df[
    (sweep_df.c_thresh  == DEFAULT_C_THRESH) &
    (sweep_df.pn_thresh == DEFAULT_PN_THRESH)
][["case", "global_cluster_id", "region"]].rename(columns={"region": "region_recomputed"})

df_on = df_on.merge(ref_sweep, on=["case", "global_cluster_id"], how="left")

# Compare recomputed region to the CSV's original 'region' column (from compare_cases.py)
if "region" in df_on.columns:
    n_csv_changed = (df_on["region"] != df_on["region_recomputed"]).sum()
    print(f"\\nClusters whose region changed vs cluster_metrics.csv original: "
          f"{n_csv_changed} / {len(df_on)}")
    STATS["n_clusters_region_vs_csv"] = int(n_csv_changed)

# Always overwrite — never leave the stale CSV region in place
df_on["region"] = df_on["region_recomputed"]
df_on = df_on.drop(columns=["region_recomputed"])

print("\\nRegion distribution at default thresholds (corrected precedence):")
dist = df_on.groupby(["case","region"])["n_posts"].agg(n_clusters="count", n_posts="sum")
print(dist.to_string())
"""))


# ── Section 5: Cross-case statistics ─────────────────────────────────────────
cells.append(md("""\
## 5  Cross-Case Statistics

- Region distribution: chi-square test on cluster counts; bootstrap CIs on volume (post) shares
- OLS persistence fits: log10(n_posts) → persistence_frac; compare R² pre/post filter
- Region × persistence-residual contingency table
"""))

cells.append(code("""\
from scipy.stats import chi2_contingency
import warnings

# ── Region distribution chi-square ───────────────────────────────────────────
region_counts = (
    df_on.groupby(["case", "region"])["global_cluster_id"]
    .count()
    .unstack(fill_value=0)
)
print("Region cluster counts:")
print(region_counts)

# Only test if ≥ 2 cases and ≥ 2 regions present
if region_counts.shape[0] >= 2 and region_counts.shape[1] >= 2:
    chi2, p_chi2, dof, expected = chi2_contingency(region_counts.values)
    print(f"\\nChi-square (region × case, cluster counts): χ²={chi2:.2f}, df={dof}, p={p_chi2:.4f}")
    STATS["region_chisq_chi2"]  = round(chi2, 3)
    STATS["region_chisq_p"]     = round(p_chi2, 4)
    STATS["region_chisq_dof"]   = dof
else:
    print("Insufficient categories for chi-square — skipping.")
"""))

cells.append(code("""\
# ── Bootstrap CIs on volume (post) shares ────────────────────────────────────
rng = np.random.default_rng(42)
N_BOOT = 2000

def bootstrap_share(sub_df, n_boot, rng):
    data = sub_df["n_posts"].values.astype(float)
    total = data.sum()
    region_labels = sub_df["region"].values
    boot_shares = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(data), size=len(data))
        boot_total = data[idx].sum()
        for reg in np.unique(region_labels):
            s = data[idx][region_labels[idx] == reg].sum() / boot_total if boot_total > 0 else 0
            boot_shares.setdefault(reg, []).append(s)
    return {reg: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for reg, v in boot_shares.items()}

print("\\nVolume (post) share bootstrap 95% CIs:")
for case in sorted(df_on["case"].unique()):
    sub = df_on[df_on.case == case]
    total_posts = sub["n_posts"].sum()
    cis = bootstrap_share(sub, N_BOOT, rng)
    print(f"  {case}:")
    for reg in sorted(cis):
        lo, hi = cis[reg]
        obs = sub[sub.region == reg]["n_posts"].sum() / total_posts
        STATS[f"{case}_{reg}_volume_share"]    = round(obs, 4)
        STATS[f"{case}_{reg}_volume_ci_lo"]    = round(lo, 4)
        STATS[f"{case}_{reg}_volume_ci_hi"]    = round(hi, 4)
        print(f"    {reg}: {obs:.3f}  [{lo:.3f}, {hi:.3f}]")
"""))

cells.append(code("""\
# ── OLS persistence fits: pre-filter vs post-filter R² ───────────────────────
from sklearn.linear_model import LinearRegression

def ols_r2(sub):
    valid = sub.dropna(subset=["n_posts", "persistence_frac"])
    if len(valid) < 3:
        return np.nan
    X = np.log10(valid["n_posts"].clip(lower=1)).values.reshape(-1, 1)
    y = valid["persistence_frac"].values
    return LinearRegression().fit(X, y).score(X, y)

print("OLS R² (log10(n_posts) → persistence_frac):")
print(f"  {'case':<12} {'pre-filter':>12} {'post-filter':>12}")
for case in sorted(df["case"].unique()):
    r2_pre  = ols_r2(df[df.case == case])
    r2_post = ols_r2(df_on[df_on.case == case])
    STATS[f"{case}_ols_r2_pre"]  = round(r2_pre, 3)
    STATS[f"{case}_ols_r2_post"] = round(r2_post, 3)
    print(f"  {case:<12} {r2_pre:>12.3f} {r2_post:>12.3f}")

# Expected pre-filter values from spec: Venezuela 0.81, Iran 0.59, Russia 0.50
"""))

cells.append(code("""\
# ── Region × persistence-residual contingency table ──────────────────────────
# Residuals already in cluster_metrics.csv if compare_cases.py was run;
# recompute here from filtered data for correctness.

from sklearn.linear_model import LinearRegression

def add_residuals(sub):
    sub = sub.copy()
    valid_mask = sub["n_posts"].notna() & sub["persistence_frac"].notna()
    X = np.log10(sub.loc[valid_mask, "n_posts"].clip(lower=1)).values.reshape(-1, 1)
    y = sub.loc[valid_mask, "persistence_frac"].values
    if len(y) < 3:
        sub["persistence_residual_std"] = np.nan
        return sub
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    resid  = y - y_pred
    std    = resid.std(ddof=1)
    sub.loc[valid_mask, "persistence_residual_std"] = resid / std if std > 0 else 0
    return sub

dfs_with_resid = []
for case in df_on["case"].unique():
    dfs_with_resid.append(add_residuals(df_on[df_on.case == case]))
df_on = pd.concat(dfs_with_resid, ignore_index=True)

# Tertile bins for residual: low / mid / high
df_on["resid_tertile"] = pd.qcut(
    df_on["persistence_residual_std"].dropna(), q=3,
    labels=["low", "mid", "high"], duplicates="drop"
)
contingency = pd.crosstab(df_on["region"], df_on["resid_tertile"])
print("\\nRegion × persistence residual tertile:")
print(contingency)
if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
    chi2, p, dof, _ = chi2_contingency(contingency.values)
    STATS["region_resid_chisq_chi2"] = round(chi2, 3)
    STATS["region_resid_chisq_p"]    = round(p, 4)
    print(f"Chi-square: χ²={chi2:.2f}, df={dof}, p={p:.4f}")
"""))


# ── Section 5b: Per-case narrative statistics ──────────────────────────────────
cells.append(md("""\
## 5b  Per-Case Narrative Statistics

One function called once per case; output follows the prose template order so
it can be read top-to-bottom while drafting each case section.

Every quantity is written to `STATS` under `{case}_{key}`.

**Window-assignments vs unique posts**: `df_on.n_posts` is unique posts per cluster
(deduplicated by post_id within each cluster). Its cross-cluster sum exceeds the
true unique-post count because a post that appears in overlapping windows can belong
to multiple clusters. Cross-cluster sums are labelled *window-assignments* here;
per-cluster `n_posts` values are *posts in cluster*.
"""))

cells.append(code("""\
from sklearn.linear_model import LinearRegression as _LR

_CASE_LABEL = {'venezuela': 'Venezuela', 'iran': 'Iran', 'russia': 'Russia-Ukraine'}


def report_case(case, MIN_POSTS_REVIEW=10,
                N_TOP_VOL=8, N_TOP_C=5, N_TOP_RESID=5, N_FR_SAMPLE=5):
    '''
    Print per-case narrative statistics in prose order; write all quantities
    to STATS. Reads df, df_on, STATS, ROOT, CASE_WINDOWS_CONFIG,
    bootstrap_share from notebook global scope.
    '''
    sub      = df_on[df_on['case'] == case].copy()
    sub_raw  = df[df['case'] == case].copy()
    cfg      = CASE_WINDOWS_CONFIG[case]
    step_h   = cfg['step_h']
    n_days   = cfg['n_days']
    SEP      = '-' * 60

    def SK(key, val):
        STATS[f'{case}_{key}'] = val
        return val

    print(f'\\n{"=" * 62}')
    print(f'  {_CASE_LABEL[case]}')
    print(f'{"=" * 62}')

    # ── 1. Corpus and parameters ──────────────────────────────────────────────
    print(f'\\n  1. Corpus and parameters')
    print(f'  {SEP}')

    gc_path   = ROOT / 'data' / 'evaluated' / case / 'global_clusters.parquet'
    repr_path = ROOT / 'data' / 'processed'  / case / 'posts_repr.parquet'

    if gc_path.exists():
        gc = pd.read_parquet(gc_path, columns=['post_id', 'global_cluster_id', 'is_noise'])
        gc['post_id']  = gc['post_id'].astype(str)
        gc['is_noise'] = gc['is_noise'].fillna(False).astype(bool)
        n_corpus      = gc['post_id'].nunique()
        gc_nn         = gc[~gc['is_noise'] & gc['global_cluster_id'].notna()]
        n_clustered   = gc_nn['post_id'].nunique()
        n_assignments = len(gc_nn)
        noise_rate    = 1.0 - n_clustered / n_corpus if n_corpus else float('nan')
        mean_wposts   = n_assignments / n_clustered  if n_clustered else float('nan')
        SK('n_posts_corpus',           int(n_corpus))
        SK('n_posts_clustered_unique', int(n_clustered))
        SK('n_window_assignments',     int(n_assignments))
        SK('noise_rate',               round(float(noise_rate), 4))
        SK('mean_windows_per_post',    round(float(mean_wposts), 2))
        print(f'  Corpus unique posts          : {n_corpus:,}')
        print(f'  Clustered unique posts       : {n_clustered:,}  '
              f'(noise rate: {noise_rate:.1%})')
        print(f'  Total window-assignments     : {n_assignments:,}')
        print(f'  Mean windows per post        : {mean_wposts:.2f}')
    else:
        print(f'  WARNING: {gc_path} not found — corpus counts unavailable')

    n_win = (sub['n_windows_total'].iloc[0]
             if 'n_windows_total' in sub.columns and len(sub) else '?')
    SK('n_windows_total_reported', n_win if n_win != '?' else None)
    print(f'  Window / step                : {cfg["window_h"]}h / {step_h}h')
    print(f'  n_windows_total              : {n_win}')
    print(f'  Collection span              : {n_days} days')

    # ── 2. Cluster yield ──────────────────────────────────────────────────────
    print(f'\\n  2. Cluster yield')
    print(f'  {SEP}')

    n_raw     = len(sub_raw)
    n_ontopic = len(sub)
    n_off     = n_raw - n_ontopic
    SK('n_clusters_raw',      n_raw)
    SK('n_clusters_ontopic',  n_ontopic)
    SK('n_clusters_offtopic', n_off)
    print(f'  Raw clusters                 : {n_raw}')
    print(f'  Off-topic flagged            : {n_off}')

    if 'on_topic' in sub_raw.columns and 'offtopic_category' in sub_raw.columns:
        off_sub = sub_raw[
            ~sub_raw['on_topic'] &
            sub_raw['offtopic_category'].notna() &
            (sub_raw['offtopic_category'] != '')
        ]
        for cat, cnt in (off_sub.groupby('offtopic_category').size()
                                .sort_values(ascending=False).items()):
            SK(f'n_clusters_offtopic_{cat}', int(cnt))
            print(f'    {cat:<34}: {cnt}')

    print(f'  On-topic clusters            : {n_ontopic}')

    # ── 3. Size distribution ──────────────────────────────────────────────────
    print(f'\\n  3. Size distribution  (window-assignments per cluster)')
    print(f'  {SEP}')

    sz = sub['n_posts']
    q  = sz.quantile([0, 0.25, 0.5, 0.75, 1.0])
    mu = float(sz.mean())
    SK('size_min',    int(q[0.00]));  SK('size_p25',    int(q[0.25]))
    SK('size_median', int(q[0.50]));  SK('size_mean',   round(mu, 1))
    SK('size_p75',    int(q[0.75]));  SK('size_max',    int(q[1.00]))
    print(f'  min={int(q[0]):,}  p25={int(q[0.25]):,}  median={int(q[0.5]):,}  '
          f'mean={mu:.1f}  p75={int(q[0.75]):,}  max={int(q[1.0]):,}')
    skew_note = 'right-skewed (large clusters inflate mean)' if mu > int(q[0.5]) * 1.1 else 'near-symmetric'
    print(f'  Median / mean = {int(q[0.5]) / mu:.2f}  ({skew_note})')

    if 'persistence_windows' in sub.columns:
        n_single    = int((sub['persistence_windows'] == 1).sum())
        frac_single = n_single / len(sub) if len(sub) else 0.0
        SK('n_single_window_clusters', n_single)
        SK('frac_single_window',       round(frac_single, 4))
        print(f'  Single-window clusters       : {n_single}  ({frac_single:.1%} of on-topic)')

    # ── 4. Persistence and drift ──────────────────────────────────────────────
    print(f'\\n  4. Persistence and drift')
    print(f'  {SEP}')

    p_med_w = float(sub['persistence_windows'].median())
    p_med_f = float(sub['persistence_frac'].median())
    SK('persistence_windows_median', round(p_med_w, 2))
    SK('persistence_frac_median',    round(p_med_f, 4))
    print(f'  Median persistence           : {p_med_w:.1f} windows  ({p_med_f:.3f} of case span)')

    valid_p = sub.dropna(subset=['n_posts', 'persistence_frac'])
    if len(valid_p) >= 3:
        X_p = np.log10(valid_p['n_posts'].clip(lower=1)).values.reshape(-1, 1)
        y_p = valid_p['persistence_frac'].values
        ols = _LR().fit(X_p, y_p)
        slp = float(ols.coef_[0])
        r2  = float(ols.score(X_p, y_p))
        SK('ols_persistence_slope', round(slp, 4))
        SK('ols_persistence_r2',    round(r2, 3))
        print(f'  OLS persistence ~ log10(n)   : slope={slp:.3f}  R²={r2:.3f}')

    wpd = 24.0 / step_h
    sub['drift_rate_per_day'] = sub['drift_rate'] * wpd
    med_drift = float(sub['drift_rate_per_day'].median())
    SK('drift_rate_rad_per_day_median', round(med_drift, 5))
    print(f'  Median drift rate            : {med_drift:.4f} rad/day  '
          f'({float(sub["drift_rate"].median()):.5f} rad/window x {wpd:.1f} win/day)')

    dn       = sub['directedness_normalized']
    dn_undef = int(dn.isna().sum())
    med_dn   = float(dn.dropna().median()) if dn.notna().any() else float('nan')
    SK('directedness_normalized_median',    round(med_dn, 3) if not np.isnan(med_dn) else None)
    SK('directedness_normalized_undefined', dn_undef)
    dn_str = f'{med_dn:.3f}' if not np.isnan(med_dn) else 'N/A'
    print(f'  Median directedness (norm.)  : {dn_str}  '
          f'(undefined for {dn_undef} clusters with < 3 active windows)')

    # ── 5. Region distribution ────────────────────────────────────────────────
    print(f'\\n  5. Region distribution')
    print(f'  {SEP}')

    tot     = float(sub['n_posts'].sum())
    rng_bs  = np.random.default_rng(42)
    cis     = bootstrap_share(sub, 2000, rng_bs)
    REGS    = ['contested', 'consolidated', 'fact-relaying']
    print(f'  {"region":<16}  {"n_clusters":>10}  {"clust%":>7}  '
          f'{"vol_share":>9}  {"95% CI":<20}  {"mean_size":>9}')
    for reg in REGS:
        rs   = sub[sub['region'] == reg]
        nc   = len(rs)
        cp   = nc / len(sub) if len(sub) else 0.0
        vs   = float(rs['n_posts'].sum()) / tot if tot else 0.0
        lo, hi = cis.get(reg, (float('nan'), float('nan')))
        msz  = float(rs['n_posts'].mean()) if nc else 0.0
        SK(f'region_{reg}_n_clusters', nc)
        SK(f'region_{reg}_cluster_pct', round(cp, 4))
        SK(f'region_{reg}_vol_share',   round(vs, 4))
        SK(f'region_{reg}_vol_ci_lo',   round(lo, 4) if not np.isnan(lo) else None)
        SK(f'region_{reg}_vol_ci_hi',   round(hi, 4) if not np.isnan(hi) else None)
        SK(f'region_{reg}_mean_size',   round(msz, 1))
        ci_str = f'[{lo:.3f}, {hi:.3f}]' if not np.isnan(lo) else '[N/A]'
        print(f'  {reg:<16}  {nc:>10}  {cp:>6.1%}  {vs:>9.3f}  {ci_str:<20}  {msz:>9.0f}')

    # ── 6. Top clusters for review ────────────────────────────────────────────
    print(f'\\n  6. Top clusters for qualitative review')
    print(f'  {SEP}')

    RC = [c for c in ['global_cluster_id', 'theme', 'n_posts',
                       'persistence_windows', 'C', 'p_neutral', 'region']
          if c in sub.columns]

    def _tbl(title, frame):
        print(f'\\n  {title}:')
        for row in frame[RC].itertuples(index=False):
            th = str(getattr(row, 'theme', ''))[:52]
            print(f'    [{int(row.global_cluster_id):>3}]  {th:<52}  '
                  f'n={int(row.n_posts):>6}  C={row.C:.2f}  '
                  f'pn={row.p_neutral:.2f}  {row.region}')

    top_vol   = sub.nlargest(N_TOP_VOL, 'n_posts')
    top_c     = sub[sub['n_posts'] >= MIN_POSTS_REVIEW].nlargest(N_TOP_C, 'C')
    top_resid = (sub.dropna(subset=['persistence_residual_std'])
                    .nlargest(N_TOP_RESID, 'persistence_residual_std')
                 if 'persistence_residual_std' in sub.columns else pd.DataFrame())

    _tbl(f'Top {N_TOP_VOL} by window-assignments', top_vol)
    _tbl(f'Top {N_TOP_C} by controversy (C), n_posts >= {MIN_POSTS_REVIEW}', top_c)
    if len(top_resid):
        _tbl(f'Top {N_TOP_RESID} by positive persistence residual', top_resid)

    # ── 7. Qualitative review export ──────────────────────────────────────────
    print(f'\\n  7. Qualitative review export')
    print(f'  {SEP}')

    rng_qr  = np.random.default_rng(97 + abs(hash(case)) % 503)
    fr_pool = sub[sub['region'] == 'fact-relaying']
    n_fr    = min(N_FR_SAMPLE, len(fr_pool))
    fr_rand = (fr_pool.sample(n=n_fr, random_state=int(rng_qr.integers(0, 2**31)))
               if n_fr > 0 else pd.DataFrame())

    parts = [top_vol[RC], top_c[RC]]
    if len(top_resid): parts.append(top_resid[RC])
    if len(fr_rand):   parts.append(fr_rand[RC])
    review = (pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset=['global_cluster_id'])
                .copy())
    review['label_accurate']  = ''
    review['region_accurate'] = ''
    review['reviewer_notes']  = ''

    rev_path = ROOT / f'qualitative_review_{case}.csv'
    review.to_csv(rev_path, index=False)
    print(f'  qualitative_review_{case}.csv       : {len(review)} clusters')

    # Nearest-centroid posts for each selected cluster
    if gc_path.exists() and repr_path.exists():
        gc_full = pd.read_parquet(gc_path, columns=['post_id', 'global_cluster_id', 'is_noise'])
        gc_full['post_id']  = gc_full['post_id'].astype(str)
        gc_full['is_noise'] = gc_full['is_noise'].fillna(False).astype(bool)
        gc_full = gc_full[~gc_full['is_noise'] & gc_full['global_cluster_id'].notna()].copy()
        gc_full['global_cluster_id'] = gc_full['global_cluster_id'].astype(int)

        repr_df = pd.read_parquet(repr_path)
        for alias in ('Resource Id', 'tweet_id', 'tweetid', 'post id', 'postid'):
            if alias in repr_df.columns and 'post_id' not in repr_df.columns:
                repr_df = repr_df.rename(columns={alias: 'post_id'})
        repr_df['post_id'] = repr_df['post_id'].astype(str)
        repr_df = repr_df.drop_duplicates('post_id')

        if 'embedding' in repr_df.columns:
            sel_cids = review['global_cluster_id'].tolist()
            gc_sel   = gc_full[gc_full['global_cluster_id'].isin(sel_cids)]
            repr_sel = (repr_df[repr_df['post_id'].isin(gc_sel['post_id'])]
                               [['post_id', 'text', 'embedding']])
            post_rows = []
            for cid in sel_cids:
                pids   = gc_sel[gc_sel['global_cluster_id'] == cid]['post_id'].unique()
                emb_df = repr_sel[repr_sel['post_id'].isin(pids)].reset_index(drop=True)
                if len(emb_df) == 0:
                    continue
                embs     = np.vstack(emb_df['embedding'].values).astype(float)
                nrm      = np.linalg.norm(embs, axis=1, keepdims=True)
                embs_n   = embs / np.where(nrm == 0, 1.0, nrm)
                centroid = embs_n.mean(axis=0)
                c_nrm    = np.linalg.norm(centroid)
                centroid = centroid / c_nrm if c_nrm > 0 else centroid
                dists    = np.arccos(np.clip(embs_n @ centroid, -1.0, 1.0))
                for rank, idx in enumerate(np.argsort(dists)[:10], 1):
                    post_rows.append({
                        'global_cluster_id': int(cid),
                        'rank': rank,
                        'post_id': emb_df.at[idx, 'post_id'],
                        'dist_to_centroid': round(float(dists[idx]), 4),
                        'text': emb_df.at[idx, 'text'],
                    })
            posts_path = ROOT / f'qualitative_review_{case}_posts.csv'
            pd.DataFrame(post_rows).to_csv(posts_path, index=False)
            print(f'  qualitative_review_{case}_posts.csv : {len(post_rows)} posts')
        else:
            print('  WARNING: embedding column missing in posts_repr.parquet — posts skipped')
    else:
        print('  WARNING: source files missing — centroid posts not exported')


# ── Run for all three cases ───────────────────────────────────────────────────
for case in CASES:
    report_case(case)
"""))


# ── Section 6: Near-duplicate theme detection ─────────────────────────────────
cells.append(md("""\
## 6  Near-Duplicate Theme Detection

String-distance detection for duplicate or near-duplicate theme labels within a case.
Known cases:
- Russia: "Media accountability for the discredited Russian bounties narrative" × 3
- Russia: "Ukraine's legal victory against Russia at the ICJ" × 2
- Iran: Pakistan peace broker theme × 4
"""))

cells.append(code("""\
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False
    print("rapidfuzz not installed — falling back to difflib (slower, less accurate)")
    from difflib import SequenceMatcher

SIM_THRESHOLD = 0.85  # themes with similarity > this are flagged as near-duplicates

dup_records = []
for case in df_on["case"].unique():
    themes = df_on[df_on.case == case][["global_cluster_id", "theme"]].dropna()
    pairs_seen = set()
    for i, (gid_a, t_a) in themes.iterrows():
        for j, (gid_b, t_b) in themes.iterrows():
            if i >= j:
                continue
            key = tuple(sorted([gid_a, gid_b]))
            if key in pairs_seen:
                continue
            pairs_seen.add(key)
            if HAVE_RAPIDFUZZ:
                sim = fuzz.token_sort_ratio(t_a, t_b) / 100.0
            else:
                sim = SequenceMatcher(None, t_a, t_b).ratio()
            if sim >= SIM_THRESHOLD:
                dup_records.append({
                    "case": case,
                    "global_cluster_id_a": gid_a,
                    "global_cluster_id_b": gid_b,
                    "theme_a": t_a,
                    "theme_b": t_b,
                    "similarity": round(sim, 3),
                })

dup_df = pd.DataFrame(dup_records)
print(f"Near-duplicate theme pairs (similarity ≥ {SIM_THRESHOLD}): {len(dup_df)}")
if len(dup_df):
    print(dup_df.to_string(index=False))
    STATS["n_near_duplicate_pairs"] = len(dup_df)
"""))


# ── Section 7: Stratified annotation sample ───────────────────────────────────
cells.append(md("""\
## 7  Stratified Annotation Sample

Stratify by case × stance class × region.
Also inspect p_neutral distribution — high p_neutral clusters may indicate
under-specified themes or genuine consensus topics.
"""))

cells.append(code("""\
# ── p_neutral distribution ────────────────────────────────────────────────────
print("p_neutral distribution by case:")
print(df_on.groupby("case")["p_neutral"].describe().T)

HIGH_NEUTRAL_THRESH = 0.70
for case in df_on["case"].unique():
    sub = df_on[df_on.case == case]
    n_high = (sub.p_neutral >= HIGH_NEUTRAL_THRESH).sum()
    pct    = 100 * n_high / len(sub)
    STATS[f"{case}_n_high_neutral"] = int(n_high)
    print(f"{case}: {n_high} clusters with p_neutral ≥ {HIGH_NEUTRAL_THRESH} ({pct:.1f}%)")
"""))

cells.append(code("""\
# ── Stratified sample ─────────────────────────────────────────────────────────
N_PER_STRATUM = 3
rng2 = np.random.default_rng(99)

# Stance class: assign based on dominant stance (or neutral if p_neutral > 0.5)
def stance_class(row):
    if row["p_neutral"] > 0.50:
        return "neutral-dominant"
    if row["p_support"] >= row["p_oppose"]:
        return "support-dominant"
    return "oppose-dominant"

df_on["stance_class"] = df_on.apply(stance_class, axis=1)

sample_rows = []
for (case, sc, reg), grp in df_on.groupby(["case", "stance_class", "region"]):
    n = min(N_PER_STRATUM, len(grp))
    if n == 0:
        continue
    chosen = grp.sample(n=n, random_state=rng2.integers(0, 2**31))
    sample_rows.append(chosen)

sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
sample_df = sample_df[["case","global_cluster_id","theme","stance_class","region",
                        "p_support","p_oppose","p_neutral","C","n_posts"]].sort_values(
    ["case","region","stance_class"]
)
sample_df.to_csv(SAMPLE_CSV, index=False)
STATS["annotation_sample_n"] = len(sample_df)
print(f"Annotation sample: {len(sample_df)} rows → {SAMPLE_CSV}")
print(sample_df.groupby(["case","region","stance_class"]).size().to_string())
"""))


# ── Section 7b: Post-level annotation sample ─────────────────────────────────
cells.append(md("""\
## 7b  Post-Level Annotation Sample

Exports a dual-annotator package for validating cluster-mode stance labels.

**Design choices:**
- Stratify by (case × model-stance class × region) — 27 strata × 8 posts ≈ 216 posts
- `sample_id` is UUID v5 of `post_id` — stable across reruns, not traceable to
  case or cluster
- Annotation files contain no metadata (no case, region, cluster id, model label)
  to avoid biasing judgment
- `annotation_key.csv` holds all metadata for post-labeling join
- `annotation_a.csv` / `annotation_b.csv` carry identical rows in different random
  orders so the two annotators don't work through the same sequence
- `annotation_instructions.txt` contains the exact gpt-4o-mini system prompt,
  so annotator instructions can match it verbatim

**Note on `gpt_stance`:** per-post cluster-mode stance is not saved by the
pipeline (only cluster-level proportions are). `gpt_stance` here is the cluster's
modal stance (argmax of p_support, p_oppose, p_neutral). The scoring cell compares
adjudicated human labels against this modal label.
"""))

cells.append(code("""\
import uuid as _uuid

POST_SAMPLE_N = 8        # posts per (case x gpt_stance x region) stratum
ANNOT_DIR     = ROOT / "annotation"
ANNOT_DIR.mkdir(exist_ok=True)

# ── Exact system prompt used by PosthocGPTStanceClassifier ────────────────────
_SYSTEM_PROMPT_TEXT = ""\"\\
You are a stance classifier for social media posts.

Given a narrative claim and a numbered list of posts, classify each post's \\
stance toward the claim. Reply with a JSON object in this exact format:
{"stances": ["support", "neutral", "oppose", ...]}

Rules:
- support: the post affirms, endorses, agrees with, or spreads the claim
- oppose: the post rejects, counters, disputes, or contradicts the claim
- neutral: the post is unrelated to the claim, ambiguous, or takes no clear position

Return one label per post, in the same order as the input. Use only the \\
words "support", "oppose", or "neutral".\\
""\"

instructions_text = f""\"ANNOTATION INSTRUCTIONS
=======================

Your task: for each post, decide whether it SUPPORTS, OPPOSES, or is NEUTRAL
toward the stance_target claim shown in the same row.

Labels
------
  support  — the post affirms, endorses, agrees with, or spreads the claim
  oppose   — the post rejects, counters, disputes, or contradicts the claim
  neutral  — the post is unrelated, ambiguous, or takes no clear position

Procedure
---------
1. Read the stance_target (a short narrative claim).
2. Read the post_text.
3. Enter your label in the annotator_stance column (support / oppose / neutral).
4. Use annotator_notes for anything noteworthy — irony, ambiguity, non-English
   text, clearly off-topic content, etc.  Leave blank if straightforward.
5. Do not skip rows; mark genuinely unclear posts as neutral.

The exact system prompt used when classifying these posts with gpt-4o-mini —
match these label definitions verbatim in your instructions to annotators:

---
{_SYSTEM_PROMPT_TEXT}
---
""\"

(ANNOT_DIR / "annotation_instructions.txt").write_text(instructions_text, encoding="utf-8")
print("Wrote annotation_instructions.txt")
print()
print("System prompt printed to output for record:")
print("─" * 60)
print(_SYSTEM_PROMPT_TEXT)
print("─" * 60)

# ── Derive gpt_stance as cluster modal stance ─────────────────────────────────
# Per-post cluster-mode stance is not saved by run_stance_classification.py.
# Modal stance = argmax(p_support, p_oppose, p_neutral) for the cluster.
df_on["gpt_stance"] = df_on.apply(
    lambda r: max(["support", "oppose", "neutral"], key=lambda k: r[f"p_{k}"]),
    axis=1,
)

# ── Load post-level data from pipeline artifacts ──────────────────────────────
post_frames = []
for case in CASES:
    eval_dir  = ROOT / "data" / "evaluated" / case
    repr_path = ROOT / "data" / "processed" / case / "posts_repr.parquet"
    gc_path   = eval_dir / "global_clusters.parquet"

    if not gc_path.exists():
        print(f"  {case}: global_clusters.parquet not found — skipping")
        continue
    if not repr_path.exists():
        print(f"  {case}: posts_repr.parquet not found — skipping")
        continue

    # Post -> cluster assignment: non-noise, one row per post (first window seen)
    gc = pd.read_parquet(gc_path, columns=["post_id", "global_cluster_id", "is_noise"])
    gc["post_id"] = gc["post_id"].astype(str)
    gc = gc[~gc["is_noise"].fillna(False) & gc["global_cluster_id"].notna()].copy()
    gc["global_cluster_id"] = gc["global_cluster_id"].astype(int)
    gc = gc.drop_duplicates(subset=["post_id"], keep="first")

    # Post text
    repr_df = pd.read_parquet(repr_path)
    for alias in ("Resource Id", "tweet_id", "tweetid", "post id", "postid"):
        if alias in repr_df.columns and "post_id" not in repr_df.columns:
            repr_df = repr_df.rename(columns={alias: "post_id"})
    repr_df["post_id"] = repr_df["post_id"].astype(str)
    repr_df = repr_df.drop_duplicates(subset=["post_id"])[["post_id", "text"]]

    merged = gc.merge(repr_df, on="post_id", how="inner")
    merged["case"] = case
    post_frames.append(merged)
    print(f"  {case}: {len(merged):,} post-cluster pairs loaded")

if not post_frames:
    print("\\nNo post-level data found. Ensure global_clusters.parquet and")
    print("posts_repr.parquet exist for each case, then re-run this cell.")
else:
    posts_all = pd.concat(post_frames, ignore_index=True)

    # Join to df_on: adds region, theme, gpt_stance; filters to on-topic clusters
    posts_joined = posts_all.merge(
        df_on[["case", "global_cluster_id", "theme", "region", "gpt_stance"]],
        on=["case", "global_cluster_id"],
        how="inner",
    )
    print(f"\\nPost-level on-topic coverage: {len(posts_joined):,} posts "
          f"across {posts_joined['global_cluster_id'].nunique()} clusters")

    # Stratum distribution
    strat_counts = posts_joined.groupby(["case", "gpt_stance", "region"]).size()
    print("\\nAvailable posts per (case x gpt_stance x region) stratum:")
    print(strat_counts.to_string())

    # ── Stratified sample ────────────────────────────────────────────────────
    rng_annot  = np.random.default_rng(2024)
    sample_rows, stratum_report = [], []

    for (case, stance, region), grp in posts_joined.groupby(["case","gpt_stance","region"]):
        n_avail = len(grp)
        n_take  = min(POST_SAMPLE_N, n_avail)
        chosen  = grp.sample(n=n_take, random_state=rng_annot.integers(0, 2**31))
        sample_rows.append(chosen)
        stratum_report.append({
            "case": case, "gpt_stance": stance, "region": region,
            "n_available": n_avail, "n_sampled": n_take,
            "underpopulated": n_avail < POST_SAMPLE_N,
        })

    sample_all  = pd.concat(sample_rows, ignore_index=True)
    stratum_df  = pd.DataFrame(stratum_report)

    n_full  = (stratum_df.n_sampled == POST_SAMPLE_N).sum()
    n_under = stratum_df.underpopulated.sum()

    print(f"\\nStratum report  (target: {POST_SAMPLE_N} posts each):")
    print(stratum_df.to_string(index=False))
    print(f"\\nTotal posts sampled : {len(sample_all)}")
    print(f"Full strata         : {n_full} / {len(stratum_df)}")
    print(f"Underpopulated      : {n_under} / {len(stratum_df)}")

    STATS["annotation_post_n"]           = len(sample_all)
    STATS["annotation_strata_full"]      = int(n_full)
    STATS["annotation_strata_under"]     = int(n_under)

    # ── Stable sample_id (UUID v5, not derivable from case/cluster) ──────────
    _NS = _uuid.UUID("3b4a9c2e-11f0-4e87-b2d8-0f9c1a7e5342")
    sample_all["sample_id"] = sample_all["post_id"].apply(
        lambda pid: str(_uuid.uuid5(_NS, str(pid)))
    )

    # ── Key file: metadata for post-labeling join ─────────────────────────────
    key_df = sample_all[["sample_id","case","global_cluster_id","region","gpt_stance"]].copy()
    key_path = ANNOT_DIR / "annotation_key.csv"
    key_df.to_csv(key_path, index=False)
    print(f"\\nWrote {len(key_df)} rows -> annotation_key.csv")

    # ── Annotation files: no metadata that could bias judgment ────────────────
    sample_all["stance_target"]    = sample_all["theme"]
    sample_all["post_text"]        = sample_all["text"]
    sample_all["annotator_stance"] = ""
    sample_all["annotator_notes"]  = ""
    ANNOT_COLS = ["sample_id","stance_target","post_text","annotator_stance","annotator_notes"]

    annot_base = sample_all[ANNOT_COLS].copy()

    # Annotator A and B — same rows, different random orders
    idx_a = rng_annot.permutation(len(annot_base))
    idx_b = rng_annot.permutation(len(annot_base))
    # Verify they differ (astronomically unlikely to collide but assert for safety)
    assert not np.array_equal(idx_a, idx_b), "Permutations collided — change seed"

    annot_a = annot_base.iloc[idx_a].reset_index(drop=True)
    annot_b = annot_base.iloc[idx_b].reset_index(drop=True)

    path_a = ANNOT_DIR / "annotation_a.csv"
    path_b = ANNOT_DIR / "annotation_b.csv"
    annot_a.to_csv(path_a, index=False)
    annot_b.to_csv(path_b, index=False)
    print(f"Wrote annotation_a.csv ({len(annot_a)} rows)")
    print(f"Wrote annotation_b.csv ({len(annot_b)} rows)")
    print(f"\\nAll annotation files -> {ANNOT_DIR}/")
"""))


# ── Section 7c: Inter-annotator agreement scoring (stub) ─────────────────────
cells.append(md("""\
## 7c  Annotation Scoring

Run after both annotators have completed their files.
Rename the filled-in files to `annotation_a_done.csv` and `annotation_b_done.csv`.

Computes:
- Cohen's κ between annotators A and B
- Disagreement list (exported for manual adjudication)
- Macro-F1 and 3×3 confusion matrix of adjudicated labels vs `gpt_stance`

Results are written to `STATS`.
"""))

cells.append(code("""\
from pathlib import Path

KEY_PATH    = ANNOT_DIR / "annotation_key.csv"
DONE_A_PATH = ANNOT_DIR / "annotation_a_done.csv"
DONE_B_PATH = ANNOT_DIR / "annotation_b_done.csv"

if not DONE_A_PATH.exists() or not DONE_B_PATH.exists():
    missing = [p for p in (DONE_A_PATH, DONE_B_PATH) if not p.exists()]
    print("Scoring stub — annotation not yet complete.")
    for p in missing:
        print(f"  Missing: {p}")
    print("\\nOnce annotators have finished:")
    print("  1. Copy annotation_a.csv -> annotation_a_done.csv (filled in)")
    print("  2. Copy annotation_b.csv -> annotation_b_done.csv (filled in)")
    print("  3. Re-run this cell.")
else:
    from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix

    STANCE_ORDER = ["support", "oppose", "neutral"]
    VALID_LABELS = set(STANCE_ORDER)

    key    = pd.read_csv(KEY_PATH)
    done_a = pd.read_csv(DONE_A_PATH)
    done_b = pd.read_csv(DONE_B_PATH)

    for df_ in (done_a, done_b):
        df_["annotator_stance"] = df_["annotator_stance"].str.strip().str.lower()

    merged = (
        done_a[["sample_id","annotator_stance"]]
        .rename(columns={"annotator_stance": "label_a"})
        .merge(
            done_b[["sample_id","annotator_stance"]]
            .rename(columns={"annotator_stance": "label_b"}),
            on="sample_id",
        )
        .merge(key[["sample_id","case","global_cluster_id","region","gpt_stance"]],
               on="sample_id")
    )

    # Drop rows with invalid or missing labels
    scorable = merged[
        merged["label_a"].isin(VALID_LABELS) & merged["label_b"].isin(VALID_LABELS)
    ].copy()
    n_invalid = len(merged) - len(scorable)
    if n_invalid:
        print(f"WARNING: {n_invalid} rows dropped (invalid/blank label_a or label_b)")
    print(f"Scorable rows: {len(scorable)} / {len(key)}")

    # ── Cohen's kappa ─────────────────────────────────────────────────────────
    kappa = cohen_kappa_score(scorable["label_a"], scorable["label_b"],
                              labels=STANCE_ORDER)
    print(f"\\nInter-annotator Cohen's kappa: {kappa:.3f}")
    STATS["annotation_kappa"] = round(float(kappa), 3)

    # ── Adjudication ──────────────────────────────────────────────────────────
    # Agreement -> take that label; disagreement -> None (requires manual resolution)
    scorable["adj_label"]          = np.where(
        scorable["label_a"] == scorable["label_b"], scorable["label_a"], None
    )
    scorable["needs_adjudication"] = scorable["adj_label"].isna()

    n_disagree = scorable["needs_adjudication"].sum()
    pct_disagree = 100 * n_disagree / len(scorable) if len(scorable) else 0
    print(f"Agreements  : {len(scorable) - n_disagree} ({100-pct_disagree:.1f}%)")
    print(f"Disagreements requiring adjudication: {n_disagree} ({pct_disagree:.1f}%)")
    STATS["annotation_n_disagreements"] = int(n_disagree)

    if n_disagree > 0:
        disagree_path = ANNOT_DIR / "annotation_disagreements.csv"
        disagree_cols = ["sample_id","label_a","label_b","gpt_stance","case","region"]
        scorable[scorable.needs_adjudication][disagree_cols].to_csv(
            disagree_path, index=False
        )
        print(f"Wrote {disagree_path.name} for manual adjudication")
        print("\\nDisagreement breakdown by case:")
        print(scorable[scorable.needs_adjudication]
              .groupby(["case","label_a","label_b"]).size().to_string())

    # ── Macro-F1 and confusion matrix (adjudicated rows only) ─────────────────
    adj = scorable[~scorable.needs_adjudication].copy()
    print(f"\\nAdjudicated rows available for F1/confusion: {len(adj)}")

    if len(adj) >= 3:
        f1_macro = f1_score(
            adj["gpt_stance"], adj["adj_label"],
            labels=STANCE_ORDER, average="macro", zero_division=0
        )
        print(f"Macro-F1 (adjudicated human vs gpt_stance modal): {f1_macro:.3f}")
        STATS["annotation_f1_macro"] = round(float(f1_macro), 3)

        cm = confusion_matrix(adj["gpt_stance"], adj["adj_label"],
                              labels=STANCE_ORDER)
        cm_df = pd.DataFrame(
            cm,
            index  =[f"gpt_{s}" for s in STANCE_ORDER],
            columns=[f"human_{s}" for s in STANCE_ORDER],
        )
        print("\\nConfusion matrix (rows=gpt_stance, cols=adjudicated human label):")
        print(cm_df.to_string())
        STATS["annotation_confusion_matrix"] = cm_df.to_dict()

        # Per-case F1
        print("\\nMacro-F1 per case:")
        for case in sorted(adj["case"].unique()):
            sub = adj[adj.case == case]
            if len(sub) >= 3:
                f1_c = f1_score(sub["gpt_stance"], sub["adj_label"],
                                labels=STANCE_ORDER, average="macro", zero_division=0)
                print(f"  {case:<12}: {f1_c:.3f}  (n={len(sub)})")
                STATS[f"annotation_f1_macro_{case}"] = round(float(f1_c), 3)
    else:
        print("Insufficient adjudicated rows for macro-F1 / confusion matrix.")
"""))


# ── Section 8: Regenerate figures ─────────────────────────────────────────────
cells.append(md("""\
## 8  Regenerate Figures from Filtered Data

Three figures rebuilt from `df_on` (on-topic clusters only):
1. C vs p_neutral triangle (stance geometry)
2. drift_rate vs directedness_normalized (NOT net_displacement; no y=x diagonal)
3. Persistence OLS (log10(n_posts) vs persistence_frac) with updated R²
"""))

cells.append(code("""\
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.spines.top"]   = False
matplotlib.rcParams["axes.spines.right"] = False

CASES      = ["venezuela", "iran", "russia"]
CASE_LABEL = {"venezuela": "Venezuela", "iran": "Iran", "russia": "Russia–Ukraine"}
FILL_COLOR = "#4878CF"
S_SCALE    = 50
MIN_POSTS  = 10
C_THRESH   = 0.40
PN_THRESH  = 0.50

def _savefig_dual(fig, stem, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_dir}/{stem}.{{pdf,png}}")

def bubble_area(n_posts, s_scale=S_SCALE):
    return s_scale * np.log10(np.maximum(n_posts, 1) + 1)
"""))

cells.append(code("""\
# ── Figure A: C vs p_neutral triangle ─────────────────────────────────────────
fig_a, axes_a = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True, sharex=True)
fig_a.subplots_adjust(wspace=0.12)

for ax, case in zip(axes_a, CASES):
    sub = df_on[(df_on.case == case) & (df_on.n_posts >= MIN_POSTS)].copy()

    # Region shading
    tri_contested    = mpatches.Polygon(
        [(C_THRESH, 0), (1 - PN_THRESH, 0), (C_THRESH, 1 - PN_THRESH - C_THRESH)],
        closed=True, facecolor="#d62728", alpha=0.08, zorder=0
    )
    tri_fact         = mpatches.Polygon(
        [(0, PN_THRESH), (0, 1), (1 - PN_THRESH, PN_THRESH)],
        closed=True, facecolor="#2ca02c", alpha=0.08, zorder=0
    )
    ax.add_patch(tri_contested)
    ax.add_patch(tri_fact)

    ax.scatter(
        sub["p_neutral"], sub["C"],
        s=bubble_area(sub["n_posts"]),
        color=FILL_COLOR, alpha=0.55, linewidths=0.3, edgecolors="white", zorder=2
    )
    ax.axhline(C_THRESH,  color="grey", lw=0.6, ls="--", zorder=1)
    ax.axvline(PN_THRESH, color="grey", lw=0.6, ls="--", zorder=1)
    ax.set_title(CASE_LABEL[case], fontsize=8)
    ax.set_xlabel("p_neutral", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

axes_a[0].set_ylabel("Controversy score C", fontsize=8)
fig_a.suptitle("Stance geometry: controversy vs neutrality fraction", fontsize=8, y=1.01)
_savefig_dual(fig_a, "stance_geometry_filtered", OUT_DIR)
plt.show()
"""))

cells.append(code("""\
# ── Figure B: drift_rate vs directedness_normalized ───────────────────────────
# No y=x diagonal (directedness_normalized is NOT a ratio of drift_rate)
fig_b, axes_b = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True, sharex=True)
fig_b.subplots_adjust(wspace=0.12)

for ax, case in zip(axes_b, CASES):
    sub = df_on[
        (df_on.case == case) &
        df_on["drift_rate"].notna() &
        df_on["directedness_normalized"].notna()
    ].copy()

    ax.scatter(
        sub["drift_rate"], sub["directedness_normalized"],
        s=bubble_area(sub["n_posts"], s_scale=30),
        color=FILL_COLOR, alpha=0.55, linewidths=0.3, edgecolors="white", zorder=2
    )
    # Reference line at directedness_normalized = 1 (random-walk expectation)
    ax.axhline(1.0, color="grey", lw=0.6, ls="--", zorder=1, label="random walk")
    ax.set_title(CASE_LABEL[case], fontsize=8)
    ax.set_xlabel("Drift rate (rad/window)", fontsize=8)
    ax.tick_params(labelsize=7)

axes_b[0].set_ylabel("Directedness (normalised)", fontsize=8)
fig_b.suptitle("Centroid drift: rate vs directedness", fontsize=8, y=1.01)
_savefig_dual(fig_b, "drift_directedness_filtered", OUT_DIR)
plt.show()
"""))

cells.append(code("""\
# ── Figure C: Persistence OLS (filtered) ─────────────────────────────────────
try:
    from adjustText import adjust_text
    HAVE_AT = True
except ImportError:
    HAVE_AT = False

fig_c, axes_c = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False, sharex=False)
fig_c.subplots_adjust(wspace=0.30)

for i, (ax, case) in enumerate(zip(axes_c, CASES)):
    sub = df_on[
        (df_on.case == case) &
        df_on["n_posts"].notna() &
        df_on["persistence_frac"].notna()
    ].copy()

    x = np.log10(sub["n_posts"].clip(lower=1))
    y = sub["persistence_frac"]

    ax.scatter(x, y, s=20, color=FILL_COLOR, alpha=0.5, linewidths=0, zorder=2)

    if len(sub) >= 3:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, model.predict(x_line.reshape(-1, 1)),
                color="#d62728", lw=1.0, zorder=3)
        r2 = model.score(x.values.reshape(-1, 1), y.values)
        ax.text(0.05, 0.93, f"R²={r2:.2f}", transform=ax.transAxes, fontsize=7, va="top")
        STATS[f"{case}_ols_r2_filtered"] = round(r2, 3)

    # Annotate top-3 positive and negative residuals
    if "persistence_residual_std" in sub.columns:
        resid_col = sub["persistence_residual_std"].dropna()
        if len(resid_col) >= 6:
            top_idx    = resid_col.nlargest(3).index
            bottom_idx = resid_col.nsmallest(3).index
            flag_idx   = list(top_idx) + list(bottom_idx)
            texts = []
            for idx in flag_idx:
                row = sub.loc[idx]
                lbl = str(int(row["global_cluster_id"]))
                t = ax.text(
                    np.log10(row["n_posts"] + 1), row["persistence_frac"],
                    lbl, fontsize=5, ha="left", va="bottom", color="#444"
                )
                texts.append(t)
            if HAVE_AT and texts:
                adjust_text(texts, ax=ax, time_lim=1.0,
                            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.4))

    ax.set_xlabel("log₁₀(n_posts)", fontsize=8)
    ax.set_title(CASE_LABEL[case], fontsize=8)
    ax.tick_params(labelsize=7)

axes_c[0].set_ylabel("Persistence fraction", fontsize=8)
fig_c.suptitle("Narrative persistence vs. volume", fontsize=8, y=1.01)
_savefig_dual(fig_c, "persistence_filtered", OUT_DIR)
plt.show()
"""))


# ── Section 9: Export ──────────────────────────────────────────────────────────
cells.append(md("## 9  Export `analysis_stats.json` + Final CSV"))

cells.append(code("""\
# ── Headline table ────────────────────────────────────────────────────────────
print("\\n" + "="*60)
print("HEADLINE STATISTICS")
print("="*60)
for case in CASES:
    lbl = CASE_LABEL[case]
    print(f"\\n{lbl}")
    for key in ["n_clusters_raw","n_clusters_offtopic","n_clusters_ontopic",
                "ols_r2_pre","ols_r2_post","ols_r2_filtered"]:
        k = f"{case}_{key}"
        if k in STATS:
            print(f"  {key:<30}: {STATS[k]}")
    for reg in ["contested","consolidated","fact-relaying"]:
        k_share = f"{case}_{reg}_volume_share"
        k_lo    = f"{case}_{reg}_volume_ci_lo"
        k_hi    = f"{case}_{reg}_volume_ci_hi"
        if k_share in STATS:
            print(f"  {reg:<30}: {STATS[k_share]:.3f}  [{STATS.get(k_lo, '?'):.3f},"
                  f" {STATS.get(k_hi, '?'):.3f}]")

print()
for k in ["region_chisq_chi2","region_chisq_p","region_chisq_dof",
          "region_resid_chisq_chi2","region_resid_chisq_p",
          "n_fragile_clusters","n_near_duplicate_pairs","annotation_sample_n"]:
    if k in STATS:
        print(f"  {k:<40}: {STATS[k]}")

# ── Compare to previous run: flag stats that changed >10% ────────────────────
PREV_STATS_JSON = ROOT / "analysis_stats_prev.json"
if PREV_STATS_JSON.exists():
    with open(PREV_STATS_JSON) as fh:
        prev_stats = json.load(fh)
    changed_10pct = []
    for k, v in STATS.items():
        if k not in prev_stats:
            continue
        try:
            prev = float(prev_stats[k])
            curr = float(v)
        except (TypeError, ValueError):
            continue
        if prev == 0:
            continue
        rel_change = abs(curr - prev) / abs(prev)
        if rel_change > 0.10:
            changed_10pct.append((k, prev, curr, rel_change))
    if changed_10pct:
        print("\\n** STATS CHANGED >10% RELATIVE — RE-EXAMINE BEFORE WRITING **")
        print(f"  {'stat':<45} {'prev':>10} {'curr':>10} {'chg%':>8}")
        for k, prev, curr, rel in sorted(changed_10pct, key=lambda x: -x[3]):
            print(f"  {k:<45} {prev:>10.4f} {curr:>10.4f} {rel*100:>7.1f}%")
        STATS["n_stats_changed_10pct"] = len(changed_10pct)
    else:
        print("\\nNo stats changed >10% from previous run.")
        STATS["n_stats_changed_10pct"] = 0
else:
    print(f"\\nNo previous stats file at {PREV_STATS_JSON}.")
    print("After reviewing this run, run:")
    print("  import shutil; shutil.copy('analysis_stats.json', 'analysis_stats_prev.json')")
    print("to enable >10% change detection on future runs.")

# ── Write JSON ────────────────────────────────────────────────────────────────
with open(STATS_JSON, "w") as fh:
    json.dump(STATS, fh, indent=2, default=str)
print(f"\\nWrote {len(STATS)} stats -> {STATS_JSON}")

# ── Write final CSV ───────────────────────────────────────────────────────────
df_on.to_csv(FINAL_CSV, index=False)
print(f"Wrote {len(df_on)} on-topic cluster rows → {FINAL_CSV}")
"""))


# ── Assemble and write ────────────────────────────────────────────────────────
nb.cells = cells

out_path = Path("analysis_stats.ipynb")
nbf.write(nb, str(out_path))
print(f"Written -> {out_path}  ({len(cells)} cells)")
