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
