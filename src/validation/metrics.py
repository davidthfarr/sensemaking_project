"""Controversy score, drift statistics, silhouette, and significance tests."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.validation.centroids import angular_distance, cosine_distance

# Minimum non-neutral post count required to trust C_tilde
CONTROVERSY_MIN_NONNEUTRAL = 20


# ── Controversy score ──────────────────────────────────────────────────────────

def controversy_score_correct(s: float, o: float, neu: float) -> float:
    """C = 1 - neu - |s - o|  (= 2 * min(s, o)).  The manuscript formula."""
    return float(1.0 - neu - abs(s - o))


def controversy_score_stored(s: float, o: float, neu: float) -> float:
    """(1 - |s-o|)*(1-neu) — the formula that was stored in cluster_stance.parquet."""
    return float((1.0 - abs(s - o)) * (1.0 - neu))


def controversy_tilde(
    c: float,
    neu: float,
    n_posts: int,
    min_nonneutral: int = CONTROVERSY_MIN_NONNEUTRAL,
) -> "float | None":
    """
    Normalised contestation ratio: C_tilde = C / (1 - neu).

    Returns None when the cluster has fewer than min_nonneutral non-neutral posts.
    """
    n_nonneutral = round(n_posts * (1.0 - neu))
    if n_nonneutral < min_nonneutral or (1.0 - neu) <= 0:
        return None
    return float(c / (1.0 - neu))


def recompute_controversy(
    stance_df: pd.DataFrame,
    min_nonneutral: int = CONTROVERSY_MIN_NONNEUTRAL,
) -> pd.DataFrame:
    """
    Add C_correct, C_2min, C_tilde, and audit columns to a cluster_stance DataFrame.

    Verifies the closed-form identity C = 2*min(s,o) and flags stored vs correct discrepancies.
    """
    df = stance_df.copy()
    s   = df["support_pct"]
    o   = df["oppose_pct"]
    neu = df["neutral_pct"]

    df["C_correct"] = (1.0 - neu - (s - o).abs()).clip(lower=0.0)
    df["C_2min"]    = (2.0 * np.minimum(s, o)).clip(lower=0.0)

    df["C_closed_form_match"] = np.isclose(df["C_correct"], df["C_2min"], atol=1e-6)

    if "controversy_score" in df.columns:
        df["C_stored"] = df["controversy_score"]
        df["C_stored_matches_correct"] = np.isclose(df["C_correct"], df["C_stored"], atol=1e-6)

    df["C_tilde"] = [
        controversy_tilde(c, n, npts, min_nonneutral)
        for c, n, npts in zip(df["C_correct"], neu, df.get("n_posts", [0] * len(df)))
    ]
    df["C_tilde_reliable"] = df["C_tilde"].notna()

    return df


# ── Drift ─────────────────────────────────────────────────────────────────────

def compute_drift_stats(centroid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cluster drift statistics from a (global_cluster_id, window, centroid) DataFrame.

    Returns one row per global_cluster_id with:
      n_windows, cumulative_path_angular, net_displacement_angular,
      mean_step_angular, net_path_ratio, cumulative_path_cosine
    """
    rows = []
    for gcid, grp in centroid_df.groupby("global_cluster_id"):
        centroids = list(grp.sort_values("window")["centroid"])
        n = len(centroids)
        if n < 2:
            rows.append({
                "global_cluster_id": gcid, "n_windows": n,
                "cumulative_path_angular": 0.0, "net_displacement_angular": 0.0,
                "mean_step_angular": np.nan, "net_path_ratio": np.nan,
                "cumulative_path_cosine": 0.0,
            })
            continue

        step_ang = [angular_distance(centroids[i], centroids[i + 1]) for i in range(n - 1)]
        step_cos = [cosine_distance(centroids[i], centroids[i + 1]) for i in range(n - 1)]
        cumul    = float(np.sum(step_ang))
        net      = angular_distance(centroids[0], centroids[-1])

        rows.append({
            "global_cluster_id":       gcid,
            "n_windows":               n,
            "cumulative_path_angular": cumul,
            "net_displacement_angular": net,
            "mean_step_angular":       cumul / (n - 1),
            "net_path_ratio":          net / cumul if cumul > 0 else np.nan,
            "cumulative_path_cosine":  float(np.sum(step_cos)),
        })

    return pd.DataFrame(rows)


# ── Volume–neutrality test ────────────────────────────────────────────────────

def volume_neutrality_test(stance_df: pd.DataFrame) -> dict:
    """
    Test whether high-volume clusters are more neutral (Mann-Whitney U, one-sided).

    Effect size: rank-biserial correlation r = 2U / (n1*n2) - 1.
    """
    df = stance_df.dropna(subset=["neutral_pct", "n_posts"]).copy()
    threshold = df["n_posts"].quantile(0.75)
    hi = df[df["n_posts"] >= threshold]["neutral_pct"].values
    lo = df[df["n_posts"] <  threshold]["neutral_pct"].values

    if len(hi) < 2 or len(lo) < 2:
        return {"error": "Insufficient data"}

    stat, p = mannwhitneyu(hi, lo, alternative="greater")
    r = float(2 * stat / (len(hi) * len(lo)) - 1.0)
    size_label = "large" if abs(r) > 0.5 else "medium" if abs(r) > 0.3 else "small"

    return {
        "n_high_volume": len(hi), "n_low_volume": len(lo),
        "volume_threshold_n_posts": float(threshold),
        "mean_neutral_high_vol": float(hi.mean()),
        "mean_neutral_low_vol":  float(lo.mean()),
        "mannwhitney_U": float(stat), "p_value": float(p),
        "rank_biserial_r": r,
        "effect_size_label": size_label,
    }


# ── Silhouette ────────────────────────────────────────────────────────────────

def compute_silhouette_per_window(
    window_df: pd.DataFrame,
    emb_map: dict,
    min_clusters: int = 2,
) -> "float | None":
    """
    Mean silhouette coefficient for non-noise posts in one window (cosine distance).

    Returns None if fewer than min_clusters clusters exist.
    """
    from sklearn.metrics import silhouette_score

    df = window_df[~window_df["is_noise"] & window_df["cluster_id"].notna()].copy()
    if df["cluster_id"].nunique() < min_clusters:
        return None

    pids = df["post_id"].tolist()
    valid_idx = [
        i for i, pid in enumerate(pids)
        if pid in emb_map
        and emb_map[pid] is not None
        and hasattr(emb_map[pid], "shape")
        and emb_map[pid].shape == (768,)
    ]
    if len(valid_idx) < 2:
        return None

    X      = np.vstack([emb_map[pids[i]] for i in valid_idx]).astype(np.float32)
    labels = df["cluster_id"].values[valid_idx]

    if len(np.unique(labels)) < 2:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(silhouette_score(X, labels, metric="cosine"))
