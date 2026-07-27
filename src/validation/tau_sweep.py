"""Re-run cluster alignment for a grid of τ values and collect sensitivity metrics."""
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_POST_ID_ALIASES = ("Resource Id", "tweet_id", "tweetid", "post id", "postid", "id")


def _normalise_post_id(df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns:
        for alias in _POST_ID_ALIASES:
            if alias in df.columns:
                return df.rename(columns={alias: "post_id"})
    return df


def _run_alignment_pass(window_files: list, emb_map: dict, tau: float) -> pd.DataFrame:
    """
    Step through window parquets and assign global cluster IDs for a given τ.

    Reimplements the run_alignment.py logic but parameterised by τ, returning
    a global_clusters-style DataFrame without writing to disk.
    """
    from sensemaking.clustering.alignment import align_clusters
    from sensemaking.data.schemas import Post

    next_global_id = 0
    prev_posts: list = []
    prev_local_to_global: dict = {}
    all_rows: list = []

    for wf in sorted(window_files):
        wdf = _normalise_post_id(pd.read_parquet(wf))
        posts = []
        for _, row in wdf.iterrows():
            pid   = str(row["post_id"])
            emb   = emb_map.get(pid)
            p     = Post(post_id=pid, text="", embedding=emb)
            is_noise = bool(row["is_noise"])
            cid   = None if is_noise or pd.isna(row.get("cluster_id")) else int(row["cluster_id"])
            p.cluster_id = cid
            p.is_noise   = is_noise
            posts.append(p)

        curr_to_prev: dict = {}
        if prev_posts:
            alignment    = align_clusters(prev_posts, posts, tau)
            curr_to_prev = {curr: prev for prev, curr in alignment.items()}

        local_to_global: dict = {}
        for p in posts:
            if p.cluster_id is None or p.cluster_id in local_to_global:
                continue
            prev_local = curr_to_prev.get(p.cluster_id)
            if prev_local is not None and prev_local in prev_local_to_global:
                local_to_global[p.cluster_id] = prev_local_to_global[prev_local]
            else:
                local_to_global[p.cluster_id] = next_global_id
                next_global_id += 1

        for p in posts:
            gid = local_to_global.get(p.cluster_id) if p.cluster_id is not None else None
            all_rows.append({"post_id": p.post_id, "window": wf.stem,
                             "global_cluster_id": gid, "is_noise": p.is_noise})

        prev_posts = posts
        prev_local_to_global = local_to_global

    return pd.DataFrame(all_rows)


def _compute_sweep_metrics(gc_df: pd.DataFrame, tau: float, case: str) -> dict:
    """Summary metrics from a single-τ alignment result."""
    df = gc_df[gc_df["global_cluster_id"].notna()].copy()
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    if df.empty:
        return {"tau": tau, "case": case, "n_global_clusters": 0,
                "mean_persistence": np.nan, "median_persistence": np.nan,
                "frac_one_window": np.nan, "frac_posts_3plus_windows": np.nan,
                "n_reactivations": 0, "largest_cluster_size": 0}

    persistence  = df.groupby("global_cluster_id")["window"].nunique()
    long_gcids   = set(persistence[persistence > 3].index)

    # Reactivation: a cluster reappears after a gap (windows are not consecutive)
    all_windows  = sorted(gc_df["window"].unique())
    win_idx      = {w: i for i, w in enumerate(all_windows)}
    n_react      = 0
    for gcid, grp in df.groupby("global_cluster_id"):
        idxs = sorted(win_idx[w] for w in grp["window"].unique())
        if any(idxs[i + 1] - idxs[i] > 1 for i in range(len(idxs) - 1)):
            n_react += 1

    return {
        "tau":                     tau,
        "case":                    case,
        "n_global_clusters":       int(len(persistence)),
        "mean_persistence":        float(persistence.mean()),
        "median_persistence":      float(persistence.median()),
        "frac_one_window":         float((persistence == 1).mean()),
        "frac_posts_3plus_windows": float(df["global_cluster_id"].isin(long_gcids).mean()),
        "n_reactivations":         n_react,
        "largest_cluster_size":    int(df.groupby("global_cluster_id")["post_id"].count().max()),
    }


def sweep_case(
    case: str,
    window_files: list,
    emb_map: dict,
    taus: list,
    cache_dir: Path = Path("outputs/.cache"),
) -> pd.DataFrame:
    """
    Run alignment at each τ value for one case.

    Results are cached per (case, τ) to avoid re-running on repeated notebook executions.
    Returns a DataFrame indexed by tau with metric columns.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for tau in taus:
        cache_path = cache_dir / f"tau_sweep_{case}_{tau:.2f}.parquet"
        if cache_path.exists():
            print(f"  [cache] τ={tau:.2f} for '{case}'")
            gc_df = pd.read_parquet(cache_path)
        else:
            print(f"  [computing] τ={tau:.2f} for '{case}' ({len(window_files)} windows)...")
            gc_df = _run_alignment_pass(window_files, emb_map, tau)
            gc_df.to_parquet(cache_path, index=False)

        m = _compute_sweep_metrics(gc_df, tau, case)
        rows.append(m)
        print(f"    → {m['n_global_clusters']} clusters | "
              f"mean persistence={m['mean_persistence']:.2f} | "
              f"1-window frac={m['frac_one_window']:.2f}")

    return pd.DataFrame(rows).set_index("tau")
