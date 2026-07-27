"""Per-window centroid computation and disk caching."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def compute_window_centroids(gc_df: pd.DataFrame, emb_map: dict) -> pd.DataFrame:
    """
    Compute the L2-normalized mean embedding for each (global_cluster_id, window) pair.

    Parameters
    ----------
    gc_df : DataFrame
        global_clusters.parquet — must have post_id, window, global_cluster_id, is_noise.
    emb_map : dict
        {post_id: np.ndarray} from io.build_emb_map().

    Returns
    -------
    DataFrame with columns: global_cluster_id (int), window (str), centroid (np.ndarray shape (768,))
    """
    df = gc_df[gc_df["global_cluster_id"].notna() & ~gc_df["is_noise"]].copy()
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)

    rows = []
    for (gcid, window), group in df.groupby(["global_cluster_id", "window"]):
        valid = [
            emb_map[pid]
            for pid in group["post_id"]
            if pid in emb_map
            and emb_map[pid] is not None
            and hasattr(emb_map[pid], "shape")
            and emb_map[pid].shape == (768,)
        ]
        if not valid:
            continue
        centroid = _l2_normalize(np.vstack(valid).astype(np.float32).mean(axis=0))
        rows.append({"global_cluster_id": gcid, "window": window, "centroid": centroid})

    return pd.DataFrame(rows)


def load_or_compute_centroids(
    case: str,
    gc_df: pd.DataFrame,
    emb_map: dict,
    cache_dir: Path = Path("outputs/.cache"),
) -> pd.DataFrame:
    """
    Return cached centroids if valid, otherwise compute and cache to disk.

    Cache is keyed on (n_non_noise_rows, n_unique_clusters) to detect stale caches.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_nn = gc_df[gc_df["global_cluster_id"].notna() & ~gc_df["is_noise"]]
    key_str = f"{len(df_nn)}_{df_nn['global_cluster_id'].nunique()}"
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]

    cache_path = cache_dir / f"centroids_{case}.parquet"
    key_file = cache_dir / f"centroids_{case}_{key_hash}.ok"

    if cache_path.exists() and key_file.exists():
        print(f"  [cache] centroids for '{case}' (key {key_hash})")
        return pd.read_parquet(cache_path)

    print(f"  [computing] window centroids for '{case}'...")
    cdf = compute_window_centroids(gc_df, emb_map)

    # Invalidate any old key files for this case
    for old_key in cache_dir.glob(f"centroids_{case}_*.ok"):
        old_key.unlink(missing_ok=True)

    cdf.to_parquet(cache_path, index=False)
    key_file.write_text(key_str)
    print(f"  [cached] {len(cdf):,} (cluster, window) centroids → {cache_path}")
    return cdf


def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Angular distance (radians) between two L2-normalized vectors."""
    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity between two L2-normalized vectors."""
    return float(1.0 - np.clip(float(np.dot(a, b)), -1.0, 1.0))
