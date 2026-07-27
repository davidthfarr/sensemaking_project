"""Data loading helpers for validation analyses."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("data")
PROC_ROOT = DATA_ROOT / "processed"
EVAL_ROOT = DATA_ROOT / "evaluated"
CASES = ["venezuela", "iran", "russia"]

_EXCLUDED_STEMS = frozenset({
    "global_clusters", "cluster_stance", "cluster_themes",
    "topic_stance", "topic_stance_by_window", "cluster_labels",
    "cluster_summaries", "results",
})

_POST_ID_ALIASES = ("Resource Id", "tweet_id", "tweetid", "post id", "postid", "id")


def _normalise_post_id(df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns:
        for alias in _POST_ID_ALIASES:
            if alias in df.columns:
                return df.rename(columns={alias: "post_id"})
    return df


def load_posts_repr(case: str) -> pd.DataFrame:
    """Load posts_repr.parquet (text + embeddings) for a case."""
    path = PROC_ROOT / case / "posts_repr.parquet"
    if not path.exists():
        raise FileNotFoundError(f"posts_repr.parquet not found for '{case}': {path}")
    df = pd.read_parquet(path)
    df = _normalise_post_id(df)
    df["post_id"] = df["post_id"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset=["post_id"], keep="first")
    if len(df) < before:
        print(f"  [{case}] Dropped {before - len(df):,} duplicate post_ids from posts_repr")
    return df


def build_emb_map(repr_df: pd.DataFrame) -> dict:
    """Return {post_id: np.ndarray} for all rows with a valid embedding."""
    return {
        pid: emb
        for pid, emb in zip(repr_df["post_id"], repr_df["embedding"])
        if emb is not None
    }


def load_global_clusters(case: str) -> pd.DataFrame:
    """Load global_clusters.parquet for a case."""
    path = EVAL_ROOT / case / "global_clusters.parquet"
    if not path.exists():
        raise FileNotFoundError(f"global_clusters.parquet not found for '{case}': {path}")
    df = pd.read_parquet(path)
    df = _normalise_post_id(df)
    df["post_id"] = df["post_id"].astype(str)
    df["is_noise"] = df["is_noise"].astype(bool)
    return df


def load_cluster_stance(case: str) -> pd.DataFrame:
    """Load cluster_stance.parquet (cluster-level) for a case."""
    path = EVAL_ROOT / case / "cluster_stance.parquet"
    if not path.exists():
        raise FileNotFoundError(f"cluster_stance.parquet not found for '{case}': {path}")
    df = pd.read_parquet(path)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    return df


def load_cluster_themes(case: str) -> pd.DataFrame:
    """Load cluster_themes.parquet for a case."""
    path = EVAL_ROOT / case / "cluster_themes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"cluster_themes.parquet not found for '{case}': {path}")
    df = pd.read_parquet(path)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    return df


def load_topic_stance(case: str) -> "pd.DataFrame | None":
    """Load topic_stance.parquet (post-level) if present; None otherwise."""
    path = EVAL_ROOT / case / "topic_stance.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = _normalise_post_id(df)
    df["post_id"] = df["post_id"].astype(str)
    return df


def load_window_files(case: str) -> list:
    """Return per-window parquet paths sorted chronologically, excluding pipeline outputs."""
    eval_dir = EVAL_ROOT / case
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluated directory not found: {eval_dir}")
    return sorted(
        f for f in eval_dir.glob("*.parquet")
        if f.stem not in _EXCLUDED_STEMS
    )


def load_event_timeline(case: str) -> pd.DataFrame:
    """Load event timeline CSV for a case. Returns empty DataFrame if not populated yet."""
    path = DATA_ROOT / "event_timelines" / f"{case}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "event_label", "source_url"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "event_label", "source_url"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.dropna(subset=["date"]).reset_index(drop=True)
