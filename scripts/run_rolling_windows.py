from pathlib import Path
import pandas as pd
from datetime import timedelta

from sensemaking.clustering.hdbscan import HDBSCANClusterer
from sensemaking.data.schemas import Post


# -------------------------
# Configuration
# -------------------------
PROCESSED_PATH = Path("data/processed/venezuela/posts_repr.parquet")
OUTPUT_DIR = Path("data/evaluated/ven/hourly")

WINDOW_DAYS = 12  # currently in hours
STEP_DAYS = 4

MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = 2
STANCE_WEIGHT = 0.05
CLUSTER_SELECTION_EPSILON = 0.0  # raise to merge fragmented sub-clusters (0.1–0.5)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Load representation layer
# -------------------------
df = pd.read_parquet(PROCESSED_PATH)

# Ensure sorted by time
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

min_time = df["timestamp"].min().floor("H")
max_time = df["timestamp"].max().floor("H")


# -------------------------
# Initialize clusterer
# -------------------------
clusterer = HDBSCANClusterer(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    stance_weight=STANCE_WEIGHT,
    cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
)


# -------------------------
# Rolling window loop
# -------------------------
window_start = min_time

while window_start <= max_time:
    window_end = window_start + timedelta(hours=WINDOW_DAYS)

    window_df = df[
        (df["timestamp"] >= window_start) &
        (df["timestamp"] < window_end)
    ]

    if len(window_df) == 0:
        window_start += timedelta(hours=STEP_DAYS)
        continue

    # Build Post objects
    posts = [
        Post(
            post_id=row.post_id,
            timestamp=row.timestamp,
            text=row.text,
            embedding=row.embedding,
            stance=row.stance,
        )
        for _, row in window_df.iterrows()
    ]

    # Cluster
    posts = clusterer.fit_predict(posts)

    labels = [p.cluster_id for p in posts if not p.is_noise]

    num_clusters = len(set(labels))
    noise_frac = sum(p.is_noise for p in posts) / len(posts)

    print(
        f"Window {window_start.date()} | "
        f"posts={len(posts):4d} | "
        f"clusters={num_clusters:2d} | "
        f"noise={noise_frac:.2f}"
    )

    # Write evaluated output
    out_df = pd.DataFrame({
        "post_id": [p.post_id for p in posts],
        "window": window_start.strftime("%Y-%m-%d-%H"),
        "cluster_id": [
            p.cluster_id if not p.is_noise else None
            for p in posts
        ],
        "is_noise": [p.is_noise for p in posts],
    })

    out_path = OUTPUT_DIR / f"{window_start.strftime('%Y-%m-%d-%H')}.parquet"
    out_df.to_parquet(out_path, index=False)

    print(
        f"Wrote {len(out_df):5d} posts "
        f"for window starting {window_start.date()}"
    )

    window_start += timedelta(hours=STEP_DAYS)
