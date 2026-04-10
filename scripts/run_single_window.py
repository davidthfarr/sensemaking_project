import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

from sensemaking.data.schemas import Post
from sensemaking.embeddings.encoder import EmbeddingEncoder
from sensemaking.embeddings.stance import ZeroShotStanceLabeler
from sensemaking.clustering.hdbscan import HDBSCANClusterer

# =====================
# Paths
# =====================

PROCESSED_PATH = Path("data/processed/ukraine_en_clean.parquet")
OUT_DIR = Path("data/evaluated/weekly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# Parameters
# =====================

WINDOW_DAYS = 7
STANCE_WEIGHT = 0.1
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 5
CLUSTER_SELECTION_EPSILON = 0.0  # raise to merge fragmented sub-clusters (0.1–0.5)

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
STANCE_MODEL = "facebook/bart-large-mnli"

DEVICE = "cuda"  # change to "cpu" if needed

# =====================
# Load processed data
# =====================

print(f"Loading processed data from {PROCESSED_PATH}")
df = pd.read_parquet(PROCESSED_PATH)

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")
print(df.head())

# =====================
# Convert to Post objects
# =====================

posts = [
    Post(
        post_id=row.post_id,
        text=row.text,
        timestamp=row.timestamp,
        user_id=row.user_id,
    )
    for row in df.itertuples(index=False)
]

# =====================
# Embeddings
# =====================

print("Computing embeddings...")
encoder = EmbeddingEncoder(
    model_name=EMBED_MODEL,
    device=DEVICE,
    batch_size=64,
)

posts = encoder(posts)

# =====================
# Stance
# =====================

print("Computing stance labels...")
stance_labeler = ZeroShotStanceLabeler(
    model_name=STANCE_MODEL,
    device=DEVICE,
    batch_size=32,
)

posts = stance_labeler(posts)


# =====================
# Select a single time window
# =====================

start_time = posts[0].timestamp
end_time = start_time + timedelta(days=WINDOW_DAYS)

window_posts = [
    p for p in posts
    if start_time <= p.timestamp < end_time
]

print(f"Posts in window [{start_time.date()} – {end_time.date()}]: {len(window_posts):,}")

# =====================
# Clustering
# =====================

print("Running HDBSCAN...")
clusterer = HDBSCANClusterer(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    stance_weight=STANCE_WEIGHT,
    cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
)

window_posts = clusterer.fit_predict(window_posts)

# =====================
# Convert to evaluated DataFrame
# =====================

df_eval = pd.DataFrame(
    {
        "post_id": [p.post_id for p in window_posts],
        "user_id": [p.user_id for p in window_posts],
        "timestamp": [p.timestamp for p in window_posts],
        "text": [p.text for p in window_posts],
        "stance": [p.stance for p in window_posts],
        "cluster_id": [p.cluster_id for p in window_posts],
        "is_noise": [p.is_noise for p in window_posts],
    }
)

df_eval["window_start"] = start_time
df_eval["window_end"] = end_time

# =====================
# Write evaluated data
# =====================

out_path = OUT_DIR / f"{start_time.date()}_{end_time.date()}.parquet"
df_eval.to_parquet(out_path, index=False)

print(f"Evaluated window written to {out_path}")
print(df_eval.head())
