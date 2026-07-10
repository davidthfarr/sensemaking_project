from pathlib import Path
import pandas as pd
from datetime import timedelta, datetime

from sensemaking.clustering.hdbscan import HDBSCANClusterer
from sensemaking.data.schemas import Post

from scripts_environment_wrapper import environment
from tqdm import tqdm


# -------------------------
# Configuration
# -------------------------
PROCESSED_PATH = Path(environment.PROCESSED_FILE_PATH())
OUTPUT_DIR = Path(environment.EVALUATED_AUDIENCE_CLUSTER_DIR())

ONLY_INFLUENCERS = False
ONLY_REPLIES = False

MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 2
STANCE_WEIGHT = 0
CLUSTER_SELECTION_EPSILON = 0.5  # raise to merge fragmented sub-clusters (0.1–0.5)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Load representation layer
# -------------------------
df = pd.read_parquet(PROCESSED_PATH)

# Ensure sorted by time
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
#print(df.shape)
#print(df.columns)
if ONLY_INFLUENCERS:
    df = df[(df['reply_parent_id'].isna()) & (df['reply_root_id'].isna())]
if ONLY_REPLIES:
    df = df[(df['reply_parent_id'].notna()) | (df['reply_root_id'].notna())]

#print(df.shape)

min_time = df["timestamp"].min().floor("h")
max_time = df["timestamp"].max().floor("h")


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
# influencer loop
# -------------------------

influencers = set(df[(df['reply_parent_id'].isna()) & (df['reply_root_id'].isna())]['user_id'])

for influencer in tqdm(influencers):

    window_df = df[
        (df["user_id"] == influencer) |
        (df["reply_parent_author"] == influencer)
    ]

    if len(window_df) == 0:
        continue

    # Build Post objects
    posts = [
        Post(
            post_id=row.post_id,
            user_id=row.user_id,
            timestamp=row.timestamp,
            text=row.text,
            embedding=row.embedding,
            stance=row.stance,
            sample_type=row.sample_type
        )
        for _, row in window_df.iterrows()
    ]

    # Cluster
    try:
        posts = clusterer.fit_predict(posts)

        labels = [p.cluster_id for p in posts if not p.is_noise]

        num_clusters = len(set(labels))
        noise_frac = sum(p.is_noise for p in posts) / len(posts)

    #print(
    #    f"Influencer {influencer} | "
    #    f"posts={len(posts):4d} | "
    #    f"clusters={num_clusters:2d} | "
    #    f"noise={noise_frac:.2f} | "
    #    f"cur_time={datetime.now()}"
    #)

    # Write evaluated output
        out_df = pd.DataFrame({
            "post_id": [p.post_id for p in posts],
            "influencer": influencer,
            "cluster_id": [
                p.cluster_id if not p.is_noise else None
                for p in posts
            ],
            "is_noise": [p.is_noise for p in posts],
        })

        out_path = OUTPUT_DIR / f"{influencer}.parquet"
        out_df.to_parquet(out_path, index=False)
    except:
        print(f"Error in clustering for influencer {influencer}, with {len(posts)} posts, skipping")


    #print(
    #    f"Wrote {len(out_df):5d} posts "
    #    f"for influencer {influencer}"
    #)