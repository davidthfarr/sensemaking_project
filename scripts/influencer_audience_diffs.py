from pathlib import Path
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

from sensemaking.clustering.hdbscan import HDBSCANClusterer
from sensemaking.data.schemas import Post

from scripts_environment_wrapper import environment

# -------------------------
# Configuration
# -------------------------
PROCESSED_PATH = Path(environment.PROCESSED_FILE_PATH())
CLUSTER_DIR = Path(environment.EVALUATED_DIR())

WINDOW_DAYS = 4 #currently in hours
STEP_DAYS = 2

df = pd.read_parquet(PROCESSED_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
print(df.shape)

min_time = df["timestamp"].min().floor("h")
max_time = df["timestamp"].max().floor("h")

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

    cluster_path = CLUSTER_DIR / f"{window_start.strftime('%Y-%m-%d-%H')}.parquet"

    cluster_df = pd.read_parquet(cluster_path)
    posts_in_clusters = cluster_df[~cluster_df['is_noise']]
    cluster_df_supplemented = pd.merge(posts_in_clusters, window_df, how='inner', on='post_id')
    influencers_posts_subset = cluster_df_supplemented[cluster_df_supplemented['sample_type'] == 'influencers']

    influencers_subset = influencers_subset['user_id'].unique()
    print(f'influencers: {len(influencers_subset)}')
    for author in influencers_subset:
        author_posts = influencers_posts_subset[influencers_posts_subset['user_id'] == author]
        print(f'{len(author_posts)} posts by {author} in {window_start}')
        audience_posts = cluster_df_supplemented[cluster_df_supplemented['reply_parent_author'] == author]
        print(f'{len(audience_posts)} comments on posts by {author} in {window_start}')
        print(cluster_df_supplemented.columns)

