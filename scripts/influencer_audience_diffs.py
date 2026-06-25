from pathlib import Path
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

from sensemaking.data.schemas import Post

from scripts_environment_wrapper import environment

# -------------------------
# Configuration
# -------------------------
PROCESSED_PATH = Path(environment.PROCESSED_FILE_PATH())
OUTPUT_PATH = Path('data/ck_distances.parquet')

WINDOW_DAYS = 4 #currently in hours
STEP_DAYS = 2

df = pd.read_parquet(PROCESSED_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
print(df.columns)
print(df.sample_type.value_counts())
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
    print(window_df.sample_type.value_counts())

    if len(window_df) == 0:
        window_start += timedelta(hours=STEP_DAYS)
        continue

    cluster_path = Path(f"{environment.EVALUATED_DIR()}/{window_start.strftime('%Y-%m-%d-%H')}.parquet")

    cluster_df = pd.read_parquet(cluster_path)
    print(cluster_df.columns)
    posts_in_clusters = cluster_df[~cluster_df['is_noise']]
    cluster_df_supplemented = pd.merge(posts_in_clusters, window_df, how='inner', on='post_id')
    print(cluster_df_supplemented.shape)
    if cluster_df_supplemented.shape[0] > 0:
        influencers_posts_subset = cluster_df_supplemented[cluster_df_supplemented['sample_type'] == 'influencers']
        print(cluster_df_supplemented.columns)
        print(cluster_df_supplemented.sample_type.unique())
        influencers_subset = set(influencers_posts_subset['user_id'].unique())
        print(f'influencers: {len(influencers_subset)}')
        for author in influencers_subset:
            author_posts = influencers_posts_subset[influencers_posts_subset['user_id'] == author]
            #print(f'{len(author_posts)} posts by {author} in {window_start}')
            audience_posts = cluster_df_supplemented[cluster_df_supplemented['reply_parent_author'] == author]
            #print(f'{len(audience_posts)} comments on posts by {author} in {window_start}')
            #print(f'{audience_posts.cluster_id.nunique()} clusters of comments on posts by {author} in {window_start}')
            
            # Average the author_posts embeddings
            if len(author_posts) > 0:
                author_average_embedding = author_posts.embedding.mean()
            else:
                author_average_embedding = None

            # get all audience posts

            if len(audience_posts) > 0:
                audience_average_embedding = audience_posts.embedding.mean()
            else:
                audience_average_embedding = None

            # Get per-cluster audience numbers, IF there are multiple clusters
            if audience_posts.cluster_id.nunique() > 1:
                cluster_ids = list(audience_posts.cluster_id.unique())
                cluster_averages = list()
                for i in range(0, len(cluster_ids)):
                    posts_in_cluster = audience_posts[audience_posts['cluster_id'] == cluster_ids[i]]
                    cluster_averages.append(posts_in_cluster.embedding.mean())
            else:
                cluster_ids = None
                cluster_averages = None

    else:
        print("No clusterable posts in interval, skipping")

    window_start = window_start + timedelta(hours=STEP_DAYS)

