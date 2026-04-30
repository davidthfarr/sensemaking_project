import pandas as pd
from sensemaking.embeddings.encoder import EmbeddingEncoder
from sensemaking.embeddings.stance import ZeroShotStanceLabeler
from sensemaking.data.schemas import Post

# Load cleaned / filtered data
df1 = pd.read_parquet("data/processed/ck/posts_from_top_accounts_ck.parquet")
df2 = pd.read_parquet("data/processed/ck/top_level_replies_to_posts_from_top_accounts_ck.parquet")

# For posts from influential accounts, even if they have parents, for this purpose not applicable
df1['reply_parent_uri'] = None
df1['reply_parent_author'] = None
df1['reply_root_uri'] = None
df1['reply_root_author'] = None

df2['timestamp'] = df2['timestamp'].astype(str)

print(df1.head()['timestamp'])
print(df2.head()['timestamp'])

df = pd.concat([df1, df2], keys=['originals', 'replies'])

posts = [
    Post(
        post_id=row.post_id,
        user_id=row.user_id,
        timestamp=row.timestamp,
        text=row.text,
        reply_parent_id=row.reply_parent_uri,
        reply_parent_author=row.reply_parent_author,
        reply_root_id=row.reply_root_uri,
        reply_root_author=row.reply_root_author
        
    )
    for _, row in df.iterrows()
]

# Encode + stance ONCE
encoder = EmbeddingEncoder(require_cuda=False)
#stance = ZeroShotStanceLabeler(require_cuda=False)

posts = encoder(posts)
#posts = stance(posts)

# Persist representation
out = pd.DataFrame({
    "post_id": [p.post_id for p in posts],
    "user_id": [p.user_id for p in posts],
    "timestamp": [p.timestamp for p in posts],
    "text": [p.text for p in posts],
    "reply_parent_id": [p.reply_parent_id for p in posts],
    "reply_parent_author": [p.reply_parent_author for p in posts],
    "reply_root_id": [p.reply_root_id for p in posts],
    "reply_root_author": [p.reply_root_author for p in posts],
    "embedding": [p.embedding for p in posts],
    "stance": [0 for p in posts],
})

out.to_parquet("data/processed/posts_repr_ck_with_top_level_replies.parquet", index=False)
