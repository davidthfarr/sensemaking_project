import pandas as pd
from sensemaking.embeddings.encoder import EmbeddingEncoder
from sensemaking.embeddings.stance import ZeroShotStanceLabeler
from sensemaking.data.schemas import Post

# Load cleaned / filtered data
df = pd.read_parquet("data/processed/ck/posts_from_top_accounts_ck.parquet")

posts = [
    Post(
        post_id=row.post_id,
        user_id=row.user_id,
        timestamp=row.timestamp,
        text=row.text,
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
    "embedding": [p.embedding for p in posts],
    "stance": [0 for p in posts],
})

out.to_parquet("data/processed/posts_repr_ck.parquet", index=False)
