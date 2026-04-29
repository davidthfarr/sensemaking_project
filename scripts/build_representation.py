import pandas as pd
from sensemaking.embeddings.encoder import EmbeddingEncoder
from sensemaking.data.schemas import Post

# Load cleaned / filtered data
df = pd.read_parquet("data/processed/venezuela/ven_en_clean.parquet")

posts = [
    Post(
        post_id=row.post_id,
        user_id=row.user_id,
        timestamp=row.timestamp,
        text=row.text,
    )
    for _, row in df.iterrows()
]

# Encode embeddings ONCE
encoder = EmbeddingEncoder(require_cuda=True)

posts = encoder(posts)

# Persist representation
out = pd.DataFrame({
    "post_id": [p.post_id for p in posts],
    "user_id": [p.user_id for p in posts],
    "timestamp": [p.timestamp for p in posts],
    "text": [p.text for p in posts],
    "embedding": [p.embedding for p in posts],
})

out.to_parquet("data/processed/posts_repr.parquet", index=False)
