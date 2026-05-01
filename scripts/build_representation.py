import pandas as pd
from sensemaking.embeddings.encoder import EmbeddingEncoder
from sensemaking.embeddings.stance import ZeroShotStanceLabeler
from sensemaking.data.schemas import Post
from scripts_environment_wrapper import environment

# Load cleaned / filtered data
df = pd.read_parquet(environment.CLEANED_FILE_PATH())

posts = [
    Post(
        post_id=row.post_id,
        user_id=row.user_id,
        timestamp=row.timestamp,
        text=row.text,
        reply_parent_id=row.reply_parent_uri,
        reply_parent_author=row.reply_parent_author,
        reply_root_id=row.reply_root_uri,
        reply_root_author=row.reply_root_author,
        sample_type = row.sample_type
        
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
    "sample_type": [p.sample_type for p in posts]
})

out.to_parquet(environment.PROCESSED_FILE_PATH(), index=False)
