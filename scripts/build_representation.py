"""
Compute and cache semantic embeddings for a case (run once per dataset).

Usage
-----
python scripts/build_representation.py --case iran
python scripts/build_representation.py --case russia
python scripts/build_representation.py --case venezuela

Input:  data/processed/<case>/<case>_en_clean.parquet
Output: data/processed/<case>/posts_repr.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from sensemaking.data.schemas import Post
from sensemaking.embeddings.encoder import EmbeddingEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=["iran", "russia", "venezuela"],
                   help="Case name — resolves input/output paths automatically")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path  = Path("data/processed") / args.case / f"{args.case}_en_clean.parquet"
    out_path = Path("data/processed") / args.case / "posts_repr.parquet"

    print(f"Loading clean data from {in_path}")
    df = pd.read_parquet(in_path)

    posts = [
        Post(
            post_id=row.post_id,
            user_id=row.user_id,
            timestamp=row.timestamp,
            text=row.text,
        )
        for _, row in df.iterrows()
    ]
    print(f"Posts loaded: {len(posts):,}")

    encoder = EmbeddingEncoder(require_cuda=(args.device == "cuda"), device=args.device)
    posts = encoder(posts)

    out = pd.DataFrame({
        "post_id":   [p.post_id for p in posts],
        "user_id":   [p.user_id for p in posts],
        "timestamp": [p.timestamp for p in posts],
        "text":      [p.text for p in posts],
        "embedding": [p.embedding for p in posts],
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Written → {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
