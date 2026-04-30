"""
Post-hoc GPT stance classification for all non-noise clusters in a case.

For each cluster that has a theme label, classifies every post as
support / oppose / neutral relative to that theme.

Usage
-----
python scripts/run_stance_classification.py --case iran
python scripts/run_stance_classification.py --case russia
python scripts/run_stance_classification.py --case venezuela

Input:
  data/evaluated/<case>/global_clusters.parquet
  data/evaluated/<case>/cluster_themes.parquet
  data/processed/<case>/posts_repr.parquet

Output:
  data/evaluated/<case>/cluster_stance.parquet
    post_id, global_cluster_id, theme, stance

Log:
  logs/<case>_stance.log

Requires OPENAI_API_KEY in .env or the environment.
Resume-safe: clusters already present in cluster_stance.parquet are skipped.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sensemaking.data.schemas import Post
from sensemaking.stance.posthoc_gpt import PosthocGPTStanceClassifier

MIN_POSTS_PER_CLUSTER = 3   # skip clusters smaller than this
SLEEP_BETWEEN_CLUSTERS = 1.0  # seconds between API bursts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=["iran", "russia", "venezuela"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--batch-size", type=int, default=20,
                   help="Posts per GPT call")
    p.add_argument("--min-posts", type=int, default=MIN_POSTS_PER_CLUSTER,
                   help="Skip clusters with fewer posts than this")
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CLUSTERS,
                   help="Seconds to sleep between clusters")
    return p.parse_args()


def setup_logging(case: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case}_stance.log"

    logger = logging.getLogger("stance")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    load_dotenv()
    args = parse_args()

    log = setup_logging(args.case, Path("logs"))
    log.info("=== Stance classification: case=%s ===", args.case)
    log.info("model=%s  batch_size=%d  min_posts=%d", args.model, args.batch_size, args.min_posts)

    eval_dir  = Path("data/evaluated") / args.case
    repr_path = Path("data/processed") / args.case / "posts_repr.parquet"
    gc_path   = eval_dir / "global_clusters.parquet"
    themes_path = eval_dir / "cluster_themes.parquet"
    out_path  = eval_dir / "cluster_stance.parquet"

    # ------------------------------------------------------------------ load
    log.info("Loading global clusters from %s", gc_path)
    gc_df = pd.read_parquet(gc_path)
    gc_df["post_id"] = gc_df["post_id"].astype(str)
    gc_df = gc_df[~gc_df["is_noise"] & gc_df["global_cluster_id"].notna()].copy()
    gc_df["global_cluster_id"] = gc_df["global_cluster_id"].astype(int)

    # Deduplicate: keep one row per (post_id, global_cluster_id) — a post can
    # appear in multiple windows but its stance is the same for a given cluster.
    gc_df = gc_df.drop_duplicates(subset=["post_id", "global_cluster_id"])
    log.info("  %d unique (post, cluster) pairs", len(gc_df))

    log.info("Loading themes from %s", themes_path)
    if not themes_path.exists():
        log.error("cluster_themes.parquet not found — run run_theme_labeling.py first")
        sys.exit(1)
    themes_df = pd.read_parquet(themes_path, columns=["global_cluster_id", "theme"])
    themes_df["global_cluster_id"] = themes_df["global_cluster_id"].astype(int)
    themes_df = themes_df.dropna(subset=["theme"])
    theme_map: dict[int, str] = dict(zip(themes_df["global_cluster_id"], themes_df["theme"]))
    log.info("  %d clusters with themes", len(theme_map))

    log.info("Loading post text from %s", repr_path)
    repr_df = pd.read_parquet(repr_path, columns=["post_id", "text"])
    repr_df["post_id"] = repr_df["post_id"].astype(str)
    text_map: dict[str, str] = dict(zip(repr_df["post_id"], repr_df["text"]))
    log.info("  %d posts with text", len(text_map))

    # ------------------------------------------------- resume support
    already_done: set[int] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        already_done = set(existing_df["global_cluster_id"].astype(int).tolist())
        existing_rows = existing_df.to_dict("records")
        log.info("Resuming: %d clusters already classified", len(already_done))

    # ------------------------------------------- determine work queue
    cluster_ids = sorted(
        cid for cid in gc_df["global_cluster_id"].unique()
        if cid in theme_map and cid not in already_done
    )
    log.info("%d clusters to classify", len(cluster_ids))

    # ---------------------------------------- filter by min_posts
    posts_per_cluster = gc_df.groupby("global_cluster_id")["post_id"].count()
    skipped_small = [
        cid for cid in cluster_ids
        if posts_per_cluster.get(cid, 0) < args.min_posts
    ]
    if skipped_small:
        log.info("Skipping %d clusters with < %d posts", len(skipped_small), args.min_posts)
    cluster_ids = [cid for cid in cluster_ids if cid not in set(skipped_small)]

    if not cluster_ids:
        log.info("Nothing to classify — exiting.")
        return

    classifier = PosthocGPTStanceClassifier(
        model=args.model,
        batch_size=args.batch_size,
    )

    new_rows: list[dict] = []

    for i, cid in enumerate(cluster_ids, 1):
        theme = theme_map[cid]
        post_ids = gc_df.loc[gc_df["global_cluster_id"] == cid, "post_id"].tolist()

        posts = [
            Post(post_id=pid, text=text_map.get(pid, ""))
            for pid in post_ids
        ]

        log.info(
            "[%d/%d] cluster %d | posts=%d | theme: %s",
            i, len(cluster_ids), cid, len(posts), theme,
        )

        try:
            posts = classifier.classify_posts(posts, theme)
        except Exception as exc:
            log.error("  cluster %d failed: %s — marking all neutral", cid, exc)
            for p in posts:
                p.stance = "neutral"

        counts = {s: sum(p.stance == s for p in posts) for s in ("support", "oppose", "neutral")}
        log.debug(
            "  cluster %d results — support=%d oppose=%d neutral=%d",
            cid, counts["support"], counts["oppose"], counts["neutral"],
        )

        for p in posts:
            new_rows.append({
                "post_id":            p.post_id,
                "global_cluster_id":  cid,
                "theme":              theme,
                "stance":             p.stance,
            })

        if args.sleep > 0 and i < len(cluster_ids):
            time.sleep(args.sleep)

    # ----------------------------------------------------------- write output
    all_rows = existing_rows + new_rows
    df_out = pd.DataFrame(all_rows)
    df_out["global_cluster_id"] = df_out["global_cluster_id"].astype(int)
    df_out = df_out.sort_values(["global_cluster_id", "post_id"]).reset_index(drop=True)
    df_out.to_parquet(out_path, index=False)

    support = (df_out["stance"] == "support").sum()
    oppose  = (df_out["stance"] == "oppose").sum()
    neutral = (df_out["stance"] == "neutral").sum()
    log.info(
        "Done. %d rows written → %s  (support=%d  oppose=%d  neutral=%d)",
        len(df_out), out_path, support, oppose, neutral,
    )


if __name__ == "__main__":
    main()
