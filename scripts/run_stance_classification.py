"""
Post-hoc stance classification — two modes, three backends.

cluster mode (default)
  Classify every post against its own cluster's stationary theme.
  Output: data/evaluated/<case>/cluster_stance.parquet
          post_id, global_cluster_id, theme, stance
  Resume-safe: skips clusters already present in the output file.

topic mode
  Classify every post against a fixed case-level claim.
  Each unique post is classified once; results are fanned out to all
  (post_id, window) rows so per-window aggregates can be computed.
  Output: data/evaluated/<case>/topic_stance.parquet
            post_id, global_cluster_id, window, stance
          data/evaluated/<case>/topic_stance_by_window.parquet
            window, n_posts, support_pct, oppose_pct, neutral_pct
  Resume-safe: skips post_ids already present in the output file.

Backends (--model):
  gpt-4o-mini   OpenAI gpt-4o-mini  (default; requires OPENAI_API_KEY)
  gpt-4o        OpenAI gpt-4o       (requires OPENAI_API_KEY)
  llama         meta-llama/Meta-Llama-3-8B-Instruct loaded locally via
                transformers (requires HF access; uses --device)

Usage
-----
python scripts/run_stance_classification.py --case iran
python scripts/run_stance_classification.py --case russia --mode topic
python scripts/run_stance_classification.py --case venezuela --model llama --device cuda
python scripts/run_stance_classification.py --case iran --model gpt-4o --batch-size 30

Input (both modes):
  data/evaluated/<case>/global_clusters.parquet
  data/processed/<case>/posts_repr.parquet

Additional input (cluster mode only):
  data/evaluated/<case>/cluster_themes.parquet

Log: logs/<case>_stance_<mode>.log

OPENAI_API_KEY (from .env) required only for gpt-4o-mini / gpt-4o backends.
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Union

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import pandas as pd
from dotenv import load_dotenv

from sensemaking.data.schemas import Post
from sensemaking.stance.posthoc_gpt import (
    LocalLlamaClassifier,
    PosthocGPTStanceClassifier,
)

# ---------------------------------------------------------------------------
# Case-level topic claims
# ---------------------------------------------------------------------------

TOPIC_CLAIMS = {
    "venezuela": "The U.S. capture of Maduro was justified.",
    "iran":      "U.S. military action against Iran is justified.",
    "russia":    "Russia's invasion of Ukraine is justified.",
}

MIN_POSTS_PER_CLUSTER = 3
SLEEP_BETWEEN_BATCHES = 1.0

StanceClassifier = Union[PosthocGPTStanceClassifier, LocalLlamaClassifier]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=list(TOPIC_CLAIMS))
    p.add_argument("--mode", choices=["cluster", "topic"], default="cluster",
                   help="cluster: per-cluster theme; topic: fixed case-level claim")
    p.add_argument("--model", choices=["gpt-4o-mini", "gpt-4o", "llama"],
                   default="gpt-4o-mini",
                   help="gpt-4o-mini / gpt-4o: OpenAI API; llama: local inference")
    p.add_argument("--device", default="cuda",
                   help="Compute device for llama backend (cuda or cpu)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Posts per call (default: 20 for GPT, 16 for llama)")
    p.add_argument("--min-posts", type=int, default=MIN_POSTS_PER_CLUSTER,
                   help="(cluster mode) skip clusters with fewer posts than this")
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_BATCHES,
                   help="Seconds to sleep between API calls (GPT only; 0 to disable)")
    p.add_argument("--overwrite", action="store_true",
                   help="Reclassify from scratch even if output already exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(case: str, mode: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case}_stance_{mode}.log"

    logger = logging.getLogger(f"stance.{case}.{mode}")
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


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def build_classifier(args: argparse.Namespace, log: logging.Logger) -> StanceClassifier:
    if args.model == "llama":
        batch_size = args.batch_size if args.batch_size is not None else 32
        log.info("Loading LocalLlamaClassifier (device_map=auto, batch_size=%d)…", batch_size)
        clf = LocalLlamaClassifier(batch_size=batch_size)
        return clf

    # GPT backends
    batch_size = args.batch_size if args.batch_size is not None else 20
    log.info("Using PosthocGPTStanceClassifier (model=%s, batch_size=%d)",
             args.model, batch_size)
    return PosthocGPTStanceClassifier(model=args.model, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Shared data loading helpers
# ---------------------------------------------------------------------------

def load_gc_nonoise(gc_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(gc_path)
    df["post_id"] = df["post_id"].astype(str)
    df["is_noise"] = df["is_noise"].astype(bool)
    df = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    return df


def load_text_map(repr_path: Path) -> dict[str, str]:
    df = pd.read_parquet(repr_path, columns=["post_id", "text"])
    df["post_id"] = df["post_id"].astype(str)
    return dict(zip(df["post_id"], df["text"]))


def is_gpt(args: argparse.Namespace) -> bool:
    return args.model in ("gpt-4o-mini", "gpt-4o")


# ---------------------------------------------------------------------------
# Cluster mode
# ---------------------------------------------------------------------------

def run_cluster_mode(
    args: argparse.Namespace,
    classifier: StanceClassifier,
    log: logging.Logger,
) -> None:
    eval_dir    = Path("data/evaluated") / args.case
    repr_path   = Path("data/processed") / args.case / "posts_repr.parquet"
    gc_path     = eval_dir / "global_clusters.parquet"
    themes_path = eval_dir / "cluster_themes.parquet"
    out_path    = eval_dir / "cluster_stance.parquet"

    log.info("Loading global clusters from %s", gc_path)
    gc_df = load_gc_nonoise(gc_path)
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
    text_map = load_text_map(repr_path)
    log.info("  %d posts with text", len(text_map))

    # Resume (skipped when --overwrite is set)
    already_done: set[int] = set()
    existing_rows: list[dict] = []
    if out_path.exists() and not args.overwrite:
        existing_df = pd.read_parquet(out_path)
        already_done = set(existing_df["global_cluster_id"].astype(int).tolist())
        existing_rows = existing_df.to_dict("records")
        log.info("Resuming: %d clusters already classified", len(already_done))
    elif out_path.exists() and args.overwrite:
        log.info("--overwrite set: ignoring existing %s", out_path.name)

    posts_per_cluster = gc_df.groupby("global_cluster_id")["post_id"].count()
    cluster_ids = sorted(
        cid for cid in gc_df["global_cluster_id"].unique()
        if cid in theme_map
        and cid not in already_done
        and posts_per_cluster.get(cid, 0) >= args.min_posts
    )
    skipped = sum(
        1 for cid in gc_df["global_cluster_id"].unique()
        if cid in theme_map and cid not in already_done
        and posts_per_cluster.get(cid, 0) < args.min_posts
    )
    if skipped:
        log.info("Skipping %d clusters with < %d posts", skipped, args.min_posts)
    log.info("%d clusters to classify", len(cluster_ids))

    if not cluster_ids:
        log.info("Nothing to classify — exiting.")
        return

    new_rows: list[dict] = []

    for i, cid in enumerate(cluster_ids, 1):
        theme = theme_map[cid]
        post_ids = gc_df.loc[gc_df["global_cluster_id"] == cid, "post_id"].tolist()
        posts = [Post(post_id=pid, text=text_map.get(pid, "")) for pid in post_ids]

        log.info("[%d/%d] cluster %d | posts=%d | theme: %s",
                 i, len(cluster_ids), cid, len(posts), theme)

        try:
            posts = classifier.classify_posts(posts, theme)
        except Exception as exc:
            log.error("  cluster %d failed: %s — marking all neutral", cid, exc)
            for p in posts:
                p.stance = "neutral"

        counts = {s: sum(p.stance == s for p in posts) for s in ("support", "oppose", "neutral")}
        log.debug("  support=%d oppose=%d neutral=%d",
                  counts["support"], counts["oppose"], counts["neutral"])

        for p in posts:
            new_rows.append({
                "post_id":           p.post_id,
                "global_cluster_id": cid,
                "theme":             theme,
                "stance":            p.stance,
            })

        if is_gpt(args) and args.sleep > 0 and i < len(cluster_ids):
            time.sleep(args.sleep)

    all_rows = existing_rows + new_rows
    df_out = pd.DataFrame(all_rows)
    df_out["global_cluster_id"] = df_out["global_cluster_id"].astype(int)
    df_out = df_out.sort_values(["global_cluster_id", "post_id"]).reset_index(drop=True)
    df_out.to_parquet(out_path, index=False)

    support = (df_out["stance"] == "support").sum()
    oppose  = (df_out["stance"] == "oppose").sum()
    neutral = (df_out["stance"] == "neutral").sum()
    log.info("Done. %d rows → %s  (support=%d  oppose=%d  neutral=%d)",
             len(df_out), out_path, support, oppose, neutral)


# ---------------------------------------------------------------------------
# Topic mode
# ---------------------------------------------------------------------------

def run_topic_mode(
    args: argparse.Namespace,
    classifier: StanceClassifier,
    log: logging.Logger,
) -> None:
    claim = TOPIC_CLAIMS[args.case]
    log.info("Topic claim: %s", claim)

    eval_dir  = Path("data/evaluated") / args.case
    repr_path = Path("data/processed") / args.case / "posts_repr.parquet"
    gc_path   = eval_dir / "global_clusters.parquet"
    out_path  = eval_dir / "topic_stance.parquet"
    agg_path  = eval_dir / "topic_stance_by_window.parquet"

    # Load ALL rows (noise and non-noise) to preserve window tracking
    log.info("Loading global clusters from %s", gc_path)
    gc_df = pd.read_parquet(gc_path)
    gc_df["post_id"] = gc_df["post_id"].astype(str)
    gc_df["is_noise"] = gc_df["is_noise"].astype(bool)
    log.info("  %d total post-window rows", len(gc_df))

    log.info("Loading post text from %s", repr_path)
    text_map = load_text_map(repr_path)
    log.info("  %d posts with text", len(text_map))

    # Resume: skip post_ids already classified (skipped when --overwrite is set)
    existing_stance_map: dict[str, str] = {}
    if out_path.exists() and not args.overwrite:
        existing_df = pd.read_parquet(out_path)
        existing_stance_map = dict(
            zip(existing_df["post_id"].astype(str), existing_df["stance"])
        )
        log.info("Resuming: %d post_ids already classified", len(existing_stance_map))
    elif out_path.exists() and args.overwrite:
        log.info("--overwrite set: ignoring existing %s", out_path.name)

    all_post_ids = gc_df["post_id"].unique().tolist()
    todo_ids = [pid for pid in all_post_ids if pid not in existing_stance_map]
    log.info("%d unique posts to classify (%d already done)",
             len(todo_ids), len(existing_stance_map))

    new_stances: dict[str, str] = {}

    if todo_ids:
        # Build Post objects and classify in one pass using the classifier's
        # internal batching — pass the topic claim as the theme.
        batch_size = classifier.batch_size
        total_batches = (len(todo_ids) + batch_size - 1) // batch_size

        for b_idx, start in enumerate(range(0, len(todo_ids), batch_size), 1):
            batch_ids = todo_ids[start: start + batch_size]
            batch_posts = [
                Post(post_id=pid, text=text_map.get(pid, ""))
                for pid in batch_ids
            ]

            log.info("[batch %d/%d] %d posts", b_idx, total_batches, len(batch_ids))

            try:
                # Pass claim as theme — both classifiers treat this identically
                classified = classifier.classify_posts(batch_posts, claim)
            except Exception as exc:
                log.error("  batch %d failed: %s — marking all neutral", b_idx, exc)
                classified = batch_posts
                for p in classified:
                    p.stance = "neutral"

            for p in classified:
                new_stances[p.post_id] = p.stance or "neutral"

            counts = {s: sum(p.stance == s for p in classified)
                      for s in ("support", "oppose", "neutral")}
            log.debug("  support=%d oppose=%d neutral=%d",
                      counts["support"], counts["oppose"], counts["neutral"])

            if is_gpt(args) and args.sleep > 0 and start + batch_size < len(todo_ids):
                time.sleep(args.sleep)

    stance_map = {**existing_stance_map, **new_stances}

    # Fan out to all (post_id, window) rows
    out_rows = [
        {
            "post_id":           row.post_id,
            "global_cluster_id": getattr(row, "global_cluster_id", None),
            "window":            row.window,
            "stance":            stance_map.get(row.post_id, "neutral"),
        }
        for row in gc_df.itertuples(index=False)
    ]

    df_out = pd.DataFrame(out_rows)
    df_out.to_parquet(out_path, index=False)
    log.info("Written %d rows → %s", len(df_out), out_path)

    # Per-window aggregates
    agg_rows = []
    for window, wdf in df_out.groupby("window"):
        n = len(wdf)
        support = (wdf["stance"] == "support").sum()
        oppose  = (wdf["stance"] == "oppose").sum()
        neutral = (wdf["stance"] == "neutral").sum()
        agg_rows.append({
            "window":      window,
            "n_posts":     n,
            "support_pct": support / n if n > 0 else 0.0,
            "oppose_pct":  oppose  / n if n > 0 else 0.0,
            "neutral_pct": neutral / n if n > 0 else 0.0,
        })

    agg_df = pd.DataFrame(agg_rows).sort_values("window").reset_index(drop=True)
    agg_df.to_parquet(agg_path, index=False)
    log.info("Written %d windows → %s", len(agg_df), agg_path)

    total_support = (df_out["stance"] == "support").sum()
    total_oppose  = (df_out["stance"] == "oppose").sum()
    total_neutral = (df_out["stance"] == "neutral").sum()
    log.info(
        "Done. Overall: support=%d (%.1f%%)  oppose=%d (%.1f%%)  neutral=%d (%.1f%%)",
        total_support, 100 * total_support / len(df_out),
        total_oppose,  100 * total_oppose  / len(df_out),
        total_neutral, 100 * total_neutral / len(df_out),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Load .env unconditionally — harmless if absent; required for GPT backends
    load_dotenv()
    args = parse_args()

    log = setup_logging(args.case, args.mode, Path("logs"))
    log.info("=== Stance classification: case=%s  mode=%s  model=%s ===",
             args.case, args.mode, args.model)

    if is_gpt(args):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            log.error("OPENAI_API_KEY not set — required for model '%s'", args.model)
            sys.exit(1)

    classifier = build_classifier(args, log)

    if args.mode == "cluster":
        run_cluster_mode(args, classifier, log)
    else:
        run_topic_mode(args, classifier, log)


if __name__ == "__main__":
    main()
