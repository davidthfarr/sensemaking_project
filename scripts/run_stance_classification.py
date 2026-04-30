"""
Post-hoc GPT stance classification — two modes.

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

Usage
-----
python scripts/run_stance_classification.py --case iran
python scripts/run_stance_classification.py --case russia --mode topic
python scripts/run_stance_classification.py --case venezuela --mode cluster --batch-size 30

Input (both modes):
  data/evaluated/<case>/global_clusters.parquet
  data/processed/<case>/posts_repr.parquet

Additional input (cluster mode only):
  data/evaluated/<case>/cluster_themes.parquet

Log: logs/<case>_stance_<mode>.log

Requires OPENAI_API_KEY in .env or the environment.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from sensemaking.data.schemas import Post
from sensemaking.stance.posthoc_gpt import PosthocGPTStanceClassifier

# ---------------------------------------------------------------------------
# Case-level topic claims for topic mode
# ---------------------------------------------------------------------------

TOPIC_CLAIMS = {
    "venezuela": "The U.S. capture of Maduro was justified.",
    "iran":      "U.S. military action against Iran is justified.",
    "russia":    "Russia's invasion of Ukraine is justified.",
}

_TOPIC_SYSTEM = (
    "You are a stance classifier. "
    "Reply with exactly one word: SUPPORT, OPPOSE, or NEUTRAL."
)

MIN_POSTS_PER_CLUSTER = 3
SLEEP_BETWEEN_BATCHES = 1.0


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
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--batch-size", type=int, default=20,
                   help="Posts per GPT call (cluster mode) or topic classification chunk")
    p.add_argument("--min-posts", type=int, default=MIN_POSTS_PER_CLUSTER,
                   help="(cluster mode) skip clusters with fewer posts than this")
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_BATCHES,
                   help="Seconds to sleep between API calls")
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
# Shared data loading
# ---------------------------------------------------------------------------

def load_gc(gc_path: Path, noise_only: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(gc_path)
    df["post_id"] = df["post_id"].astype(str)
    df["is_noise"] = df["is_noise"].astype(bool)
    if not noise_only:
        df = df[~df["is_noise"] & df["global_cluster_id"].notna()].copy()
    df["global_cluster_id"] = pd.to_numeric(df["global_cluster_id"], errors="coerce")
    return df


def load_text_map(repr_path: Path) -> dict[str, str]:
    df = pd.read_parquet(repr_path, columns=["post_id", "text"])
    df["post_id"] = df["post_id"].astype(str)
    return dict(zip(df["post_id"], df["text"]))


# ---------------------------------------------------------------------------
# Cluster mode
# ---------------------------------------------------------------------------

def run_cluster_mode(args: argparse.Namespace, log: logging.Logger) -> None:
    eval_dir    = Path("data/evaluated") / args.case
    repr_path   = Path("data/processed") / args.case / "posts_repr.parquet"
    gc_path     = eval_dir / "global_clusters.parquet"
    themes_path = eval_dir / "cluster_themes.parquet"
    out_path    = eval_dir / "cluster_stance.parquet"

    log.info("Loading global clusters from %s", gc_path)
    gc_df = load_gc(gc_path)
    gc_df["global_cluster_id"] = gc_df["global_cluster_id"].astype(int)
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

    # Resume
    already_done: set[int] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        already_done = set(existing_df["global_cluster_id"].astype(int).tolist())
        existing_rows = existing_df.to_dict("records")
        log.info("Resuming: %d clusters already classified", len(already_done))

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

    classifier = PosthocGPTStanceClassifier(model=args.model, batch_size=args.batch_size)
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
        log.debug("  support=%d oppose=%d neutral=%d", counts["support"], counts["oppose"], counts["neutral"])

        for p in posts:
            new_rows.append({
                "post_id":           p.post_id,
                "global_cluster_id": cid,
                "theme":             theme,
                "stance":            p.stance,
            })

        if args.sleep > 0 and i < len(cluster_ids):
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

def _topic_prompt(text: str, claim: str) -> str:
    return (
        f"Does this post SUPPORT, OPPOSE, or take a NEUTRAL stance toward "
        f"the following claim: '{claim}'?\n"
        f"Reply with exactly one word: SUPPORT, OPPOSE, or NEUTRAL.\n\n"
        f"Post: {text[:500]}"
    )


def _classify_topic_single(client: OpenAI, model: str, text: str, claim: str,
                            max_retries: int = 3, retry_delay: float = 5.0) -> str:
    prompt = _topic_prompt(text, claim)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TOPIC_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            word = resp.choices[0].message.content.strip().lower()
            if word in ("support", "oppose", "neutral"):
                return word
            # Try to salvage a partial match
            for label in ("support", "oppose", "neutral"):
                if label in word:
                    return label
        except RateLimitError:
            time.sleep(retry_delay * (attempt + 1))
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)
    return "neutral"


def _classify_topic_batch(client: OpenAI, model: str, texts: list[str], claim: str,
                           max_retries: int = 3, retry_delay: float = 5.0) -> list[str]:
    """
    Classify a batch via a JSON-array call; fall back to per-post if it fails.
    Uses the same prompt framing as the single-post variant but batched.
    """
    numbered = "\n".join(f"{i + 1}. {t[:400]}" for i, t in enumerate(texts))
    batch_prompt = (
        f"For each post below, does it SUPPORT, OPPOSE, or take a NEUTRAL stance "
        f"toward the following claim: '{claim}'?\n"
        f"Reply with a JSON object: {{\"stances\": [\"SUPPORT\", \"OPPOSE\", ...]}}\n"
        f"One label per post, in order. Use only SUPPORT, OPPOSE, or NEUTRAL.\n\n"
        f"Posts:\n{numbered}"
    )

    import json

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TOPIC_SYSTEM},
                    {"role": "user",   "content": batch_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            data = json.loads(resp.choices[0].message.content)
            labels = data.get("stances", [])
            normalized = [str(l).lower().strip() for l in labels]
            if (len(normalized) == len(texts) and
                    all(l in ("support", "oppose", "neutral") for l in normalized)):
                return normalized
        except RateLimitError:
            time.sleep(retry_delay * (attempt + 1))
        except Exception:
            if attempt == max_retries - 1:
                break
            time.sleep(retry_delay)

    # Fallback: classify one at a time
    return [
        _classify_topic_single(client, model, t, claim, max_retries, retry_delay)
        for t in texts
    ]


def run_topic_mode(args: argparse.Namespace, log: logging.Logger) -> None:
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

    # Resume: skip post_ids already classified
    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        already_done = set(existing_df["post_id"].astype(str).tolist())
        existing_rows = existing_df.to_dict("records")
        log.info("Resuming: %d post_ids already classified", len(already_done))

    # Unique post_ids to classify
    all_post_ids = gc_df["post_id"].unique().tolist()
    todo_ids = [pid for pid in all_post_ids if pid not in already_done]
    log.info("%d unique posts to classify (%d already done)",
             len(todo_ids), len(already_done))

    if not todo_ids:
        log.info("Nothing to classify — skipping to aggregation.")
        stance_map: dict[str, str] = {}
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            stance_map = dict(zip(existing_df["post_id"].astype(str), existing_df["stance"]))
    else:
        client = OpenAI()
        new_stances: dict[str, str] = {}

        total_batches = (len(todo_ids) + args.batch_size - 1) // args.batch_size
        for b_idx, start in enumerate(range(0, len(todo_ids), args.batch_size), 1):
            batch_ids = todo_ids[start: start + args.batch_size]
            batch_texts = [text_map.get(pid, "") for pid in batch_ids]

            log.info("[batch %d/%d] %d posts", b_idx, total_batches, len(batch_ids))

            try:
                labels = _classify_topic_batch(client, args.model, batch_texts, claim)
            except Exception as exc:
                log.error("  batch %d failed: %s — marking all neutral", b_idx, exc)
                labels = ["neutral"] * len(batch_ids)

            for pid, label in zip(batch_ids, labels):
                new_stances[pid] = label

            counts = {s: labels.count(s) for s in ("support", "oppose", "neutral")}
            log.debug("  support=%d oppose=%d neutral=%d",
                      counts["support"], counts["oppose"], counts["neutral"])

            if args.sleep > 0 and start + args.batch_size < len(todo_ids):
                time.sleep(args.sleep)

        # Merge with existing classified posts
        existing_stance_map: dict[str, str] = {}
        if existing_rows:
            existing_stance_map = {
                r["post_id"]: r["stance"] for r in existing_rows
            }
        stance_map = {**existing_stance_map, **new_stances}

    # Fan out: one row per (post_id, window) row in global_clusters
    out_rows = []
    for row in gc_df.itertuples(index=False):
        pid = row.post_id
        gid = getattr(row, "global_cluster_id", None)
        window = row.window
        stance = stance_map.get(pid, "neutral")
        out_rows.append({
            "post_id":           pid,
            "global_cluster_id": gid,
            "window":            window,
            "stance":            stance,
        })

    df_out = pd.DataFrame(out_rows)
    df_out.to_parquet(out_path, index=False)
    log.info("Written %d rows → %s", len(df_out), out_path)

    # ------------------------------------------------- per-window aggregates
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
    load_dotenv()
    args = parse_args()

    log = setup_logging(args.case, args.mode, Path("logs"))
    log.info("=== Stance classification: case=%s  mode=%s ===", args.case, args.mode)
    log.info("model=%s  batch_size=%d", args.model, args.batch_size)

    if args.mode == "cluster":
        run_cluster_mode(args, log)
    else:
        run_topic_mode(args, log)


if __name__ == "__main__":
    main()
