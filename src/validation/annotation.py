"""Stratified annotation sample export and instructions generation."""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATION_N    = 400
ANNOTATION_SEED = 42

_STANCE_LABELS = ["support", "oppose", "neutral"]

# Case-level claim used by the automatic classifier (from posthoc_gpt.py TOPIC_CLAIMS)
TOPIC_CLAIMS = {
    "venezuela": "The U.S. was correct in actions with respect to Maduro and Venezuela.",
    "iran":      "U.S. military action against Iran was appropriate.",
    "russia":    "Russia had legitimate reasons for its military actions in Ukraine.",
}

# Exact system prompt issued to GPT for stance classification
MODEL_SYSTEM_PROMPT = (
    "You are a stance classifier for social media posts.\n\n"
    "Given a narrative claim and a numbered list of posts, classify each post's "
    "stance toward the claim. Reply with a JSON object in this exact format:\n"
    '{\"stances\": [\"support\", \"neutral\", \"oppose\", ...]}\n\n'
    "Rules:\n"
    "- support: the post affirms, endorses, agrees with, or spreads the claim\n"
    "- oppose: the post rejects, counters, disputes, or contradicts the claim\n"
    "- neutral: the post is unrelated to the claim, ambiguous, or takes no clear position\n\n"
    "Return one label per post, in the same order as the input. "
    'Use only the words "support", "oppose", or "neutral".'
)


def _modal_stance(row: pd.Series) -> str:
    pcts = {l: row.get(f"{l}_pct", 0.0) for l in _STANCE_LABELS}
    return max(pcts, key=pcts.get)


def build_annotation_pool(
    cases: list,
    gc_dfs: dict,
    stance_dfs: dict,
    repr_dfs: dict,
    themes_dfs: dict,
) -> pd.DataFrame:
    """
    Build the full pool of eligible posts for annotation.

    Filters to non-noise cluster-assigned posts, assigns each post its
    cluster's modal stance (argmax of pct columns), and joins text + theme.
    """
    frames = []
    for case in cases:
        gc     = gc_dfs[case][~gc_dfs[case]["is_noise"] & gc_dfs[case]["global_cluster_id"].notna()].copy()
        gc["global_cluster_id"] = gc["global_cluster_id"].astype(int)

        stance = stance_dfs[case][["global_cluster_id", "support_pct", "oppose_pct", "neutral_pct"]].copy()
        stance["modal_stance"] = stance.apply(_modal_stance, axis=1)

        gc = gc.merge(stance[["global_cluster_id", "modal_stance"]], on="global_cluster_id", how="left")
        gc = gc.merge(repr_dfs[case][["post_id", "text"]], on="post_id", how="left")

        if case in themes_dfs:
            gc = gc.merge(
                themes_dfs[case][["global_cluster_id", "theme"]].rename(columns={"theme": "cluster_theme"}),
                on="global_cluster_id", how="left",
            )
        else:
            gc["cluster_theme"] = ""

        gc["case"] = case
        frames.append(gc)

    return pd.concat(frames, ignore_index=True)


def stratified_sample(
    pool_df: pd.DataFrame,
    n: int = ANNOTATION_N,
    seed: int = ANNOTATION_SEED,
) -> pd.DataFrame:
    """
    Draw a stratified annotation sample.

    Allocation:
      1. Proportional to cluster-assigned post volume per case.
      2. Within each case, balanced equally across the 3 stance classes.
      3. Fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    case_volumes = pool_df.groupby("case").size()
    case_alloc   = (case_volumes / case_volumes.sum() * n).round().astype(int)

    # Fix rounding so total == n exactly
    diff = n - case_alloc.sum()
    for case in list(case_alloc.index)[:abs(diff)]:
        case_alloc[case] += int(np.sign(diff))

    samples = []
    for case, n_case in case_alloc.items():
        case_pool  = pool_df[pool_df["case"] == case]
        n_per_cls  = max(1, n_case // 3)
        for stance in _STANCE_LABELS:
            cls_pool = case_pool[case_pool["modal_stance"] == stance]
            k = min(n_per_cls, len(cls_pool))
            if k == 0:
                print(f"  WARNING: no '{stance}' posts available for case '{case}'")
                continue
            idxs = rng.choice(len(cls_pool), size=k, replace=False)
            samples.append(cls_pool.iloc[idxs])

    return pd.concat(samples, ignore_index=True).reset_index(drop=True)


def export_annotation_files(
    sample_df: pd.DataFrame,
    output_dir: Path,
    seed: int = ANNOTATION_SEED,
) -> None:
    """
    Export blind annotator CSVs, the gold key, and INSTRUCTIONS.md.

    Files written:
      annotator_A.csv  — post_id, case, post_text, cluster_theme_label, stance_label (blank)
      annotator_B.csv  — identical to A (independent annotation)
      _key.csv         — post_id, model_stance, global_cluster_id, case, stratum, seed
      INSTRUCTIONS.md  — complete annotation protocol
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    blind = pd.DataFrame({
        "post_id":             sample_df["post_id"],
        "case":                sample_df["case"],
        "post_text":           sample_df["text"].fillna(""),
        "cluster_theme_label": sample_df.get("cluster_theme", pd.Series([""] * len(sample_df))).fillna(""),
        "stance_label":        "",
    })

    for annotator in ("A", "B"):
        blind.to_csv(output_dir / f"annotator_{annotator}.csv", index=False)
    print(f"  Exported annotator_A.csv and annotator_B.csv ({len(blind):,} posts each)")

    key = pd.DataFrame({
        "post_id":           sample_df["post_id"],
        "model_stance":      sample_df["modal_stance"],
        "global_cluster_id": sample_df["global_cluster_id"],
        "case":              sample_df["case"],
        "stratum":           sample_df["case"] + "/" + sample_df["modal_stance"],
        "random_seed":       seed,
    })
    key.to_csv(output_dir / "_key.csv", index=False)
    print(f"  Exported _key.csv")

    _write_instructions(output_dir, sample_df["case"].unique().tolist())
    print(f"  Exported INSTRUCTIONS.md")


def _write_instructions(output_dir: Path, cases: list) -> None:
    claims_block = "\n".join(
        f"- **{case.capitalize()}:** {TOPIC_CLAIMS[case]}"
        for case in cases if case in TOPIC_CLAIMS
    )
    instructions = textwrap.dedent(f"""\
    # Stance Annotation Instructions

    ## Overview

    Annotate the stance of social media posts toward a fixed narrative claim.
    Each post comes from a cluster of topically similar posts. A short cluster
    theme label is provided for context — **do not use it to infer the claim**.

    **Do not look up posts online. Label based solely on the text provided.**

    ## Case-Level Claims

    {claims_block}

    ## Labels

    | Label | Definition |
    |-------|-----------|
    | **support** | Post affirms, endorses, agrees with, or spreads the claim. |
    | **oppose**  | Post rejects, counters, disputes, or contradicts the claim. |
    | **neutral** | Post is off-topic, ambiguous, or takes no clear position. |

    These definitions are identical to those given to the automatic classifier.

    ## Instructions

    1. Read the cluster theme label (context only — not the target claim).
    2. Read the post text.
    3. Decide: does this post **support**, **oppose**, or **neutral** toward the
       case-level claim above?
    4. Enter your label in the `stance_label` column.
    5. Leave no rows blank.
    6. Non-English posts → **neutral**.
    7. Too short / garbled to interpret → **neutral**.

    ## Edge Cases

    - Sarcasm that clearly opposes the claim → **oppose**
    - Neutral reporting of a claim (no endorsement) → **neutral**
    - Sharing news that amplifies the claim → **support**
    - Off-topic personal content → **neutral**

    ## System Prompt Given to the Automatic Classifier

    ```
    {MODEL_SYSTEM_PROMPT}
    ```
    """)
    (output_dir / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
