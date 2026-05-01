# build_drift_html.py
"""
Single-file HTML Narrative Drift visualization (no server required).

Creates ONE self-contained HTML file with:
- global 2D projection of cluster centroids (PCA)
- slider over windows (frames)
- cluster-driven coloring via stable global lineage IDs
- hover shows representative (deduped) tweets + size (+ optional stance)

Run (from project root):
  pip install -q plotly pandas pyarrow scikit-learn numpy
  python build_drift_html.py

Inputs:
  data/processed/posts_repr.parquet
    required: post_id, timestamp, embedding
    strongly recommended: text
    optional: stance
  data/evaluated/daily/*.parquet
    required: post_id, window (or inferred from filename), cluster_id, is_noise

Output:
  narrative_drift.html   (self-contained; open/download it)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go

import environment


# -------------------------
# Paths / Config
# -------------------------
REP_PATH = Path(environment.PROCESSED_FILE_PATH())
EVAL_DIR = Path(environment.EVALUATED_DIR())

OUT_HTML = Path(environment.OUTPUT_HTML_FILE_PATH())

TOP_K_REP = 4                  # reps shown in hover
MIN_CLUSTER_POSTS = 8          # filter small clusters in viz
LINEAGE_SIM_THRESHOLD = 0.85   # centroid match threshold
DIVERSITY_SIM_THRESHOLD = 0.92 # rep tweet semantic diversity

MAX_WINDOWS: Optional[int] = None  # set to e.g., 80 to speed up

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d",
]


# -------------------------
# Helpers
# -------------------------
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"^rt\s+@\w+:\s*", "", t)  # strip RT prefix
    t = re.sub(r"http\S+", "", t)         # strip URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _as_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.array(x, dtype=np.float32)


def pick_representatives(
    df_cluster: pd.DataFrame,
    top_k: int,
    diversity_sim_threshold: float,
) -> List[Dict]:
    """Central + diverse + deduped exemplars."""
    if len(df_cluster) == 0:
        return []

    dfc = df_cluster.copy()

    # Prefer 'text' but accept 'tweet' if present
    if "text" not in dfc.columns and "tweet" in dfc.columns:
        dfc = dfc.rename(columns={"tweet": "text"})

    if "text" not in dfc.columns:
        # No text available; return empty reps but still allow clustering visuals
        return []

    # Lexical dedupe (kills RT spam / duplicates)
    dfc["text_norm"] = dfc["text"].map(normalize_text)
    dfc = dfc.drop_duplicates(subset=["text_norm"])
    if len(dfc) == 0:
        return []

    embs = np.stack(dfc["embedding"].values)
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(embs, centroid).ravel()
    dfc["sim_to_centroid"] = sims
    dfc = dfc.sort_values("sim_to_centroid", ascending=False)

    selected = []
    selected_embs: List[np.ndarray] = []

    for _, row in dfc.iterrows():
        emb = row["embedding"]

        if selected_embs:
            s = cosine_similarity([emb], selected_embs).ravel()
            if np.max(s) >= diversity_sim_threshold:
                continue

        selected.append({
            "post_id": row.get("post_id", None),
            "text": row["text"],
            "stance": row.get("stance", None),
            "sim": float(row["sim_to_centroid"]),
        })
        selected_embs.append(emb)

        if len(selected) >= top_k:
            break

    return selected


def greedy_window_matching(
    prev_centroids: pd.DataFrame,
    curr_centroids: pd.DataFrame,
    threshold: float,
) -> Dict[int, Optional[int]]:
    """One-to-one greedy matching by cosine similarity."""
    if len(prev_centroids) == 0 or len(curr_centroids) == 0:
        return {i: None for i in range(len(curr_centroids))}

    P = np.stack(prev_centroids["centroid"].values)
    C = np.stack(curr_centroids["centroid"].values)

    S = cosine_similarity(C, P)
    pairs: List[Tuple[float, int, int]] = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            pairs.append((float(S[i, j]), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    curr_used, prev_used = set(), set()
    match = {i: None for i in range(S.shape[0])}

    for sim, i, j in pairs:
        if sim < threshold:
            break
        if i in curr_used or j in prev_used:
            continue
        match[i] = j
        curr_used.add(i)
        prev_used.add(j)

    return match


def compute_cluster_summaries() -> Tuple[pd.DataFrame, List[str]]:
    if not REP_PATH.exists():
        raise FileNotFoundError(f"Missing representation file: {REP_PATH}")
    eval_files = sorted(EVAL_DIR.glob("*.parquet"))
    if not eval_files:
        raise FileNotFoundError(f"No evaluated parquet files in: {EVAL_DIR}")

    if MAX_WINDOWS is not None:
        eval_files = eval_files[:MAX_WINDOWS]

    df_repr = pd.read_parquet(REP_PATH)
    df_repr.columns = df_repr.columns.str.strip()

    # Normalize common schema variants
    if "text" not in df_repr.columns and "tweet" in df_repr.columns:
        df_repr = df_repr.rename(columns={"tweet": "text"})

    # Required
    required = {"post_id", "timestamp", "embedding"}
    missing = required - set(df_repr.columns)
    if missing:
        raise ValueError(f"{REP_PATH} missing columns: {sorted(missing)}")

    df_repr = df_repr.copy()
    df_repr["post_id"] = df_repr["post_id"].astype(str)
    df_repr["embedding"] = df_repr["embedding"].map(_as_np)

    rows = []

    for f in eval_files:
        df_eval = pd.read_parquet(f)
        df_eval.columns = df_eval.columns.str.strip()

        if "window" not in df_eval.columns:
            df_eval["window"] = f.stem

        df_eval["post_id"] = df_eval["post_id"].astype(str)

        # keep only clustered posts
        df_eval = df_eval.dropna(subset=["cluster_id"])
        if len(df_eval) == 0:
            continue
        df_eval["cluster_id"] = df_eval["cluster_id"].astype(int)

        df = df_eval.merge(df_repr, on="post_id", how="left")
        df.columns = df.columns.str.strip()

        # Coalesce possible merge suffixes
        # text
        if "text" not in df.columns:
            for c in ["text_x", "text_y", "tweet", "tweet_x", "tweet_y"]:
                if c in df.columns:
                    df = df.rename(columns={c: "text"})
                    break
        if "text_x" in df.columns and "text_y" in df.columns:
            df["text"] = df["text_y"].fillna(df["text_x"])

        # stance
        if "stance" not in df.columns:
            for c in ["stance_x", "stance_y", "stance_repr"]:
                if c in df.columns:
                    df = df.rename(columns={c: "stance"})
                    break
        if "stance_x" in df.columns and "stance_y" in df.columns:
            df["stance"] = df["stance_y"].fillna(df["stance_x"])

        # Drop missing embeddings; text may be absent (we still visualize)
        df = df.dropna(subset=["embedding"])
        if len(df) == 0:
            continue

        window = str(df["window"].iloc[0])

        for cid, g in df.groupby("cluster_id"):
            n = len(g)
            if n < MIN_CLUSTER_POSTS:
                continue

            embs = np.stack(g["embedding"].values)
            centroid = embs.mean(axis=0)

            reps = pick_representatives(g, TOP_K_REP, DIVERSITY_SIM_THRESHOLD)

            has_stance = "stance" in g.columns
            mean_stance = float(np.mean(g["stance"].values)) if has_stance else None

            rows.append({
                "window": window,
                "cluster_id": int(cid),
                "num_posts": int(n),
                "centroid": centroid,
                "mean_stance": mean_stance,
                "representatives": reps,
            })

    if not rows:
        raise RuntimeError("No clusters found after filtering. Lower MIN_CLUSTER_POSTS or check evaluated files.")

    clusters = pd.DataFrame(rows)

    # Order windows chronologically if possible
    try:
        win_dt = pd.to_datetime(clusters["window"])
        clusters["window_dt"] = win_dt
        clusters = clusters.sort_values(["window_dt", "cluster_id"]).reset_index(drop=True)
        windows = clusters["window"].drop_duplicates().tolist()
    except Exception:
        clusters = clusters.sort_values(["window", "cluster_id"]).reset_index(drop=True)
        windows = clusters["window"].drop_duplicates().tolist()

    # Assign stable global lineage IDs
    clusters["global_cluster_id"] = -1
    gid_counter = 0
    prev_df = pd.DataFrame()

    for w in windows:
        curr = clusters[clusters["window"] == w].copy().reset_index()
        if prev_df.empty:
            for i in range(len(curr)):
                clusters.loc[curr.loc[i, "index"], "global_cluster_id"] = gid_counter
                gid_counter += 1
        else:
            prev = prev_df.copy().reset_index(drop=True)
            match = greedy_window_matching(prev, curr, LINEAGE_SIM_THRESHOLD)
            for i in range(len(curr)):
                idx = curr.loc[i, "index"]
                j = match.get(i, None)
                if j is None:
                    clusters.loc[idx, "global_cluster_id"] = gid_counter
                    gid_counter += 1
                else:
                    clusters.loc[idx, "global_cluster_id"] = int(prev.loc[j, "global_cluster_id"])

        prev_df = clusters[clusters["window"] == w].copy()

    # Global 2D projection
    X = np.stack(clusters["centroid"].values)
    pca = PCA(n_components=2, random_state=0)
    XY = pca.fit_transform(X)
    clusters["x"] = XY[:, 0]
    clusters["y"] = XY[:, 1]

    # Stable color per global ID
    uniq = sorted(clusters["global_cluster_id"].unique().tolist())
    cmap = {gid: PALETTE[i % len(PALETTE)] for i, gid in enumerate(uniq)}
    clusters["color"] = clusters["global_cluster_id"].map(cmap)

    if "window_dt" in clusters.columns:
        clusters = clusters.drop(columns=["window_dt"])

    return clusters.reset_index(drop=True), windows


def build_html(clusters: pd.DataFrame, windows: List[str]) -> go.Figure:
    # Precompute drift segments between consecutive windows (optional)
    drift_by_window: Dict[str, List[Tuple[float, float, float, float]]] = {}

    for i in range(1, len(windows)):
        w_prev, w_curr = windows[i - 1], windows[i]
        prev = clusters[clusters["window"] == w_prev][["global_cluster_id", "x", "y"]]
        curr = clusters[clusters["window"] == w_curr][["global_cluster_id", "x", "y"]]
        m = curr.merge(prev, on="global_cluster_id", suffixes=("_curr", "_prev"))
        segs = [(r.x_prev, r.y_prev, r.x_curr, r.y_curr) for r in m.itertuples(index=False)]
        drift_by_window[w_curr] = segs

    # Initial window
    w0 = windows[0]
    df0 = clusters[clusters["window"] == w0].copy()

    def hover_text(row) -> str:
        reps = row["representatives"] or []
        rep_lines = []
        for r in reps:
            txt = normalize_text(r.get("text", ""))[:220]
            rep_lines.append(f"• {txt}")
        rep_block = "<br>".join(rep_lines) if rep_lines else "(no text stored)"
        stance_line = ""
        if row.get("mean_stance", None) is not None:
            stance_line = f"<br><b>mean stance</b>: {row['mean_stance']:.3f}"
        return (
            f"<b>window</b>: {row['window']}<br>"
            f"<b>cluster</b>: {row['cluster_id']}<br>"
            f"<b>global id</b>: {row['global_cluster_id']}<br>"
            f"<b>posts</b>: {row['num_posts']}"
            f"{stance_line}"
            f"<br><br><b>representatives</b><br>{rep_block}"
        )

    df0["hover"] = df0.apply(hover_text, axis=1)

    fig = go.Figure()

    # base scatter
    fig.add_trace(
        go.Scatter(
            x=df0["x"],
            y=df0["y"],
            mode="markers",
            marker=dict(
                size=np.clip(df0["num_posts"].values, 6, 60),
                color=df0["color"],
                opacity=0.85,
                line=dict(width=0.5, color="#111"),
            ),
            hovertext=df0["hover"],
            hoverinfo="text",
            name="Clusters",
        )
    )

    # base drift lines for first window (none)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(width=1, color="#666"),
            opacity=0.35,
            hoverinfo="skip",
            showlegend=False,
            name="Drift",
        )
    )

    # frames per window
    frames = []
    for w in windows:
        dfw = clusters[clusters["window"] == w].copy()
        dfw["hover"] = dfw.apply(hover_text, axis=1)

        # drift segments into this window
        segs = drift_by_window.get(w, [])
        lx, ly = [], []
        for (x0, y0, x1, y1) in segs:
            lx += [x0, x1, None]
            ly += [y0, y1, None]

        frames.append(
            go.Frame(
                name=w,
                data=[
                    go.Scatter(
                        x=dfw["x"],
                        y=dfw["y"],
                        mode="markers",
                        marker=dict(
                            size=np.clip(dfw["num_posts"].values, 6, 60),
                            color=dfw["color"],
                            opacity=0.85,
                            line=dict(width=0.5, color="#111"),
                        ),
                        hovertext=dfw["hover"],
                        hoverinfo="text",
                    ),
                    go.Scatter(
                        x=lx,
                        y=ly,
                        mode="lines",
                        line=dict(width=1, color="#666"),
                        opacity=0.35,
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                ],
            )
        )

    fig.frames = frames

    # slider
    steps = [
        dict(
            method="animate",
            args=[[w], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=w,
        )
        for w in windows
    ]

    fig.update_layout(
        template="plotly_dark",
        title="Narrative Drift (cluster centroids, global PCA projection)",
        xaxis_title="PCA-1",
        yaxis_title="PCA-2",
        margin=dict(l=10, r=10, t=50, b=10),
        sliders=[dict(active=0, steps=steps, x=0.06, y=0.02, len=0.92)],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.06,
                y=0.09,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=700, redraw=True), transition=dict(duration=200), fromcurrent=True)],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")],
                    ),
                ],
            )
        ],
    )

    return fig


def main():
    print("Building cluster summaries...")
    clusters, windows = compute_cluster_summaries()
    print(f"Clusters snapshots: {len(clusters):,} across windows: {len(windows)}")

    print("Building interactive figure...")
    fig = build_html(clusters, windows)

    print(f"Writing self-contained HTML to: {OUT_HTML}")
    fig.write_html(
        OUT_HTML,
        include_plotlyjs=True,   # embed plotly.js => truly single-file
        full_html=True,
        auto_open=False,
    )
    print("Done.")


if __name__ == "__main__":
    main()
