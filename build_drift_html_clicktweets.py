# build_drift_html_clicktweets.py
"""
Single-file HTML Narrative Drift visualization (no server required)
+ Click a cluster centroid to show tweet points (sampled) for that cluster.

Design:
- Plot 1: Cluster centroids over time (global PCA space)
- Plot 2 (overlay): Tweets within clicked cluster in same PCA space
  (tweet points are sampled per cluster to keep HTML small)

Inputs:
  data/processed/posts_repr.parquet
    required: post_id, timestamp, embedding
    recommended: text
    optional: stance
  data/evaluated/daily/*.parquet
    required: post_id, window (or inferred), cluster_id (NaN if noise)

Output:
  narrative_drift_click.html
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go


# -------------------------
# Paths / Config
# -------------------------
REP_PATH = Path("data/processed/posts_repr_ck.parquet")
EVAL_DIR = Path("data/evaluated/ck/hourly")
OUT_HTML = Path("ck_narrative_drift_click.html")

TOP_K_REP = 4
MIN_CLUSTER_POSTS = 8

# lineage matching between adjacent windows (centroid-to-centroid)
LINEAGE_SIM_THRESHOLD = 0.8

# tweet sampling within a cluster (for click overlay)
TWEETS_PER_CLUSTER = 250            # keep this modest: 100–500
TWEET_DIVERSITY_SIM_THRESHOLD = 0.92  # avoid near-duplicate points

MAX_WINDOWS: Optional[int] = None

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
    t = str(text).strip()
    t = re.sub(r"^RT\s+@\w+:\s*", "", t)  # remove RT prefix
    t = re.sub(r"http\S+", "", t)         # remove URLs
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
    """Central + diverse + deduped exemplars (for hover summary)."""
    if len(df_cluster) == 0:
        return []

    dfc = df_cluster.copy()

    if "text" not in dfc.columns and "tweet" in dfc.columns:
        dfc = dfc.rename(columns={"tweet": "text"})

    if "text" not in dfc.columns:
        return []

    dfc["text_norm"] = dfc["text"].map(normalize_text).str.lower()
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
            "post_id": str(row.get("post_id", "")),
            "text": row.get("text", ""),
            "stance": row.get("stance", None),
            "sim": float(row["sim_to_centroid"]),
        })
        selected_embs.append(emb)

        if len(selected) >= top_k:
            break

    return selected


def sample_cluster_tweets(
    df_cluster: pd.DataFrame,
    max_points: int,
    diversity_sim_threshold: float,
) -> pd.DataFrame:
    """
    Sample up to max_points tweets from a cluster, prioritizing centrality
    while removing duplicates (RTs, repeated text) and near-duplicates (semantic).
    Returns a DF with columns: post_id, text, embedding.
    """
    if len(df_cluster) == 0:
        return df_cluster.iloc[0:0].copy()

    dfc = df_cluster.copy()

    if "text" not in dfc.columns and "tweet" in dfc.columns:
        dfc = dfc.rename(columns={"tweet": "text"})

    if "text" not in dfc.columns:
        # Can't show tweet hover if no text stored
        return dfc.iloc[0:0].copy()

    # lexical dedupe first
    dfc["text_norm"] = dfc["text"].map(normalize_text).str.lower()
    dfc = dfc.drop_duplicates(subset=["text_norm"])
    if len(dfc) == 0:
        return dfc

    embs = np.stack(dfc["embedding"].values)
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(embs, centroid).ravel()
    dfc["sim_to_centroid"] = sims
    dfc = dfc.sort_values("sim_to_centroid", ascending=False)

    keep_rows = []
    keep_embs: List[np.ndarray] = []

    for _, row in dfc.iterrows():
        emb = row["embedding"]
        if keep_embs:
            s = cosine_similarity([emb], keep_embs).ravel()
            if np.max(s) >= diversity_sim_threshold:
                continue

        keep_rows.append(row)
        keep_embs.append(emb)

        if len(keep_rows) >= max_points:
            break

    return pd.DataFrame(keep_rows)


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


def compute_clusters_and_windows() -> Tuple[pd.DataFrame, List[str]]:
    if not REP_PATH.exists():
        raise FileNotFoundError(f"Missing representation file: {REP_PATH}")
    eval_files = sorted(EVAL_DIR.glob("*.parquet"))
    if not eval_files:
        raise FileNotFoundError(f"No evaluated parquet files in: {EVAL_DIR}")
    if MAX_WINDOWS is not None:
        eval_files = eval_files[:MAX_WINDOWS]

    df_repr = pd.read_parquet(REP_PATH)
    df_repr.columns = df_repr.columns.str.strip()

    if "text" not in df_repr.columns and "tweet" in df_repr.columns:
        df_repr = df_repr.rename(columns={"tweet": "text"})

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
        df_eval = df_eval.dropna(subset=["cluster_id"])
        if len(df_eval) == 0:
            continue
        df_eval["cluster_id"] = df_eval["cluster_id"].astype(int)

        df = df_eval.merge(df_repr, on="post_id", how="left")

        df.columns = df.columns.str.strip()

        # coalesce common suffixes
        if "text" not in df.columns:
            for c in ["text_x", "text_y", "tweet", "tweet_x", "tweet_y"]:
                if c in df.columns:
                    df = df.rename(columns={c: "text"})
                    break
        if "text_x" in df.columns and "text_y" in df.columns:
            df["text"] = df["text_y"].fillna(df["text_x"])

        if "stance" not in df.columns:
            for c in ["stance_x", "stance_y", "stance_repr"]:
                if c in df.columns:
                    df = df.rename(columns={c: "stance"})
                    break
        if "stance_x" in df.columns and "stance_y" in df.columns:
            df["stance"] = df["stance_y"].fillna(df["stance_x"])

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

            reps = pick_representatives(g, TOP_K_REP, TWEET_DIVERSITY_SIM_THRESHOLD)
            mean_stance = float(np.mean(g["stance"].values)) if "stance" in g.columns else None

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

    # order windows
    try:
        clusters["window_dt"] = pd.to_datetime(clusters["window"])
        clusters = clusters.sort_values(["window_dt", "cluster_id"]).reset_index(drop=True)
        windows = clusters["window"].drop_duplicates().tolist()
        clusters = clusters.drop(columns=["window_dt"])
    except Exception:
        clusters = clusters.sort_values(["window", "cluster_id"]).reset_index(drop=True)
        windows = clusters["window"].drop_duplicates().tolist()

    # global lineage IDs
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

    # global PCA on centroids (2D)
    Xc = np.stack(clusters["centroid"].values)
    pca = PCA(n_components=2, random_state=0)
    XYc = pca.fit_transform(Xc)
    clusters["x"] = XYc[:, 0]
    clusters["y"] = XYc[:, 1]

    uniq = sorted(clusters["global_cluster_id"].unique().tolist())
    cmap = {gid: PALETTE[i % len(PALETTE)] for i, gid in enumerate(uniq)}
    clusters["color"] = clusters["global_cluster_id"].map(cmap)

    return clusters.reset_index(drop=True), windows, pca


def build_click_tweet_payload(
    clusters: pd.DataFrame,
    windows: List[str],
    pca: PCA,
) -> Dict[str, List[Dict]]:
    """
    Build a dict mapping key = "window|cluster_id" -> list of tweet points:
      {x, y, text, post_id, stance?}
    Uses global PCA components fitted on centroids to project tweet embeddings.
    """
    df_repr = pd.read_parquet(REP_PATH)

    # Build a robust window -> parquet path map (don’t assume filename matches window)
    window_to_file: Dict[str, Path] = {}
    for fp in sorted(EVAL_DIR.glob("*.parquet")):
        try:
            d = pd.read_parquet(fp, columns=["window", "post_id", "cluster_id"])
            d.columns = d.columns.str.strip()
            if "window" in d.columns and len(d) > 0:
                wname = str(d["window"].iloc[0])
            else:
                wname = fp.stem
        except Exception:
            wname = fp.stem
        window_to_file[wname] = fp

    df_repr.columns = df_repr.columns.str.strip()
    if "text" not in df_repr.columns and "tweet" in df_repr.columns:
        df_repr = df_repr.rename(columns={"tweet": "text"})
    df_repr["post_id"] = df_repr["post_id"].astype(str)
    df_repr["embedding"] = df_repr["embedding"].map(_as_np)
    df_repr = df_repr[["post_id", "embedding"]]

    payload: Dict[str, List[Dict]] = {}

    for w in windows:
        # Which clusters exist in this window?
        cids = clusters.loc[clusters["window"] == w, "cluster_id"].unique().tolist()
        if not cids:
            continue

        f = window_to_file.get(w)
        print(f"[payload] window={w} -> file={f.name}")

        if f is None:
            # If window values differ slightly, optionally try a fallback match:
            # e.g., match by stem contains w
            candidates = [p for p in EVAL_DIR.glob("*.parquet") if w in p.stem]
            if candidates:
                f = candidates[0]
            else:
                # Nothing found -> payload remains empty for this window
                continue

        df_eval = pd.read_parquet(f)

        df_eval.columns = df_eval.columns.str.strip()
        if "window" not in df_eval.columns:
            df_eval["window"] = w
        
        # normalize IDs aggressively on BOTH sides (critical)
        df_eval["post_id"] = (
            df_eval["post_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        )
        df_repr["post_id"] = (
            df_repr["post_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        )
        
        print(f"[payload] window={w} file={Path(f).name} eval_rows={len(df_eval)} cols={df_eval.columns.tolist()}")
        
        if "cluster_id" not in df_eval.columns:
            print(f"[payload] window={w} ERROR: no cluster_id column")
            continue
        
        print(f"[payload] window={w} cluster_id non-null={df_eval['cluster_id'].notna().sum()}")
        
        df_eval = df_eval.dropna(subset=["cluster_id"])
        print(f"[payload] window={w} after dropna(cluster_id) rows={len(df_eval)}")
        if len(df_eval) == 0:
            continue
        
        df_eval["cluster_id"] = df_eval["cluster_id"].astype(int)
        
        df = df_eval.merge(df_repr, on="post_id", how="left")
        print(f"[payload] window={w} after merge rows={len(df)} embedding_nonnull={df['embedding'].notna().sum() if 'embedding' in df.columns else 'NO embedding col'}")
        
        df = df.dropna(subset=["embedding"])
        print(f"[payload] window={w} after dropna(embedding) rows={len(df)}")
        if len(df) == 0:
            # This is the smoking gun case: post_id mismatch or df_repr lacks embeddings
            # Show a few ids to compare
            print("[payload] sample eval post_ids:", df_eval["post_id"].head(5).tolist())
            print("[payload] sample repr post_ids:", df_repr["post_id"].head(5).tolist())
            continue


        # For each cluster, sample tweets and project to PCA space
        for cid in cids:
            g = df[df["cluster_id"] == cid]
            if len(g) < MIN_CLUSTER_POSTS:
                continue

            # "All tweets" (still dedupe lexically, but do not cap to TWEETS_PER_CLUSTER)
            sample = sample_cluster_tweets(g, max_points=10**9, diversity_sim_threshold=1.01)

            if len(sample) == 0:
                payload[f"{w}|{cid}"] = []
                continue

            # --- Project tweets into the SAME displayed PCA space as the centroid ---
            # We anchor tweet points around the centroid's PCA location, so they appear
            # as a cloud around the cluster marker in the plot.
            
            # tweet embeddings
            embs = np.stack(sample["embedding"].values)
            
            # find this cluster's centroid in embedding space + its PCA coords
            crow = clusters.loc[
                (clusters["window"] == w) & (clusters["cluster_id"] == cid),
                ["centroid", "x", "y"]
            ]
            if len(crow) == 0:
                # should not happen if clusters/windows are consistent
                payload[f"{w}|{cid}"] = []
                continue
            
            centroid_emb = crow["centroid"].iloc[0]   # original embedding space (d,)
            centroid_x = float(crow["x"].iloc[0])    # displayed PCA x
            centroid_y = float(crow["y"].iloc[0])    # displayed PCA y
            
            # center tweets on their own cluster centroid (in embedding space)
            delta = embs - centroid_emb
            
            # project deltas using centroid-PCA components (no mean subtraction needed now)
            delta_2d = delta @ pca.components_.T     # (n, 2)
            
            # anchor the deltas at the centroid's displayed PCA coordinates
            X2 = np.column_stack([centroid_x + delta_2d[:, 0], centroid_y + delta_2d[:, 1]])


            points = []
            for i, row in enumerate(sample.itertuples(index=False)):
                txt = getattr(row, "text", "")
                txt = normalize_text(txt)
                txt = (txt[:320] + "…") if len(txt) > 320 else txt

                stance = getattr(row, "stance", None) if hasattr(row, "stance") else None

                points.append({
                    "x": float(X2[i, 0]),
                    "y": float(X2[i, 1]),
                    "text": txt,
                    "post_id": str(getattr(row, "post_id", "")),
                    "stance": None if stance is None else float(stance),
                })
            print(f"[payload] key={w}|{cid} points={len(points)}")
            payload[f"{w}|{cid}"] = points

    return payload


def build_html_with_click(
    clusters: pd.DataFrame,
    windows: List[str],
) -> go.Figure:
    # Prepare centroid hover and customdata for click lookup
    def centroid_hover(row) -> str:
        reps = row["representatives"] or []
        rep_lines = [f"• {normalize_text(r.get('text',''))[:220]}" for r in reps]
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
            f"<br><br><b>representatives</b><br>{rep_block}<br><br>"
            f"<i>Click to show tweets</i>"
        )

    # drift segments between consecutive windows (centroid-to-centroid by global id)
    drift_by_window: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for i in range(1, len(windows)):
        w_prev, w_curr = windows[i - 1], windows[i]
        prev = clusters[clusters["window"] == w_prev][["global_cluster_id", "x", "y"]]
        curr = clusters[clusters["window"] == w_curr][["global_cluster_id", "x", "y"]]
        m = curr.merge(prev, on="global_cluster_id", suffixes=("_curr", "_prev"))
        segs = [(r.x_prev, r.y_prev, r.x_curr, r.y_curr) for r in m.itertuples(index=False)]
        drift_by_window[w_curr] = segs

    # initial window
    w0 = windows[0]
    df0 = clusters[clusters["window"] == w0].copy()
    df0["hover"] = df0.apply(centroid_hover, axis=1)

    # customdata used by JS click handler
    # key format must match payload keys
    df0["key"] = df0.apply(lambda r: f"{r['window']}|{int(r['cluster_id'])}", axis=1)

    fig = go.Figure()


    # Trace 0: visible centroid points (hover)
    fig.add_trace(
        go.Scatter(
            x=df0["x"],
            y=df0["y"],
            mode="markers",
            marker=dict(
                size=np.clip(df0["num_posts"].values, 8, 60),
                color=df0["color"],
                opacity=0.9,
                line=dict(width=0.5, color="#111"),
            ),
            hovertext=df0["hover"],
            hoverinfo="text",
            customdata=df0["key"],
            name="Clusters",
        )
    )
    
    # Trace 1: invisible hitbox points (click)
    fig.add_trace(
        go.Scatter(
            x=df0["x"],
            y=df0["y"],
            mode="markers",
            marker=dict(
                size=np.clip(df0["num_posts"].values, 18, 90),  # BIGGER click target
                color=df0["color"],
                opacity=0.6,  # effectively invisible but still clickable
                line=dict(width=0),
            ),
            hoverinfo="skip",
            customdata=df0["key"],
            name="(hitbox)",
            showlegend=False,
        )
    )


    # Trace 2: drift lines
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

    # Trace 3: tweet overlay points (initially empty; updated on click)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=7, opacity=0.95, symbol="circle-open"),
            hovertext=[],
            hoverinfo="text",
            name="Tweets (clicked cluster)",
        )
    )

    # frames per window (centroids + drift). tweet trace stays as-is (JS updates it)
    frames = []
    for w in windows:
        dfw = clusters[clusters["window"] == w].copy()
        dfw["hover"] = dfw.apply(centroid_hover, axis=1)
        dfw["key"] = dfw.apply(lambda r: f"{r['window']}|{int(r['cluster_id'])}", axis=1)

        segs = drift_by_window.get(w, [])
        lx, ly = [], []
        for (x0, y0, x1, y1) in segs:
            lx += [x0, x1, None]
            ly += [y0, y1, None]
            
        #visible clusters
        frames.append(
            go.Frame(
                name=w,
                data=[
                    # Trace 0: visible centroids
                    go.Scatter(
                        x=dfw["x"],
                        y=dfw["y"],
                        marker=dict(
                            size=np.clip(dfw["num_posts"], 8, 60),
                            color=dfw["color"],
                        ),
                        hovertext=dfw["hover"],
                        customdata=dfw["key"],
                    ),
        
                    # Trace 1: invisible hitboxes
                    go.Scatter(
                        x=dfw["x"],
                        y=dfw["y"],
                        marker=dict(
                            size=np.clip(dfw["num_posts"], 18, 90),
                            color=dfw["color"],
                            opacity=0.51,
                        ),
                        customdata=dfw["key"],
                        hoverinfo="skip",
                    ),
        
                    # Trace 2: drift lines
                    go.Scatter(
                        x=lx,
                        y=ly,
                        mode="lines",
                        line=dict(width=1, color="#666"),
                    ),
        
                    # IMPORTANT: no trace 3 update here
                ],
            )
        )



    fig.frames = frames

    steps = [
        dict(
            method="animate",
            args=[[w], dict(mode="immediate", frame=dict(duration=1000, redraw=False), transition=dict(duration=400))],
            label=w,
        )
        for w in windows
    ]


    fig.update_layout(
        template="plotly_dark",
        title="Narrative Drift (click a cluster to reveal tweets in PCA space)",
        xaxis_title="PCA-1",
        yaxis_title="PCA-2",
        clickmode="event+select",
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
                      args=[windows, dict(
                          mode="immediate",
                          frame=dict(duration=1000, redraw=False),
                          transition=dict(duration=400),
                          fromcurrent=True
                      )],
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
    print("Building centroid clusters + global PCA...")
    clusters, windows, pca = compute_clusters_and_windows()

    print("Building tweet click payload...")
    tweet_payload = build_click_tweet_payload(clusters, windows, pca)

    # --------------------------------------------------
    # JS click handler template (DEFINE FIRST)
    # --------------------------------------------------
    payload_json = json.dumps(tweet_payload).replace("</", "<\\/")
    post_script = r"""
    console.log("post_script running...");
    
    var TWEETS = TWEETS_PAYLOAD_PLACEHOLDER;
    
    function getKeyFromClick(data) {
      if (!data || !data.points || data.points.length === 0) return null;
      var pt = data.points[0];
      if (pt.customdata) return pt.customdata;
      if (pt.data && pt.data.customdata && pt.pointIndex !== undefined) {
        return pt.data.customdata[pt.pointIndex];
      }
      return null;
    }
    
    function setTweetTrace(gd, points, titleKey) {
      var xs = [], ys = [], hovers = [];
      for (var i = 0; i < points.length; i++) {
        xs.push(points[i].x);
        ys.push(points[i].y);
        var txt = points[i].text || "";
        var pid = points[i].post_id || "";
        hovers.push("<b>post_id</b>: " + pid + "<br><br>" + txt);
      }
    
      // IMPORTANT: restyle trace 3 (tweet overlay)
      Plotly.restyle(gd, { x: [xs], y: [ys], hovertext: [hovers] }, [3]);
    
      Plotly.relayout(gd, {
        "title.text": "Narrative Drift (selected: " + titleKey + ")"
      });
    
      console.log("Tweet overlay updated:", titleKey, "n=", points.length);
    }
    
    function bindWhenReady() {
      var gd = document.getElementById("{plot_id}");
      if (!gd || typeof Plotly === "undefined") {
        setTimeout(bindWhenReady, 50);
        return;
      }
      if (gd.__tweetClickBound) return;
      gd.__tweetClickBound = true;
    
      console.log("Bound to plot:", gd.id);
    
      gd.on("plotly_click", function(data) {
        try {
          var key = getKeyFromClick(data);
          console.log("clicked key:", key, data && data.points && data.points[0] && data.points[0].curveNumber);
          if (!key) return;
    
          var points = TWEETS[key] || [];
          setTweetTrace(gd, points, key);
        } catch (e) {
          console.error("click handler error:", e);
        }
      });
    }
    
    bindWhenReady();
    """





    # --------------------------------------------------
    # Inject payload AFTER definition
    # --------------------------------------------------
    post_script = post_script.replace(
        "TWEETS_PAYLOAD_PLACEHOLDER",
        json.dumps(tweet_payload)
    )

    print("Building figure...")
    fig = build_html_with_click(clusters, windows)

    print("Writing HTML...")

    fig.write_html(
        OUT_HTML,
        include_plotlyjs=True,
        full_html=True,
        post_script=post_script,
    )



    print("Done.")



if __name__ == "__main__":
    main()
