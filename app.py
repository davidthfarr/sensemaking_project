"""
Narrative Drift UI (Dash)
-------------------------
Ready-to-run Dash app that:
- loads rolling-window cluster assignments (data/evaluated/daily/*.parquet)
- joins to representations (data/processed/posts_repr.parquet)
- computes cluster centroids + deduplicated representative tweets
- assigns stable global lineage IDs (for consistent cluster-based coloring)
- projects centroids into a single global 2D space (PCA)
- renders an interactive UI with a time slider + click-to-read representatives

Run:
  pip install dash plotly pandas pyarrow scikit-learn numpy
  python app.py

Expected files:
  data/processed/posts_repr.parquet
    columns: post_id, timestamp, text, embedding, stance (stance optional)
  data/evaluated/daily/YYYY-MM-DD.parquet
    columns: post_id, window, cluster_id, is_noise
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go


# -------------------------
# Paths / Config
# -------------------------
REP_PATH = Path("data/processed/posts_repr.parquet")
EVAL_DIR = Path("data/evaluated/daily")

# UI / preprocessing knobs
TOP_K_REP = 5                 # representative tweets per cluster
MIN_CLUSTER_POSTS = 8         # drop clusters smaller than this in the UI
LINEAGE_SIM_THRESHOLD = 0.85  # cosine similarity threshold to link across windows
MAX_WINDOWS: Optional[int] = None  # set e.g. 60 for faster startup during dev

# Color palette (categorical)
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
    """Normalize for RT / trivial duplicates (fast + effective)."""
    if text is None:
        return ""
    t = text.strip()
    t = t.lower()
    t = re.sub(r"^rt\s+@\w+:\s*", "", t)  # strip RT prefix
    t = re.sub(r"http\S+", "", t)         # strip URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t


def pick_representatives(
    df_cluster: pd.DataFrame,
    top_k: int = 5,
    diversity_sim_threshold: float = 0.92,
) -> List[Dict]:
    """
    Pick representative tweets:
    - compute centroid
    - rank by similarity to centroid
    - lexically dedupe via normalized text
    - optionally enforce semantic diversity among selected reps
    Returns list of dicts: {post_id, text, stance(optional), sim_to_centroid}
    """
    if len(df_cluster) == 0:
        return []

    # Deduplicate retweets / exact copies
    dfc = df_cluster.copy()
    dfc["text_norm"] = dfc["text"].astype(str).map(normalize_text)
    dfc = dfc.drop_duplicates(subset=["text_norm"])

    if len(dfc) == 0:
        return []

    embs = np.stack(dfc["embedding"].values)
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(embs, centroid).ravel()
    dfc["sim_to_centroid"] = sims
    dfc = dfc.sort_values("sim_to_centroid", ascending=False)

    # Greedy semantic diversity filter
    selected = []
    selected_embs = []

    for _, row in dfc.iterrows():
        emb = row["embedding"]
        if selected_embs:
            sim_to_selected = cosine_similarity([emb], selected_embs).ravel()
            if np.max(sim_to_selected) >= diversity_sim_threshold:
                continue

        selected.append({
            "post_id": row["post_id"],
            "text": row["text"],
            "stance": row.get("stance", None),
            "sim_to_centroid": float(row["sim_to_centroid"]),
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
    """
    One-to-one greedy matching from curr clusters to prev clusters by cosine similarity.
    Returns mapping: curr_row_idx -> prev_row_idx (or None if no match above threshold).
    """
    if len(prev_centroids) == 0 or len(curr_centroids) == 0:
        return {i: None for i in range(len(curr_centroids))}

    P = np.stack(prev_centroids["centroid"].values)
    C = np.stack(curr_centroids["centroid"].values)

    S = cosine_similarity(C, P)  # (curr, prev)
    pairs: List[Tuple[float, int, int]] = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            pairs.append((float(S[i, j]), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    curr_used = set()
    prev_used = set()
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


def compute_cluster_summaries() -> pd.DataFrame:
    """
    Loads evaluated windows + representation layer and returns a DataFrame with one row per
    (window, cluster_id) containing centroid, size, stance stats, representatives, plus
    a stable global lineage ID and 2D coords.
    """
    if not REP_PATH.exists():
        raise FileNotFoundError(f"Missing representation file: {REP_PATH}")

    eval_files = sorted(EVAL_DIR.glob("*.parquet"))
    if not eval_files:
        raise FileNotFoundError(f"No evaluated parquet files found in: {EVAL_DIR}")

    if MAX_WINDOWS is not None:
        eval_files = eval_files[:MAX_WINDOWS]

    # Load representation layer
    df_repr = pd.read_parquet(REP_PATH)
    
    # --- Normalize column names (handles 'text ' and other whitespace) ---
    df_repr.columns = df_repr.columns.str.strip()
    
    # If your raw column is called "tweet" (common), standardize to "text"
    if "text" not in df_repr.columns and "tweet" in df_repr.columns:
        df_repr = df_repr.rename(columns={"tweet": "text"})


    required = {"post_id", "timestamp", "text", "embedding"}
    missing = required - set(df_repr.columns)
    if missing:
        raise ValueError(f"posts_repr.parquet missing columns: {sorted(missing)}")

    # Ensure embeddings are array-like
    # (No-op if already numpy arrays)
    def _as_np(x):
        if isinstance(x, np.ndarray):
            return x
        return np.array(x, dtype=np.float32)

    df_repr = df_repr.copy()
    df_repr["embedding"] = df_repr["embedding"].map(_as_np)

    has_stance = "stance" in df_repr.columns

    rows = []
    window_labels = []

    for f in eval_files:
        df_eval = pd.read_parquet(f)

        # Normalize schema
        if "window" not in df_eval.columns:
            # fallback if file name encodes window
            win = f.stem
            df_eval["window"] = win

        # Keep only non-noise clustered posts
        df_eval = df_eval.dropna(subset=["cluster_id"])
        if len(df_eval) == 0:
            continue

        df_eval["cluster_id"] = df_eval["cluster_id"].astype(int)

        df = df_eval.merge(df_repr, on="post_id", how="left")
        # --- Coalesce stance column if merge created stance_x/stance_y or stance got renamed ---
        df.columns = df.columns.str.strip()
        
        stance_candidates = ["stance", "stance_repr", "stance_x", "stance_y"]
        stance_found = [c for c in stance_candidates if c in df.columns]
        
        if stance_found:
            if "stance" not in df.columns:
                df = df.rename(columns={stance_found[0]: "stance"})
            if "stance_x" in df.columns and "stance_y" in df.columns:
                df["stance"] = df["stance_y"].fillna(df["stance_x"])
        else:
            # No stance available in this dataset; disable stance stats safely
            pass  
        
        # --- Coalesce text column if merge created text_x/text_y or text got renamed ---
        df.columns = df.columns.str.strip()
        
        text_candidates = ["text", "text_repr", "tweet", "text_x", "text_y", "tweet_x", "tweet_y"]
        
        found = [c for c in text_candidates if c in df.columns]
        if not found:
            # fail loudly with visibility
            raise ValueError(f"No text column found after merge. Columns are: {list(df.columns)}")
        
        # Prefer 'text' if present, else take the first candidate and rename it to 'text'
        if "text" not in df.columns:
            df = df.rename(columns={found[0]: "text"})
        
        # If both text_x and text_y exist, prefer the repr one (usually text_y) and fill missing
        if "text_x" in df.columns and "text_y" in df.columns:
            df["text"] = df["text_y"].fillna(df["text_x"])


        # Drop rows missing embeddings (should not happen; but protects UI)
        df = df.dropna(subset=["embedding", "text"])

        if len(df) == 0:
            continue

        window = str(df["window"].iloc[0])
        window_labels.append(window)

        # Compute per-cluster summary
        for cid, g in df.groupby("cluster_id"):
            n = len(g)
            if n < MIN_CLUSTER_POSTS:
                continue

            embs = np.stack(g["embedding"].values)
            centroid = embs.mean(axis=0)

            reps = pick_representatives(g, top_k=TOP_K_REP)

            mean_stance = float(np.mean(g["stance"].values)) if has_stance else None
            stance_std = float(np.std(g["stance"].values)) if has_stance else None

            rows.append({
                "window": window,
                "cluster_id": int(cid),
                "num_posts": int(n),
                "centroid": centroid,
                "mean_stance": mean_stance,
                "stance_std": stance_std,
                "representatives": reps,  # list[dict]
            })

    if not rows:
        raise RuntimeError("No clusters found after filtering. Try lowering MIN_CLUSTER_POSTS or check evaluated files.")

    clusters = pd.DataFrame(rows)

    # Sort windows chronologically if parseable, else lexicographically
    def _try_dt(s):
        try:
            return pd.to_datetime(s)
        except Exception:
            return None

    parsed = clusters["window"].map(_try_dt)
    if parsed.notna().all():
        clusters["window_dt"] = parsed
        clusters = clusters.sort_values(["window_dt", "cluster_id"]).reset_index(drop=True)
        ordered_windows = clusters["window"].drop_duplicates().tolist()
    else:
        clusters = clusters.sort_values(["window", "cluster_id"]).reset_index(drop=True)
        ordered_windows = clusters["window"].drop_duplicates().tolist()

    # Assign stable global lineage IDs
    global_id_counter = 0
    clusters["global_cluster_id"] = -1

    prev_df = pd.DataFrame(columns=clusters.columns)

    for w in ordered_windows:
        curr = clusters[clusters["window"] == w].copy().reset_index()
        if len(prev_df) == 0:
            # first window: new ids
            for i in range(len(curr)):
                clusters.loc[curr.loc[i, "index"], "global_cluster_id"] = global_id_counter
                global_id_counter += 1
        else:
            prev = prev_df.copy().reset_index(drop=True)
            match = greedy_window_matching(prev, curr, threshold=LINEAGE_SIM_THRESHOLD)

            # apply matches or create new
            for i in range(len(curr)):
                row_idx = curr.loc[i, "index"]
                j = match.get(i, None)
                if j is None:
                    clusters.loc[row_idx, "global_cluster_id"] = global_id_counter
                    global_id_counter += 1
                else:
                    clusters.loc[row_idx, "global_cluster_id"] = int(prev.loc[j, "global_cluster_id"])

        prev_df = clusters[clusters["window"] == w].copy()

    # Compute global 2D projection using PCA on centroids
    X = np.stack(clusters["centroid"].values)
    pca = PCA(n_components=2, random_state=0)
    XY = pca.fit_transform(X)
    clusters["x"] = XY[:, 0]
    clusters["y"] = XY[:, 1]

    # Assign stable color per global_cluster_id
    uniq = clusters["global_cluster_id"].unique().tolist()
    color_map = {gid: PALETTE[i % len(PALETTE)] for i, gid in enumerate(sorted(uniq))}
    clusters["color"] = clusters["global_cluster_id"].map(color_map)

    # Clean up
    if "window_dt" in clusters.columns:
        clusters = clusters.drop(columns=["window_dt"])

    return clusters.reset_index(drop=True)


# -------------------------
# Build data once at startup
# -------------------------
print("Loading + summarizing clusters (this may take a bit the first time)...")
CLUSTERS = compute_cluster_summaries()
WINDOWS = CLUSTERS["window"].drop_duplicates().tolist()

print(f"Loaded {len(CLUSTERS):,} cluster snapshots across {len(WINDOWS)} windows.")
print("Starting app...")


# -------------------------
# Dash App
# -------------------------
app = Dash(__name__)
app.title = "Narrative Drift UI"

app.layout = html.Div(
    style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "14px", "padding": "14px"},
    children=[
        html.Div(
            children=[
                html.H2("Narrative Drift (Cluster Centroids)", style={"margin": "0 0 8px 0"}),
                html.Div(
                    style={"display": "flex", "gap": "12px", "alignItems": "center", "marginBottom": "10px"},
                    children=[
                        html.Div("Window:", style={"minWidth": "60px"}),
                        dcc.Slider(
                            id="window_slider",
                            min=0,
                            max=max(0, len(WINDOWS) - 1),
                            step=1,
                            value=0,
                            marks={i: WINDOWS[i] for i in range(0, len(WINDOWS), max(1, len(WINDOWS)//6))},
                            tooltip={"always_visible": False, "placement": "bottom"},
                        ),
                        dcc.Dropdown(
                            id="window_dropdown",
                            options=[{"label": w, "value": w} for w in WINDOWS],
                            value=WINDOWS[0],
                            clearable=False,
                            style={"width": "260px"},
                        ),
                        dcc.Checklist(
                            id="show_lines",
                            options=[{"label": "Show drift lines", "value": "yes"}],
                            value=["yes"],
                            style={"marginLeft": "8px"},
                        ),
                    ],
                ),
                dcc.Graph(
                    id="drift_graph",
                    style={"height": "78vh"},
                    config={"displayModeBar": True},
                ),
                html.Div(
                    id="footer_stats",
                    style={"fontFamily": "monospace", "marginTop": "6px", "opacity": 0.85},
                ),
            ]
        ),
        html.Div(
            style={"borderLeft": "1px solid #333", "paddingLeft": "14px"},
            children=[
                html.H3("Cluster Details", style={"marginTop": 0}),
                html.Div(
                    id="cluster_details",
                    children=[
                        html.Div("Click a point to view representative tweets.", style={"opacity": 0.85})
                    ]
                ),
                html.Hr(style={"margin": "14px 0"}),
                html.H4("Tips", style={"margin": "0 0 8px 0"}),
                html.Ul(
                    children=[
                        html.Li("Color is driven by a global lineage ID (stable across windows)."),
                        html.Li("Size ~ number of posts in the cluster."),
                        html.Li("Representatives are deduplicated to avoid RT spam."),
                    ],
                    style={"opacity": 0.9},
                ),
            ],
        ),
    ],
)


def make_figure(selected_window: str, show_lines: bool) -> Tuple[go.Figure, str]:
    dfw = CLUSTERS[CLUSTERS["window"] == selected_window].copy()

    # Scatter points
    fig = go.Figure()

    fig.add_trace(
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
            customdata=np.stack(
                [
                    dfw["window"].values,
                    dfw["cluster_id"].values,
                    dfw["global_cluster_id"].values,
                    dfw["num_posts"].values,
                    dfw.get("mean_stance", pd.Series([None]*len(dfw))).values,
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>Window</b>: %{customdata[0]}<br>"
                "<b>Cluster</b>: %{customdata[1]}<br>"
                "<b>Global ID</b>: %{customdata[2]}<br>"
                "<b>Posts</b>: %{customdata[3]}<br>"
                "<b>Mean stance</b>: %{customdata[4]}<br>"
                "<extra></extra>"
            ),
            name="Clusters",
        )
    )

    # Drift lines: connect same global_cluster_id to next window if present
    if show_lines and len(WINDOWS) > 1:
        wi = WINDOWS.index(selected_window)
        if wi > 0:
            prev_w = WINDOWS[wi - 1]
            prev = CLUSTERS[CLUSTERS["window"] == prev_w][["global_cluster_id", "x", "y"]].copy()
            curr = dfw[["global_cluster_id", "x", "y"]].copy()
            merged = curr.merge(prev, on="global_cluster_id", suffixes=("_curr", "_prev"))

            # add line segments
            for _, r in merged.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[r["x_prev"], r["x_curr"]],
                        y=[r["y_prev"], r["y_curr"]],
                        mode="lines",
                        line=dict(width=1, color="#666"),
                        opacity=0.35,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="PCA-1",
        yaxis_title="PCA-2",
        template="plotly_dark",
        legend=dict(orientation="h"),
    )

    stats = f"window={selected_window} | clusters={len(dfw)} | posts={int(dfw['num_posts'].sum())} | unique_global_ids={dfw['global_cluster_id'].nunique()}"

    return fig, stats


@app.callback(
    Output("window_dropdown", "value"),
    Input("window_slider", "value"),
)
def slider_to_dropdown(idx):
    return WINDOWS[idx]


@app.callback(
    Output("window_slider", "value"),
    Input("window_dropdown", "value"),
)
def dropdown_to_slider(win):
    return WINDOWS.index(win)


@app.callback(
    Output("drift_graph", "figure"),
    Output("footer_stats", "children"),
    Input("window_dropdown", "value"),
    Input("show_lines", "value"),
)
def update_graph(win, show_lines_value):
    show = "yes" in (show_lines_value or [])
    fig, stats = make_figure(win, show_lines=show)
    return fig, stats


@app.callback(
    Output("cluster_details", "children"),
    Input("drift_graph", "clickData"),
    State("window_dropdown", "value"),
)
def show_cluster_details(clickData, current_window):
    if not clickData or "points" not in clickData or not clickData["points"]:
        return html.Div("Click a point to view representative tweets.", style={"opacity": 0.85})

    cd = clickData["points"][0]["customdata"]
    # customdata = [window, cluster_id, global_id, num_posts, mean_stance]
    win = str(cd[0])
    cid = int(cd[1])
    gid = int(cd[2])
    n = int(cd[3])
    mean_stance = cd[4]

    row = CLUSTERS[(CLUSTERS["window"] == win) & (CLUSTERS["cluster_id"] == cid)]
    if row.empty:
        return html.Div("Cluster not found (unexpected).", style={"color": "#ff6b6b"})

    reps = row.iloc[0]["representatives"] or []

    rep_blocks = []
    for i, r in enumerate(reps, start=1):
        stance_txt = ""
        if r.get("stance", None) is not None:
            stance_txt = f" | stance={r['stance']:+}"
        rep_blocks.append(
            html.Div(
                style={"marginBottom": "10px", "padding": "8px", "border": "1px solid #333", "borderRadius": "8px"},
                children=[
                    html.Div(f"#{i}  sim={r['sim_to_centroid']:.3f}{stance_txt}", style={"fontFamily": "monospace", "opacity": 0.85}),
                    html.Div(r["text"], style={"marginTop": "6px", "whiteSpace": "pre-wrap"}),
                ],
            )
        )

    header = html.Div(
        style={"marginBottom": "10px"},
        children=[
            html.Div(f"Window: {win}", style={"fontFamily": "monospace"}),
            html.Div(f"Cluster ID: {cid}  |  Global ID: {gid}", style={"fontFamily": "monospace"}),
            html.Div(f"Posts: {n}", style={"fontFamily": "monospace"}),
            html.Div(f"Mean stance: {mean_stance}", style={"fontFamily": "monospace"}) if mean_stance is not None else html.Div(),
        ],
    )

    return [header] + rep_blocks


if __name__ == "__main__":
    # Note: set debug=False for large datasets to reduce overhead
    app.run(host="0.0.0.0", port=8050, debug=False, use_reloader=False)
