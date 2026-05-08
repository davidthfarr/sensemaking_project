import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scripts_environment_wrapper import environment

# Path to evaluated rolling windows
EVAL_DIR = Path(environment.EVALUATED_DIR())

rows = []

for p in sorted(EVAL_DIR.glob("*.parquet")):
    df = pd.read_parquet(p)

    rows.append({
        "window": p.stem,
        "num_clusters": df["cluster_id"].nunique(dropna=True),
        "num_posts": len(df),
        "noise_frac": df["cluster_id"].isna().mean(),
    })

cluster_df = pd.DataFrame(rows)

# ---- Plot ----
fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(
    cluster_df["window"],
    cluster_df["num_clusters"],
    marker="o",
    label="Number of clusters",
)
ax1.set_ylabel("Number of clusters")
ax1.set_xlabel("Window")
ax1.tick_params(axis="x", rotation=90)

ax2 = ax1.twinx()
ax2.plot(
    cluster_df["window"],
    cluster_df["noise_frac"],
    color="red",
    linestyle="--",
    alpha=0.6,
    label="Noise fraction",
)
ax2.set_ylabel("Fraction noise")

# Title & legend
fig.suptitle("Cluster Emergence Over Time")
fig.tight_layout()

ax = plt.gca()
num_ticks = len(ax.get_xticks())
if num_ticks >= 30:
    ax.set_xticks(ax.get_xticks()[::int(num_ticks / 20)])

plt.show()
plt.savefig('data/figures/ck-influentials-plus-top-level-replies-noise.png')