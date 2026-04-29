"""
Noise fraction over time — one line per case.

A rising noise fraction can indicate narrative disruption or fragmentation;
a falling fraction suggests consolidation into coherent clusters.

Usage
-----
python analysis/noise_over_time.py --cases venezuela ukraine
python analysis/noise_over_time.py --cases venezuela ukraine --output noise.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(case: str) -> pd.DataFrame:
    path = Path("data/evaluated") / case / "results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No results for '{case}' at {path}")
    df = pd.read_parquet(path)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    return df


def noise_by_window(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("window_start")
        .apply(lambda g: g["is_noise"].sum() / len(g), include_groups=False)
        .rename("noise_fraction")
        .reset_index()
        .sort_values("window_start")
    )


def plot(cases: list[str], output: str | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    for case in cases:
        df = load_results(case)
        noise = noise_by_window(df)
        ax.plot(noise["window_start"], noise["noise_fraction"], marker="o", markersize=3,
                linewidth=1.5, label=case)

    ax.set_xlabel("Window start")
    ax.set_ylabel("Noise fraction")
    ax.set_title("Noise Fraction Over Time", fontsize=11, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    plot(args.cases, args.output)
