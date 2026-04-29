"""
Convert a raw case CSV to a clean processed parquet.

Usage
-----
python scripts/prepare_processed_data.py --case iran
python scripts/prepare_processed_data.py --case russia
python scripts/prepare_processed_data.py --case venezuela

Input:  data/<case>/*.csv       (first CSV found in the directory)
Output: data/processed/<case>/<case>_en_clean.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=["iran", "russia", "venezuela"],
                   help="Case name — resolves input/output paths automatically")
    return p.parse_args()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r", " ").replace("\n", " ").strip()


def main() -> None:
    args = parse_args()

    raw_dir = Path("data") / args.case
    out_path = Path("data/processed") / args.case / f"{args.case}_en_clean.parquet"

    # Find the first CSV in the raw directory
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV files found in {raw_dir}")
    raw_path = csvs[0]
    if len(csvs) > 1:
        print(f"Warning: multiple CSVs found in {raw_dir}, using {raw_path.name}")

    print(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Initial rows: {len(df):,}")

    # Keep English only
    if "language" in df.columns:
        df = df[df["language"] == "en"]
        print(f"After language filter (en): {len(df):,}")

    # Column name normalisation
    rename_map = {}
    if "id" not in df.columns:
        for candidate in ("post_id", "Resource Id"):
            if candidate in df.columns:
                rename_map[candidate] = "id"
                break
    if "user_id" not in df.columns:
        for candidate in ("author_id", "Author", "X Author ID"):
            if candidate in df.columns:
                rename_map[candidate] = "user_id"
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in ("id", "user_id", "timestamp", "text") if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns after normalisation: {missing}\n"
                 f"Available: {df.columns.tolist()}")

    # Parse timestamps if not already datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["text"] = df["text"].apply(clean_text)

    df = df.dropna(subset=["timestamp"])
    df = df[df["text"] != ""]
    print(f"After dropping invalid rows: {len(df):,}")

    df_out = df[["id", "user_id", "timestamp", "text"]].rename(columns={"id": "post_id"})
    df_out["post_id"] = df_out["post_id"].astype(str)
    df_out["user_id"] = df_out["user_id"].astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"Written → {out_path} ({len(df_out):,} rows)")
    print(df_out.head())


if __name__ == "__main__":
    main()
