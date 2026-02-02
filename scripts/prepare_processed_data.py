import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/ven_cleaned.csv")        # change if needed
OUT_PATH = Path("data/processed/venezuela/ven_en_clean.parquet")

def clean_text(text: str) -> str:
    """Minimal, safe text cleaning."""
    if not isinstance(text, str):
        return ""
    return (
        text.replace("\r", " ")
            .replace("\n", " ")
            .strip()
    )

def main():
    print(f"Loading raw data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)

    print(f"Initial rows: {len(df):,}")

    # Keep English only
    df = df[df["language"] == "en"]
    print(f"After language filter (en): {len(df):,}")

    # Parse timestamps
    '''df["timestamp"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )'''

    # Clean text
    #df["text"] = df["tweet"].apply(clean_text)

    # Drop invalid rows
    df = df.dropna(subset=["timestamp"])
    df = df[df["text"] != ""]

    print(f"After dropping invalid rows: {len(df):,}")
    # --- ID column compatibility (venezuela / brandwatch export) ---
    rename_map = {}

    # post id
    if "id" not in df.columns:
        if "post_id" in df.columns:
            rename_map["post_id"] = "id"
        elif "Resource Id" in df.columns:
            rename_map["Resource Id"] = "id"
    
    # user id
    if "user_id" not in df.columns:
        if "author_id" in df.columns:
            rename_map["author_id"] = "user_id"
        elif "Author" in df.columns:
            rename_map["Author"] = "user_id"
        elif "X Author ID" in df.columns:
            rename_map["X Author ID"] = "user_id"
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    missing = [c for c in ["id", "user_id"] if c not in df.columns]
    if missing:
        raise KeyError(f"Still missing required columns: {missing}. Available: {df.columns.tolist()}")

    # Select canonical processed columns
    
    df_processed = df[
        [
            "id",
            "user_id",
            "timestamp",
            "text",
            "language",
        ]
    ].rename(
        columns={
            "id": "post_id",
        }
    )

    # Ensure string IDs
    df_processed["post_id"] = df_processed["post_id"].astype(str)
    df_processed["user_id"] = df_processed["user_id"].astype(str)

    # Write to processed
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(OUT_PATH, index=False)

    print(f"Processed data written to {OUT_PATH}")
    print(df_processed.head())

if __name__ == "__main__":
    main()
