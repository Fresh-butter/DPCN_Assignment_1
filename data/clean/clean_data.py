"""Clean the raw survey CSV by dropping rows with any missing values."""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "raw" / "Survey_Results_UC.csv"
CLEAN_PATH = Path(__file__).resolve().parent / "cleaned_Survey_Results_UC.csv"


def clean(raw_path: Path = RAW_PATH, clean_path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    before = len(df)
    df = df.dropna(axis=0, how="any")
    after = len(df)
    df.to_csv(clean_path, index=False)
    print(f"Rows before: {before}, after: {after}, dropped: {before - after}")
    print(f"Cleaned CSV written to: {clean_path}")
    return df


if __name__ == "__main__":
    clean()
