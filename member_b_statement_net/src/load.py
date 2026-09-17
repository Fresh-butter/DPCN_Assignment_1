"""Load and clean the raw survey CSV for the statement (item) network.

I build a network whose NODES ARE THE 60 STATEMENTS, not the
respondents. We still need respondent rows to compute cross-respondent
correlation between statements, so cleaning happens at the respondent
level exactly as agreed in the shared conventions, and the
raw CSV is read fresh here rather than from data/clean/, since that
folder's script drops any row with a single missing answer (85 rows)
instead of the agreed >30-missing rule (87 rows).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- shared conventions ---
LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}
MISSING = {"No Comments", ""}  # -> NaN
DROP_ROWS_MISSING_MORE_THAN = 30  # leaves 87 respondents
SEED = 42

# --- paths ---
RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "Survey_Results_UC.csv"

# --- statement blocks ---
BLOCK_NAMES = {"T": "Technology", "E": "Education", "S": "Society/Ethics", "V": "Environment"}

# Items flagged as worth checking for reverse/contrarian phrasing
# before interpreting correlation signs. We do NOT flip them. I keep
# sign information (that is the point of a signed network), so a statement
# phrased against its block's grain should simply show up with negative
# edges to the rest of its block.
FLAGGED_ITEMS = ["E02", "T10"]


def _clean_column_name(raw_name: str) -> str:
    """'T01. Artificial Intelligence will ...' -> 'T01'"""
    return raw_name.split(".", 1)[0].strip()


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, keep the full item wording as a lookup, return coded df."""
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    id_col = df.columns[0]
    df = df.drop(columns=[id_col])
    return df


def item_wording() -> dict:
    """Map short item code ('T01') -> full statement text, for figure/table labels."""
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig", nrows=0)
    wording = {}
    for col in df.columns[1:]:
        code = _clean_column_name(col)
        text = col.split(".", 1)[1].strip() if "." in col else col
        wording[code] = text
    return wording


def encode_likert(df: pd.DataFrame) -> pd.DataFrame:
    """Map Likert text -> 1..5, 'No Comments'/blank -> NaN. Rename columns to item codes."""
    coded = df.copy()
    coded.columns = [_clean_column_name(c) for c in coded.columns]
    coded = coded.replace(list(MISSING), np.nan)
    coded = coded.replace(LIKERT)
    coded = coded.apply(pd.to_numeric, errors="coerce")
    return coded


def drop_near_empty_rows(coded: pd.DataFrame, max_missing: int = DROP_ROWS_MISSING_MORE_THAN) -> pd.DataFrame:
    missing_per_row = coded.isna().sum(axis=1)
    kept = coded.loc[missing_per_row <= max_missing].reset_index(drop=True)
    return kept


def zscore_items(coded: pd.DataFrame) -> pd.DataFrame:
    """Per-item (per-column) z-score, ignoring NaN. Missing stays NaN (not imputed)."""
    return (coded - coded.mean(axis=0, skipna=True)) / coded.std(axis=0, skipna=True, ddof=0)


def zscore_items_mean_imputed(coded: pd.DataFrame) -> pd.DataFrame:
    """Per-item z-score with missing values mean-imputed (-> 0), for the sensitivity check."""
    z = zscore_items(coded)
    return z.fillna(0.0)


def load_clean_responses() -> pd.DataFrame:
    """Full pipeline: raw -> coded -> drop near-empty rows -> 87 x 60 Likert matrix (1-5, NaN)."""
    raw = load_raw()
    coded = encode_likert(raw)
    clean = drop_near_empty_rows(coded)
    return clean


def item_blocks(columns) -> pd.Series:
    """Item code -> block letter (T/E/S/V), for community-vs-block comparison (NMI)."""
    return pd.Series({c: c[0] for c in columns})


if __name__ == "__main__":
    clean = load_clean_responses()
    print(f"Respondents kept: {clean.shape[0]} (expected 87)")
    print(f"Items: {clean.shape[1]} (expected 60)")
    print("Missing per item (top 5):")
    print(clean.isna().sum().sort_values(ascending=False).head())
