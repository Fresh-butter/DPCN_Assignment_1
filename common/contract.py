"""
common/contract.py -- the data contract. One implementation, three members.

Written once here and imported by every member's src/load.py, because a
cleaning rule that lives in three places will eventually say three different
things (development_guidelines.md §1, §4).

The contract, step by step:

    1. Encode   Strongly Disagree=1 ... Strongly Agree=5        -> numeric 1-5
    2. Missing  "No Comments", blank, unrecognised text -> NaN  (never 3)
    3. Drop     rows with > 30 of 60 missing                    -> 96 -> 87
    4. Matrix   respondents x items, NaN preserved              -> 87 x 60
    5. Z-score  per item column, skipna, computed on the full 87

Three things worth stating in the report:

  * "No Comments" is not Neutral. Encoding it as 3 fabricates data at the
    centre of the scale and biases every correlation toward zero.

  * The >30 cutoff is read off the data, not chosen. The missing-per-row
    distribution has a clean natural break: 68 rows have 0 missing, another
    17 have 1-4, one each has 10 and 15, and then it jumps straight to 45
    (x4) and 60 (x5). Nothing falls between 15 and 45, so any threshold in
    that gap gives the same 87, and the 9 rows dropped are abandoned
    responses rather than partial ones.

  * Z-scoring is mandatory, not stylistic. Raw responses are ~80%
    Agree/Strongly Agree, so raw cosine similarity sits near 0.96 for every
    pair of students and no network structure survives. Z-scoring is done on
    the full 87 so that all three members' z-values share one reference frame.

What each member does with the *remaining* NaNs is a modelling choice and
deliberately NOT standardised here (§2): A uses pairwise-complete cosine with
>=20 shared items, B uses pairwise-complete Pearson with min_periods=10, C
simply forms no edge for a NaN cell.

FROZEN. Once all three load.py self-tests pass, a change here silently
invalidates every figure and table in the repo. If you need different
behaviour, add a parameter defaulting to the current behaviour -- do not
change what an existing call returns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = REPO_ROOT / "Survey_Results_UC.csv"

LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}

SEED = 42
N_ITEMS = 60           # 4 blocks x 15 statements
MAX_MISSING = 30       # drop a row with more than this many missing items
N_RESPONDENTS = 87     # what remains after the drop

BLOCK_LETTERS = ("T", "E", "S", "V")

# Items whose wording may run against their block's grain, so a negative
# correlation with the rest of the block would be phrasing rather than
# disagreement. We FLAG these and never flip them: reverse-coding is a
# decision the team takes together and applies here for everyone at once
# (§2). Member B's network keeps edge signs precisely so that a
# contrarian-phrased item shows up as negative edges instead of being
# silently corrected away.
REVERSE_CODE_FLAGGED = ("E02", "T10")


# ---------------------------------------------------------------------------
# Loading and encoding
# ---------------------------------------------------------------------------

def _item_code(col_name: str) -> str:
    """'T01. Artificial Intelligence will ...' -> 'T01'."""
    return col_name.split(".")[0].strip()


def load_raw() -> pd.DataFrame:
    """Read the raw CSV as-is: 96 rows x 61 columns (1 id + 60 items)."""
    return pd.read_csv(RAW_CSV_PATH, encoding="utf-8-sig")


def item_wording() -> dict:
    """Map item code ('T01') -> full statement text, for figure/table labels."""
    header = pd.read_csv(RAW_CSV_PATH, encoding="utf-8-sig", nrows=0)
    return {
        _item_code(col): (col.split(".", 1)[1].strip() if "." in col else col)
        for col in header.columns[1:]
    }


def encode_likert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 60 Likert columns to numeric 1-5, indexed by respondent id.

    Anything not in LIKERT -- including "No Comments" and blanks -- becomes
    NaN. Never 3.
    """
    id_col = df.columns[0]
    item_cols = list(df.columns[1:])
    assert len(item_cols) == N_ITEMS, f"expected {N_ITEMS} items, got {len(item_cols)}"

    out = pd.DataFrame(index=df.index)
    out["respondent_id"] = df[id_col].astype(str)
    for col in item_cols:
        out[_item_code(col)] = df[col].astype(str).str.strip().map(LIKERT).astype(float)
    return out.set_index("respondent_id")


def drop_high_missing(coded: pd.DataFrame, max_missing: int = MAX_MISSING) -> pd.DataFrame:
    """Drop rows with more than `max_missing` NaN out of 60 items."""
    return coded[coded.isna().sum(axis=1) <= max_missing].copy()


def load_clean() -> pd.DataFrame:
    """
    The whole contract: raw CSV -> Likert-encoded -> >30-missing rows dropped.

    Returns an 87 x 60 respondents-by-items frame indexed by respondent_id
    (string), with NaN preserved.
    """
    return drop_high_missing(encode_likert(load_raw()))


# ---------------------------------------------------------------------------
# Z-scoring
# ---------------------------------------------------------------------------

def zscore_items(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Per-item z-score: (x - item_mean) / item_sd, NaN-skipping, ddof=0.

    ddof=0 is part of the contract, not a detail: switching to ddof=1 shifts
    every downstream similarity, correlation and deviation slightly. Items
    with zero variance are guarded so they yield NaN rather than inf.
    """
    mu = clean.mean(axis=0, skipna=True)
    sd = clean.std(axis=0, skipna=True, ddof=0).replace(0, np.nan)
    return (clean - mu) / sd


def zscore_items_mean_imputed(clean: pd.DataFrame) -> pd.DataFrame:
    """
    zscore_items with the remaining NaNs filled to 0 (the item mean).

    This is the least-assumption fill for algorithms that cannot tolerate NaN
    (dense cosine, eigenvector centrality). Every member reports one result
    computed this way as a sensitivity check against their primary
    pairwise-complete choice.
    """
    return zscore_items(clean).fillna(0.0)


# ---------------------------------------------------------------------------
# Item metadata
# ---------------------------------------------------------------------------

def item_block(code: str) -> str:
    """Item code -> block letter. 'T01' -> 'T'. Unknown prefixes -> '?'."""
    return code[0] if code[:1] in BLOCK_LETTERS else "?"


def item_blocks(codes) -> pd.Series:
    """Vectorised item_block over an iterable of codes, as a Series."""
    return pd.Series({c: item_block(c) for c in codes})


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = load_raw()
    print(f"raw csv shape (incl. id col): {raw.shape}")

    coded = encode_likert(raw)
    print(f"after likert encoding:        {coded.shape}  (NaN: {coded.isna().sum().sum()})")

    clean = load_clean()
    print(f"after >30-missing drop:       {clean.shape[0]} x {clean.shape[1]}  "
          f"(expect {N_RESPONDENTS} x {N_ITEMS})")
    assert clean.shape == (N_RESPONDENTS, N_ITEMS)

    z = zscore_items(clean)
    zi = zscore_items_mean_imputed(clean)
    print(f"z-score:                      {z.shape}, NaN remaining: {z.isna().sum().sum()}")
    print(f"mean-imputed z-score:         any NaN? {zi.isna().any().any()}")
    print(f"column means after z-score:   [{z.mean().min():+.4f}, {z.mean().max():+.4f}] (~0)")

    blocks = item_blocks(clean.columns).value_counts().reindex(list(BLOCK_LETTERS))
    print(f"items per block:              {blocks.to_dict()}")

    print("\nreverse-code flagged (not flipped):")
    for code in REVERSE_CODE_FLAGGED:
        print(f"  {code}: mean={clean[code].mean():.2f}, sd={clean[code].std():.2f}")
