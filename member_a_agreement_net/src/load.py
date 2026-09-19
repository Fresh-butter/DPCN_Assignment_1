"""
load.py -- member_a_agreement_net

DEVIATION FROM CLAUDE.md, BY EXPLICIT USER INSTRUCTION:
This module reads data/clean/cleaned_Survey_Results_UC.csv (the pipeline in
data/clean/clean_data.py, which drops any row containing a missing answer)
instead of reading data/raw/Survey_Results_UC.csv fresh and applying the
>30-missing-per-row rule described in CLAUDE.md. The user explicitly asked,
twice, for member_a to use the cleaned CSV. This means:
  - N respondents = 85 (not the 87 used by the >30-missing rule).
  - clean_data.py only dropped rows with truly blank cells; it did NOT
    treat the text value "No Comments" as missing, so "No Comments" cells
    still appear in this file. This module still maps "No Comments" (and
    any other unmapped text) to NaN, per CLAUDE.md rule 3 (never encode
    "No Comments" as Neutral/3) -- so there ARE missing values here, just
    fewer than in the raw file, and similarity/centrality code below still
    has to handle NaN (pairwise-complete / mean-imputation).
member_b_statement_net uses the CLAUDE.md-compliant 87-respondent path, so
this is a genuine, documented divergence between members -- see FINDINGS.md.

Everything else about the shared conventions still applies: Likert text is
mapped to 1..5, and per-item z-scoring is still required before computing
similarity (the ceiling-effect problem is a property of the response
distribution, not of which cleaning rule produced the row set).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants block (per CLAUDE.md: no magic numbers scattered in notebooks)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CLEANED_CSV_PATH = REPO_ROOT / "data" / "clean" / "cleaned_Survey_Results_UC.csv"

LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}
SEED = 42
N_ITEMS_TOTAL = 60  # 60 Likert statements
N_RESPONDENTS_EXPECTED = 85  # cleaned-CSV respondent count (not 87)
BLOCK_LETTERS = {"T", "E", "S", "V"}  # Technology, Education, Society, Environment

# Items flagged in CLAUDE.md as needing a reverse-coding check before
# interpreting correlation signs. We do NOT flip them automatically here --
# only note the concern (see notebook / FINDINGS.md discussion).
REVERSE_CODE_CHECK_ITEMS = ["E02", "T10"]


def _item_code(col_name: str) -> str:
    """Extract the short item code, e.g. 'T01', from a full column header."""
    return col_name.split(".")[0].strip()


def load_cleaned_raw() -> pd.DataFrame:
    """Read the already-cleaned CSV (85 respondents x 60 items + id col)."""
    df = pd.read_csv(CLEANED_CSV_PATH)
    return df


def encode_likert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 60 Likert-item columns to numeric 1..5. The cleaned CSV has
    no missing/`No Comments` cells left, but we still map defensively
    (unmapped text -> NaN) rather than assume.
    """
    id_col = df.columns[0]
    item_cols = list(df.columns[1:])
    assert len(item_cols) == N_ITEMS_TOTAL, f"expected 60 items, got {len(item_cols)}"

    out = pd.DataFrame(index=df.index)
    out["respondent_id"] = df[id_col].astype(str)

    for col in item_cols:
        code = _item_code(col)
        out[code] = df[col].astype(str).str.strip().map(LIKERT).astype(float)

    out = out.set_index("respondent_id")
    return out


def load_clean() -> pd.DataFrame:
    """
    Full pipeline: cleaned CSV -> Likert-encoded (1..5). Returns the
    respondents x items dataframe (expected shape: 85 x 60).
    """
    raw = load_cleaned_raw()
    coded = encode_likert(raw)
    return coded


def zscore_items(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Per-item z-score: (x - item_mean) / item_sd. This is essential: raw 1-5
    Likert vectors are almost a clique in cosine-similarity space (mean
    cosine ~0.964) because of the ceiling effect (~79% Agree/Strongly Agree
    answers). z-scoring per item removes the shared "everyone agrees"
    component and spreads similarity out.
    """
    mu = clean.mean(axis=0, skipna=True)
    sd = clean.std(axis=0, skipna=True, ddof=0)
    z = (clean - mu) / sd
    return z


def zscore_items_mean_imputed(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Same as zscore_items, but any remaining NaN is mean-imputed (-> 0),
    the least-assumption fill for algorithms that cannot tolerate NaNs
    (e.g. cosine_similarity on a dense matrix, eigenvector centrality).
    On the cleaned CSV this is a no-op in practice since there are no
    missing values; kept for interface parity / robustness.
    """
    z = zscore_items(clean)
    return z.fillna(0.0)


if __name__ == "__main__":
    raw = load_cleaned_raw()
    print(f"cleaned csv shape (incl. id col): {raw.shape}")

    coded = encode_likert(raw)
    print(f"coded shape (expect {N_RESPONDENTS_EXPECTED} x {N_ITEMS_TOTAL}): {coded.shape}")
    print(f"any NaN in coded data: {coded.isna().any().any()}")
    assert coded.shape == (N_RESPONDENTS_EXPECTED, N_ITEMS_TOTAL), "unexpected shape -- check cleaned CSV"

    z = zscore_items(coded)
    zi = zscore_items_mean_imputed(coded)
    print(f"zscore shape: {z.shape}, any NaN remaining: {z.isna().any().any()}")
    print(f"mean-imputed zscore shape: {zi.shape}, any NaN remaining: {zi.isna().any().any()}")

    print("\nReverse-coding check items (read phrasing manually):")
    for code in REVERSE_CODE_CHECK_ITEMS:
        print(f"  {code}: mean={coded[code].mean():.2f}, sd={coded[code].std():.2f}")
