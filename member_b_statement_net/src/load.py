"""
load.py -- member_b_statement_net

A thin wrapper over the shared data contract. All cleaning, encoding and
z-scoring logic lives in common/contract.py; this module only adapts it to
the one convention that is genuinely specific to the statement network.

That convention is the row index. The nodes of this network are the 60
STATEMENTS, not the respondents -- respondent rows exist here only as the
sample over which two statements are correlated. So respondent identity is
never used, and this module returns a positional index rather than the
respondent_id index that members A and C need for their node labels. The
matrix values are identical either way; only the labels differ.

What is member-specific and deliberately NOT shared: pairwise-complete
Pearson correlation with min_periods=10 (a modelling choice about leftover
NaNs, not cleaning), which lives in build.py.
"""

import pandas as pd

from common.contract import (  # noqa: F401  (re-exported for this member's modules)
    LIKERT,
    MAX_MISSING,
    N_ITEMS,
    N_RESPONDENTS,
    SEED,
    item_block,
    item_blocks,
    item_wording,
    zscore_items,
    zscore_items_mean_imputed,
)
from common.contract import REVERSE_CODE_FLAGGED as FLAGGED_ITEMS  # noqa: F401
from common.contract import load_clean as _load_clean_by_id


def load_clean() -> pd.DataFrame:
    """
    The shared 87 x 60 matrix, re-indexed 0..86.

    Identical values to common.contract.load_clean(); only the row labels
    differ. See the module docstring for why.
    """
    return _load_clean_by_id().reset_index(drop=True)


# Name this member's modules used before the contract was factored out.
load_clean_responses = load_clean


if __name__ == "__main__":
    clean = load_clean()
    print(f"member B -- clean matrix: {clean.shape[0]} x {clean.shape[1]} "
          f"(expect {N_RESPONDENTS} x {N_ITEMS})")
    assert clean.shape == (N_RESPONDENTS, N_ITEMS)
    assert list(clean.index) == list(range(N_RESPONDENTS)), "expected a positional index"

    print("missing per item (top 5):")
    print(clean.isna().sum().sort_values(ascending=False).head())
    print(f"flagged, not flipped: {list(FLAGGED_ITEMS)}")
