"""
load.py -- member_c_bipartite_net

A thin wrapper over the shared data contract. All cleaning, encoding and
z-scoring logic lives in common/contract.py, so the 87 student nodes in this
bipartite graph are provably the same 87 students as in member A's agreement
network -- which is what makes the cross-check between A's similarity and
C's mean first-passage time a valid comparison at all.

What is member-specific and deliberately NOT shared: a NaN cell simply forms
no edge (a modelling choice about leftover missingness, not cleaning), which
lives in build.py.
"""

from common.contract import (  # noqa: F401  (re-exported for this member's modules)
    LIKERT,
    MAX_MISSING,
    N_ITEMS,
    N_RESPONDENTS,
    SEED,
    item_block,
    item_blocks,
    load_clean,
    load_raw,
    zscore_items,
    zscore_items_mean_imputed,
)
from common.viz import BLOCK_NAMES

# Name this member's modules used before the contract was factored out.
zscore_mean_imputed = zscore_items_mean_imputed

N_ITEMS_TOTAL = N_ITEMS
N_RESPONDENTS_EXPECTED = N_RESPONDENTS


def item_block_name(code: str) -> str:
    """
    Item code -> full block name, e.g. 'S01' -> 'Society/Ethics'.

    The names themselves come from common.viz.BLOCK_NAMES and are defined
    nowhere else, so a table label here cannot drift away from a figure
    legend in another member's notebook.
    """
    return BLOCK_NAMES.get(item_block(code), "Unknown")


if __name__ == "__main__":
    clean = load_clean()
    print(f"member C -- clean matrix: {clean.shape[0]} x {clean.shape[1]} "
          f"(expect {N_RESPONDENTS} x {N_ITEMS})")
    assert clean.shape == (N_RESPONDENTS, N_ITEMS)

    z = zscore_items(clean)
    print(f"z-scored: {z.shape}, NaN remaining: {z.isna().sum().sum()}")
    print(f"deviating cells (|z| > 1.0): {(z.abs() > 1.0).sum().sum()}")
    print(f"block names: {sorted(set(item_block_name(c) for c in clean.columns))}")
