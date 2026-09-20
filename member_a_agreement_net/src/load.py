"""
load.py -- member_a_agreement_net

A thin wrapper over the shared data contract. All cleaning, encoding and
z-scoring logic lives in common/contract.py so that A, B and C are provably
describing the same 87 people; this module only re-exports it under the names
this member's code already uses, and keeps the self-test that
development_guidelines.md §7 requires.

Nothing member-specific happens at load time for the agreement network. What
IS member-specific -- pairwise-complete cosine with at least 20 shared items,
which is a modelling choice about leftover NaNs rather than cleaning -- lives
in build.py.
"""

from common.contract import (  # noqa: F401  (re-exported for this member's modules)
    LIKERT,
    MAX_MISSING,
    N_ITEMS,
    N_RESPONDENTS,
    REVERSE_CODE_FLAGGED,
    SEED,
    drop_high_missing,
    encode_likert,
    item_block,
    load_clean,
    load_raw,
    zscore_items,
    zscore_items_mean_imputed,
)

# Name this member's notebook and modules already use.
N_RESPONDENTS_EXPECTED = N_RESPONDENTS


if __name__ == "__main__":
    clean = load_clean()
    print(f"member A -- clean matrix: {clean.shape[0]} x {clean.shape[1]} "
          f"(expect {N_RESPONDENTS} x {N_ITEMS})")
    assert clean.shape == (N_RESPONDENTS, N_ITEMS)

    z = zscore_items(clean)
    print(f"z-scored: {z.shape}, NaN remaining: {z.isna().sum().sum()}")
    print(f"mean-imputed: any NaN? {zscore_items_mean_imputed(clean).isna().any().any()}")
