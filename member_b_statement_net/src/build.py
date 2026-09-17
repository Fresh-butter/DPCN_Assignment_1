"""Build the signed statement-statement network.

Nodes = 60 statements. Edge weight(i, j) = correlation of statement i and j
across the 87 respondents, sign retained. Two arbitrary choices are made
explicit and both are run:

1. Missing-value handling: pairwise-complete correlation (primary) vs.
   correlation on mean-imputed z-scores (sensitivity check).
2. Edge selection: a permutation-test significance threshold (primary,
   "keep edges that beat shuffled data") vs. a fixed |r| threshold
   (sensitivity check).
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from load import SEED, item_blocks, zscore_items, zscore_items_mean_imputed

# --- constants (no magic numbers inline) ---
N_PERMUTATIONS = 1000
ALPHA = 0.01  # per-edge significance level for the permutation test
FDR_Q = 0.05  # Benjamini-Hochberg FDR level used alongside the raw alpha
FIXED_R_THRESHOLD = 0.30  # sensitivity-check alternative to the permutation test


def pairwise_complete_corr(coded: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between items, pairwise-complete on missing values.

    Pearson correlation is invariant to per-column z-scoring, so this is
    identical whether computed on raw 1-5 codes or on zscore_items(coded);
    we still route through the shared z-score convention for clarity.
    """
    z = zscore_items(coded)
    return z.corr(method="pearson", min_periods=10)


def mean_imputed_corr(coded: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation after mean-imputing missing values on z-scored items."""
    z_imputed = zscore_items_mean_imputed(coded)
    return z_imputed.corr(method="pearson")


def permutation_null_pool(
    coded: pd.DataFrame,
    n_perm: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> np.ndarray:
    """Pool of |r| values from independently permuting each item's answers
    (destroys real cross-item covariance, preserves each item's marginal
    distribution and missingness pattern). Pooling across all pairs and
    permutations (rather than taking the per-permutation max) gives the
    null distribution of a single random pair's |r| under independence,
    used below for a per-edge empirical p-value + FDR correction; a
    single family-wise max-statistic threshold at alpha=0.01 turned out to
    be so conservative (|r| > 0.50) that it kept only positive edges and
    erased the signed structure the whole network is built to show.
    """
    rng = np.random.default_rng(seed)
    z = zscore_items(coded)
    n_items = z.shape[1]
    iu = np.triu_indices(n_items, k=1)
    pooled = np.empty((n_perm, len(iu[0])))
    cols = z.columns
    for p in range(n_perm):
        shuffled = z.copy()
        for c in cols:
            col = shuffled[c].to_numpy()
            valid = ~np.isnan(col)
            vals = col[valid]
            rng.shuffle(vals)
            col = col.copy()
            col[valid] = vals
            shuffled[c] = col
        corr = shuffled.corr(method="pearson", min_periods=10).to_numpy()
        pooled[p] = np.abs(corr[iu])
    return pooled.ravel()


def fdr_bh_threshold_mask(pvals: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg FDR: boolean mask of pvals significant at level q."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh_line = (np.arange(1, n + 1) / n) * q
    below = ranked <= bh_line
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_i = np.max(np.where(below)[0])
    mask = np.zeros(n, dtype=bool)
    mask[order[: max_i + 1]] = True
    return mask


def permutation_edge_pvalues(corr: pd.DataFrame, null_pool: np.ndarray) -> pd.DataFrame:
    """Empirical two-sided p-value per pair: fraction of null |r| >= observed |r|."""
    items = list(corr.columns)
    n = len(items)
    iu = np.triu_indices(n, k=1)
    r_vals = corr.to_numpy()[iu]
    pooled_sorted = np.sort(null_pool)
    # p = fraction of pooled null values >= |r|  (searchsorted on sorted ascending array)
    idx = np.searchsorted(pooled_sorted, np.abs(r_vals), side="left")
    pvals = (len(pooled_sorted) - idx) / len(pooled_sorted)
    pvals = np.clip(pvals, 1.0 / len(pooled_sorted), 1.0)
    return pd.DataFrame({"i": [items[a] for a in iu[0]], "j": [items[b] for b in iu[1]], "r": r_vals, "p": pvals})


def build_signed_graph(
    corr: pd.DataFrame,
    threshold: float | None = None,
    block_of: pd.Series | None = None,
    keep_pairs: set | None = None,
) -> nx.Graph:
    """Keep edges with |r| > threshold (fixed-threshold mode), or edges whose
    (i, j) pair appears in keep_pairs (FDR-mask mode). Sign retained as edge
    attribute 'sign' (+1/-1), magnitude as 'weight'.
    """
    G = nx.Graph()
    items = list(corr.columns)
    G.add_nodes_from(items)
    if block_of is not None:
        nx.set_node_attributes(G, block_of.to_dict(), "block")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            r = corr.iloc[i, j]
            if pd.isna(r):
                continue
            if keep_pairs is not None:
                if (items[i], items[j]) not in keep_pairs and (items[j], items[i]) not in keep_pairs:
                    continue
            elif threshold is None or abs(r) <= threshold:
                continue
            G.add_edge(
                items[i],
                items[j],
                weight=float(abs(r)),
                r=float(r),
                sign=1 if r > 0 else -1,
            )
    return G


def detect_communities(G: nx.Graph, seed: int = SEED) -> dict:
    """Louvain communities on the positive-edge subgraph (standard practice for
    signed networks: modularity maximization is undefined/unstable with mixed
    signs, so community structure is read off the agreement backbone; negative
    edges are analysed separately via structural balance in balance.py).
    Returns {node: community_id}.
    """
    pos_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["sign"] > 0]
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_weighted_edges_from([(u, v, d["weight"]) for u, v, d in pos_edges])
    communities = nx.community.louvain_communities(H, weight="weight", seed=seed)
    return {node: cid for cid, comm in enumerate(communities) for node in comm}


def hub_ranking(G: nx.Graph) -> pd.DataFrame:
    """Statement strength (sum of |r| over kept edges), split by sign."""
    rows = []
    for n in G.nodes():
        pos_strength = sum(d["weight"] for _, _, d in G.edges(n, data=True) if d["sign"] > 0)
        neg_strength = sum(d["weight"] for _, _, d in G.edges(n, data=True) if d["sign"] < 0)
        rows.append(
            {
                "item": n,
                "block": G.nodes[n].get("block"),
                "degree": G.degree(n),
                "pos_strength": pos_strength,
                "neg_strength": neg_strength,
                "total_strength": pos_strength + neg_strength,
            }
        )
    return pd.DataFrame(rows).sort_values("total_strength", ascending=False).reset_index(drop=True)


def graph_summary(G: nx.Graph) -> dict:
    pos = sum(1 for _, _, d in G.edges(data=True) if d["sign"] > 0)
    neg = sum(1 for _, _, d in G.edges(data=True) if d["sign"] < 0)
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_positive": pos,
        "n_negative": neg,
        "density": nx.density(G),
        "n_isolates": len(list(nx.isolates(G))),
    }


if __name__ == "__main__":
    from load import load_clean_responses

    coded = load_clean_responses()
    blocks = item_blocks(coded.columns)

    corr_primary = pairwise_complete_corr(coded)
    corr_alt = mean_imputed_corr(coded)
    diff = (corr_primary - corr_alt).abs()
    print(f"Mean |primary - mean-imputed| correlation difference: {diff.values[np.triu_indices(60, 1)].mean():.4f}")

    null_pool = permutation_null_pool(coded)
    edge_p = permutation_edge_pvalues(corr_primary, null_pool)
    mask = fdr_bh_threshold_mask(edge_p["p"].to_numpy(), FDR_Q)
    keep_pairs = set(zip(edge_p.loc[mask, "i"], edge_p.loc[mask, "j"]))
    print(f"Permutation+FDR: {mask.sum()} / {len(mask)} pairs significant at q={FDR_Q}")
    print(f"Smallest |r| kept: {edge_p.loc[mask, 'r'].abs().min():.3f}" if mask.any() else "none kept")

    G_primary = build_signed_graph(corr_primary, block_of=blocks, keep_pairs=keep_pairs)
    G_fixed = build_signed_graph(corr_primary, threshold=FIXED_R_THRESHOLD, block_of=blocks)

    print("Primary (permutation + FDR):", graph_summary(G_primary))
    print(f"Sensitivity (fixed |r| > {FIXED_R_THRESHOLD}):", graph_summary(G_fixed))
