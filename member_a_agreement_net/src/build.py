"""
build.py -- member_a_agreement_net

Builds the agreement network between students: nodes = 87 respondents
(raw CSV + the shared >30-missing-per-row rule -- see load.py). Edge weight
= similarity of per-item z-scored opinion vectors, pairwise-complete on
missing items.

Three arbitrary construction choices are made explicit and compared
(per CLAUDE.md: "run the alternative too and record the difference"):
  1. Edge-selection rule: fixed similarity threshold vs. mutual-kNN vs.
     disparity-filter backbone.
  2. Missing-value handling: pairwise-complete cosine similarity (primary)
     vs. cosine similarity on mean-imputed z-scores (sensitivity check).
Also provides a threshold sweep (percolation / giant-component emergence)
and a degree-and-n-matched Erdos-Renyi null model generator.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from load import SEED, load_clean, zscore_items, zscore_items_mean_imputed

# ---------------------------------------------------------------------------
# Constants (no magic numbers inline)
# ---------------------------------------------------------------------------
MIN_SHARED_ITEMS = 20          # min overlapping non-missing items required for an edge to be computable
SIM_THRESHOLD = 0.15           # primary fixed-threshold edge rule (cosine similarity on z-scores)
KNN_K = 8                      # k for mutual-kNN edge rule
DISPARITY_ALPHA = 0.10         # significance level for disparity-filter backbone
PERCOLATION_THRESHOLDS = np.round(np.arange(-0.30, 0.55, 0.025), 3)
N_ER_REALIZATIONS = 30         # seeded realizations for the ER null model


def pairwise_complete_cosine(z: pd.DataFrame, min_shared: int = MIN_SHARED_ITEMS) -> pd.DataFrame:
    """
    Cosine similarity between every pair of respondents' z-scored vectors,
    using only items both respondents answered (pairwise-complete). Pairs
    sharing fewer than `min_shared` items get NaN (similarity not
    computable / not trusted), rather than a silently noisy estimate.
    """
    ids = z.index.to_list()
    n = len(ids)
    mat = z.to_numpy()
    valid = ~np.isnan(mat)
    filled = np.nan_to_num(mat, nan=0.0)

    sim = np.full((n, n), np.nan)
    shared = np.full((n, n), 0, dtype=int)
    for i in range(n):
        vi = valid[i]
        for j in range(i, n):
            vj = valid[j]
            both = vi & vj
            k = int(both.sum())
            shared[i, j] = shared[j, i] = k
            if k < min_shared:
                continue
            a = filled[i][both]
            b = filled[j][both]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                continue
            s = float(np.dot(a, b) / (na * nb))
            sim[i, j] = sim[j, i] = s
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=ids, columns=ids), pd.DataFrame(shared, index=ids, columns=ids)


def mean_imputed_cosine(coded: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity check: cosine similarity on mean-imputed (NaN -> 0) z-scores, dense, no missing-pair issue."""
    zi = zscore_items_mean_imputed(coded)
    mat = zi.to_numpy()
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms
    sim = normed @ normed.T
    return pd.DataFrame(sim, index=coded.index, columns=coded.index)


def build_threshold_graph(sim: pd.DataFrame, threshold: float = SIM_THRESHOLD) -> nx.Graph:
    """Primary edge rule: keep edges with similarity > threshold."""
    ids = list(sim.columns)
    G = nx.Graph()
    G.add_nodes_from(ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s = sim.iloc[i, j]
            if pd.isna(s) or s <= threshold:
                continue
            G.add_edge(ids[i], ids[j], weight=float(s))
    return G


def build_mutual_knn_graph(sim: pd.DataFrame, k: int = KNN_K) -> nx.Graph:
    """Sensitivity-check edge rule: keep edge (i, j) iff j is among i's top-k neighbors AND vice versa."""
    ids = list(sim.columns)
    n = len(ids)
    s = sim.copy()
    arr = s.to_numpy().copy()
    np.fill_diagonal(arr, -np.inf)
    s = pd.DataFrame(arr, index=s.index, columns=s.columns)
    s = s.fillna(-np.inf)
    topk = {ids[i]: set(s.iloc[i].nlargest(k).index) for i in range(n)}

    G = nx.Graph()
    G.add_nodes_from(ids)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            if b in topk[a] and a in topk[b]:
                w = sim.iloc[i, j]
                if pd.notna(w):
                    G.add_edge(a, b, weight=float(w))
    return G


def build_disparity_backbone(sim: pd.DataFrame, alpha: float = DISPARITY_ALPHA, threshold: float = SIM_THRESHOLD) -> nx.Graph:
    """
    Disparity-filter backbone (Serrano, Boguna, Vespignani 2009), applied to
    the positive part of the similarity matrix (edge weights must be
    non-negative for the filter's null model to be meaningful). Starts from
    the threshold graph so only already-positive, already-meaningful edges
    are candidates, then keeps an edge at node i if its normalized weight
    p_ij = w_ij / sum_k(w_ik) is statistically significant at level alpha
    under the filter's null (uniform weight distribution given degree k_i).
    An edge survives if it is significant from EITHER endpoint's perspective.
    """
    base = build_threshold_graph(sim, threshold=threshold)
    G = nx.Graph()
    G.add_nodes_from(base.nodes())

    def alpha_ij(w, k, s):
        if k < 2 or s <= 0:
            return 1.0
        p = w / s
        return (1 - p) ** (k - 1)

    keep = set()
    for i in base.nodes():
        neighbors = list(base.neighbors(i))
        k = len(neighbors)
        if k == 0:
            continue
        s = sum(base[i][j]["weight"] for j in neighbors)
        for j in neighbors:
            w = base[i][j]["weight"]
            if alpha_ij(w, k, s) < alpha:
                keep.add((i, j) if i < j else (j, i))

    for (i, j) in keep:
        G.add_edge(i, j, weight=base[i][j]["weight"])
    return G


def giant_component_fraction(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(G)), default=0)
    return largest / G.number_of_nodes()


def percolation_sweep(sim: pd.DataFrame, thresholds: np.ndarray = PERCOLATION_THRESHOLDS) -> pd.DataFrame:
    """Sweep the similarity threshold, record giant-component fraction and edge count at each."""
    rows = []
    for t in thresholds:
        G = build_threshold_graph(sim, threshold=t)
        rows.append({
            "threshold": float(t),
            "n_edges": G.number_of_edges(),
            "giant_component_fraction": giant_component_fraction(G),
            "n_isolates": len(list(nx.isolates(G))),
        })
    return pd.DataFrame(rows)


def er_null_model(n: int, mean_degree: float, seed: int) -> nx.Graph:
    """Erdos-Renyi G(n, p) matched on n and mean degree to the observed graph."""
    p = min(1.0, mean_degree / (n - 1)) if n > 1 else 0.0
    return nx.gnp_random_graph(n, p, seed=seed)


def er_null_ensemble(n: int, mean_degree: float, n_realizations: int = N_ER_REALIZATIONS, seed: int = SEED):
    """Yield n_realizations independently seeded ER graphs matched on (n, mean_degree)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_realizations)
    return [er_null_model(n, mean_degree, int(s)) for s in seeds]


def graph_summary(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    mean_deg = (2 * m / n) if n else 0.0
    return {
        "n_nodes": n,
        "n_edges": m,
        "mean_degree": mean_deg,
        "density": nx.density(G),
        "giant_component_fraction": giant_component_fraction(G),
        "n_isolates": len(list(nx.isolates(G))),
    }


def edge_overlap(G1: nx.Graph, G2: nx.Graph) -> float:
    """Jaccard overlap between two edge sets (undirected, order-independent)."""
    e1 = {frozenset(e) for e in G1.edges()}
    e2 = {frozenset(e) for e in G2.edges()}
    if not e1 and not e2:
        return 1.0
    return len(e1 & e2) / len(e1 | e2)


if __name__ == "__main__":
    coded = load_clean()
    z = zscore_items(coded)

    sim, shared = pairwise_complete_cosine(z)
    iu = np.triu_indices(len(sim), 1)
    vals = sim.to_numpy()[iu]
    vals = vals[~np.isnan(vals)]
    print(f"pairwise-complete z-scored cosine similarity: n_pairs={len(vals)}, "
          f"mean={vals.mean():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")

    sim_imputed = mean_imputed_cosine(coded)
    ivals = sim_imputed.to_numpy()[iu]
    print(f"mean-imputed cosine similarity: mean={ivals.mean():.3f}, "
          f"min={ivals.min():.3f}, max={ivals.max():.3f}")

    G_thresh = build_threshold_graph(sim, SIM_THRESHOLD)
    G_knn = build_mutual_knn_graph(sim, KNN_K)
    G_backbone = build_disparity_backbone(sim, DISPARITY_ALPHA, SIM_THRESHOLD)

    print("\nThreshold graph:", graph_summary(G_thresh))
    print("Mutual-kNN graph:", graph_summary(G_knn))
    print("Disparity-backbone graph:", graph_summary(G_backbone))
    print(f"\nEdge overlap threshold vs kNN: {edge_overlap(G_thresh, G_knn):.3f}")
    print(f"Edge overlap threshold vs backbone: {edge_overlap(G_thresh, G_backbone):.3f}")

    sweep = percolation_sweep(sim)
    crossing = sweep[sweep["giant_component_fraction"] >= 0.5]
    if len(crossing):
        print(f"\nGiant component reaches >=50% of nodes at threshold >= {crossing.iloc[-1]['threshold']}")
