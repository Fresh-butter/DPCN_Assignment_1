"""
bipartite_metrics.py -- member_c_bipartite_net

Computes structural metrics for the bipartite deviation network and its projections:
  - Degree distributions (both sides)
  - Nestedness (NODF — Nestedness metric based on Overlap and Decreasing Fill)
  - Core-periphery detection (degree-based)
  - Bipartite clustering coefficient
  - Community detection (Louvain on one-mode projections)
"""

from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
from community import best_partition  # python-louvain

sys.path.insert(0, str(Path(__file__).parent))
from load import load_clean, zscore_items, SEED, item_block, item_block_name
from build import build_bipartite, student_projection, item_projection, PRIMARY_CUTOFF

MEMBER_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = MEMBER_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Degree distributions
# ---------------------------------------------------------------------------

def degree_table(G: nx.Graph, node_type: str) -> pd.DataFrame:
    """Return sorted degree table for nodes of given type ('student' or 'item')."""
    rows = []
    for n, d in G.nodes(data=True):
        if d.get("node_type") == node_type:
            deg = G.degree(n)
            row = {"node": n, "degree": deg}
            if node_type == "item":
                row["block"] = item_block(n)
                row["block_name"] = item_block_name(n)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("degree", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# NODF nestedness
# ---------------------------------------------------------------------------

def _nodf_from_matrix(M: np.ndarray) -> float:
    """
    NODF straight from a binary incidence matrix.

    Same definition as compute_nodf (which reports the row and column halves
    separately); kept matrix-level so the randomised null can call it directly
    without rebuilding a graph for each of its replicates.
    """
    def _pairs(mat):
        row_sums = mat.sum(axis=1)
        total, count = 0.0, 0
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[0]):
                ki, kj = row_sums[i], row_sums[j]
                if ki == kj or max(ki, kj) == 0:
                    continue
                total += int(np.dot(mat[i], mat[j])) / min(ki, kj)
                count += 1
        return (total / count * 100) if count else 0.0

    return (_pairs(M) + _pairs(M.T)) / 2


def incidence_matrix(G: nx.Graph):
    """
    Binary students x items incidence matrix, with sorted node orders.

    Sorted (not set order) because the row/col ordering reaches the randomised
    null below, and Python set iteration is not stable across processes.
    """
    students = sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "student")
    items = sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "item")
    s_idx = {s: i for i, s in enumerate(students)}
    q_idx = {q: j for j, q in enumerate(items)}

    M = np.zeros((len(students), len(items)), dtype=int)
    for u, v in G.edges():
        if u in s_idx and v in q_idx:
            M[s_idx[u], q_idx[v]] = 1
        elif v in s_idx and u in q_idx:
            M[s_idx[v], q_idx[u]] = 1
    return M, students, items


def compute_nodf(G: nx.Graph) -> dict:
    """
    Compute NODF (Nestedness based on Overlap and Decreasing Fill).
    NODF = (N_rows + N_cols) / 2, where N_rows (N_cols) is the average
    paired nestedness across all row (col) pairs, considering only pairs
    where the higher-degree row/col has strictly more filled entries.

    Returns dict with: nodf, nodf_rows, nodf_cols
    """
    M, students, items = incidence_matrix(G)

    def _nodf_pairs(mat):
        """Compute N for a 2D binary matrix (pairs along axis=0)."""
        n = mat.shape[0]
        row_sums = mat.sum(axis=1)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                ki, kj = row_sums[i], row_sums[j]
                if ki == kj:
                    continue  # no nestedness when equal fill
                k_max = max(ki, kj)
                if k_max == 0:
                    continue
                # shared = overlap
                shared = int(np.dot(mat[i], mat[j]))
                # paired nestedness = shared / min(ki, kj)
                total += shared / min(ki, kj)
                count += 1
        return (total / count * 100) if count > 0 else 0.0

    nodf_rows = _nodf_pairs(M)
    nodf_cols = _nodf_pairs(M.T)
    nodf = (nodf_rows + nodf_cols) / 2
    assert abs(nodf - _nodf_from_matrix(M)) < 1e-9, "NODF implementations disagree"

    return {"nodf": nodf, "nodf_rows": nodf_rows, "nodf_cols": nodf_cols,
            "n_students": len(students), "n_items": len(items), "n_edges": G.number_of_edges()}


# ---------------------------------------------------------------------------
# NODF against a degree-preserving null
# ---------------------------------------------------------------------------

def _curveball_swap(A: np.ndarray, rng, n_swaps: int) -> np.ndarray:
    """
    Randomise a binary matrix while preserving BOTH degree sequences.

    Repeatedly picks two rows and swaps a checkerboard pair -- one column where
    only the first row has an edge, one where only the second does. Each swap
    leaves every row sum and every column sum unchanged, so the randomised
    matrix has exactly the deviation counts of the real one and differs only in
    which student deviates on which item.
    """
    A = A.copy()
    n_rows = A.shape[0]
    done = 0
    while done < n_swaps:
        i, j = rng.integers(0, n_rows, 2)
        if i == j:
            continue
        only_i = np.flatnonzero((A[i] == 1) & (A[j] == 0))
        only_j = np.flatnonzero((A[i] == 0) & (A[j] == 1))
        if len(only_i) == 0 or len(only_j) == 0:
            continue
        ci, cj = rng.choice(only_i), rng.choice(only_j)
        A[i, ci], A[j, ci] = 0, 1
        A[i, cj], A[j, cj] = 1, 0
        done += 1
    return A


def nodf_null(G: nx.Graph, n_rep: int = 30, n_swaps: int = 20000,
              seed: int = SEED) -> dict:
    """
    Compare the observed NODF against a degree-preserving null.

    NODF rises mechanically with heterogeneous degrees, and this network has
    very heterogeneous degrees by construction (students deviate on 3 to 35
    items). A bare NODF therefore cannot distinguish "dissent is organised as a
    hierarchy" from "some students simply deviate more than others". The null
    fixes both degree sequences and destroys everything else, so whatever the
    observed value clears here is structure that degree alone does not explain.
    """
    M, _, _ = incidence_matrix(G)
    observed = _nodf_from_matrix(M)

    rng = np.random.default_rng(seed)
    null_vals = []
    for _ in range(n_rep):
        R = _curveball_swap(M, rng, n_swaps)
        assert (R.sum(axis=1) == M.sum(axis=1)).all(), "row degrees not preserved"
        assert (R.sum(axis=0) == M.sum(axis=0)).all(), "column degrees not preserved"
        null_vals.append(_nodf_from_matrix(R))

    null_vals = np.array(null_vals)
    null_mean, null_sd = null_vals.mean(), null_vals.std()
    return {
        "nodf_observed": observed,
        "nodf_null_mean": null_mean,
        "nodf_null_sd": null_sd,
        "nodf_null_min": null_vals.min(),
        "nodf_null_max": null_vals.max(),
        "z": (observed - null_mean) / null_sd if null_sd > 0 else np.nan,
        "excess": observed - null_mean,
        "n_rep": n_rep,
    }


# ---------------------------------------------------------------------------
# Core-periphery (degree-based)
# ---------------------------------------------------------------------------

def core_periphery_students(G: nx.Graph, top_frac: float = 0.2) -> dict:
    """
    Simple degree-based core-periphery on student side.
    Core = top `top_frac` fraction by degree (most contrarian students).
    Returns dict: {student_id: 'core'|'periphery'}
    """
    deg_table = degree_table(G, "student")
    n_core = max(1, int(len(deg_table) * top_frac))
    core_nodes = set(deg_table.head(n_core)["node"])
    return {row["node"]: ("core" if row["node"] in core_nodes else "periphery")
            for _, row in deg_table.iterrows()}


def core_periphery_items(G: nx.Graph, top_frac: float = 0.2) -> dict:
    """Same but on item side."""
    deg_table = degree_table(G, "item")
    n_core = max(1, int(len(deg_table) * top_frac))
    core_nodes = set(deg_table.head(n_core)["node"])
    return {row["node"]: ("core" if row["node"] in core_nodes else "periphery")
            for _, row in deg_table.iterrows()}


# ---------------------------------------------------------------------------
# Bipartite clustering coefficient
# ---------------------------------------------------------------------------

def bipartite_clustering(G: nx.Graph, node_type: str = "student") -> pd.DataFrame:
    """
    Bipartite clustering coefficient for all nodes of given type.
    Uses networkx's bipartite.clustering (Latapy et al. definition).
    """
    nodes = {n for n, d in G.nodes(data=True) if d.get("node_type") == node_type}
    cc = nx.bipartite.clustering(G, nodes=nodes)
    df = pd.DataFrame([{"node": k, "clustering": v} for k, v in cc.items()])
    return df.sort_values("clustering", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Community detection on projections
# ---------------------------------------------------------------------------

def louvain_communities(P: nx.Graph, seed: int = SEED) -> dict:
    """
    Louvain community detection on a (one-mode) projection.
    Returns {node: community_id} dict and modularity Q.
    """
    partition = best_partition(P, random_state=seed)
    Q = nx.community.modularity(
        P,
        [{n for n, c in partition.items() if c == cid}
         for cid in set(partition.values())]
    )
    return partition, Q


if __name__ == "__main__":
    clean = load_clean()
    z = zscore_items(clean)
    G = build_bipartite(z, cutoff=PRIMARY_CUTOFF)

    # Degree distributions
    std_deg = degree_table(G, "student")
    itm_deg = degree_table(G, "item")
    print("Top-10 most-contrarian students:")
    print(std_deg.head(10).to_string(index=False))
    print("\nTop-10 most-deviated items:")
    print(itm_deg.head(10).to_string(index=False))
    std_deg.to_csv(TABLES_DIR / "student_degree.csv", index=False)
    itm_deg.to_csv(TABLES_DIR / "item_degree.csv", index=False)

    # NODF nestedness
    nodf = compute_nodf(G)
    print(f"\nNODF nestedness: {nodf['nodf']:.2f}  "
          f"(rows={nodf['nodf_rows']:.2f}, cols={nodf['nodf_cols']:.2f})")
    pd.DataFrame([nodf]).to_csv(TABLES_DIR / "nodf.csv", index=False)

    # Core-periphery
    cp_s = core_periphery_students(G)
    n_core = sum(1 for v in cp_s.values() if v == "core")
    print(f"\nCore students (top 20% by degree): {n_core}")
    cp_df = pd.DataFrame([{"student": k, "role": v} for k, v in cp_s.items()])
    cp_df.to_csv(TABLES_DIR / "core_periphery_students.csv", index=False)

    # Bipartite clustering
    bc = bipartite_clustering(G, "student")
    print(f"\nBipartite clustering (students): mean={bc['clustering'].mean():.3f}")
    bc.to_csv(TABLES_DIR / "bipartite_clustering.csv", index=False)

    # Community detection on projections
    Ps = student_projection(G)
    Pi = item_projection(G)
    part_s, Q_s = louvain_communities(Ps)
    part_i, Q_i = louvain_communities(Pi)
    n_comm_s = len(set(part_s.values()))
    n_comm_i = len(set(part_i.values()))
    print(f"\nStudent co-dissent projection: {n_comm_s} communities, Q={Q_s:.3f}")
    print(f"Item co-deviation projection:  {n_comm_i} communities, Q={Q_i:.3f}")
    pd.DataFrame([{"node": k, "community": v}
                  for k, v in part_s.items()]).to_csv(
        TABLES_DIR / "student_communities.csv", index=False)
    pd.DataFrame([{"node": k, "community": v, "block": item_block(k)}
                  for k, v in part_i.items()]).to_csv(
        TABLES_DIR / "item_communities.csv", index=False)


def modularity_null(P: nx.Graph, n_realizations: int = 30, seed: int = SEED) -> dict:
    """
    Observed Louvain modularity of a one-mode projection against a
    degree-preserving (configuration-model) null.

    Needed because Louvain always returns *a* partition. On this project's
    co-dissent projection Q is around 0.04, and the only way to say whether
    0.04 means "weak structure" or "no structure at all" is to measure what a
    graph with the same degrees but no structure scores.
    """
    deg = [d for _, d in P.degree()]
    _, q_obs = louvain_communities(P, seed=seed)

    q_null = []
    for i in range(n_realizations):
        H = nx.configuration_model(deg, seed=seed + i)
        H = nx.Graph(H)
        H.remove_edges_from(nx.selfloop_edges(H))
        H = nx.relabel_nodes(H, {n: str(n) for n in H.nodes()})
        _, q = louvain_communities(H, seed=seed)
        q_null.append(q)

    mu, sd = float(np.mean(q_null)), float(np.std(q_null))
    return {
        "modularity_observed": float(q_obs),
        "modularity_null_mean": mu,
        "modularity_null_sd": sd,
        "z_score": float((q_obs - mu) / sd) if sd > 0 else float("nan"),
        "n_realizations": n_realizations,
    }
