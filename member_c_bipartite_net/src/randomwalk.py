"""
randomwalk.py -- member_c_bipartite_net

Dynamical process: Random walk on the bipartite deviation network.

A walker hops: student -> item -> student -> ...
The Mean First Passage Time (MFPT) between two student nodes is the expected
number of steps for a walker starting at student s to first reach student t
(via the item intermediaries).

This gives an "issue-mediated distance" between students — two students are
close if they share many contrarian items (likely to meet quickly via those
items), and far apart if their deviation profiles don't overlap.

We also run a robustness experiment: remove the highest-degree item nodes
(E03, E02, etc.) and measure how MFPT changes. If those items are the main
bridges between pockets of dissent, their removal should dramatically
increase average MFPT.
"""

from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load import load_clean, zscore_items, SEED
from build import build_bipartite, student_projection, PRIMARY_CUTOFF
from bipartite_metrics import degree_table

MEMBER_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = MEMBER_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Random walk transition matrix
# ---------------------------------------------------------------------------

def transition_matrix_bipartite(G: nx.Graph) -> tuple:
    """
    Build the lazy random walk transition matrix for a bipartite graph.
    Walker at node u picks a neighbour uniformly at random.
    Returns (T, nodes) where T[i,j] = P(go from node i to node j).
    """
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    T = np.zeros((n, n))
    for u in nodes:
        nbrs = list(G.neighbors(u))
        if len(nbrs) == 0:
            T[node_idx[u], node_idx[u]] = 1.0   # absorbing if isolated
            continue
        w = np.array([G[u][v].get("weight", 1.0) for v in nbrs], dtype=float)
        w /= w.sum()
        for i_nbr, v in enumerate(nbrs):
            T[node_idx[u], node_idx[v]] = w[i_nbr]
    return T, nodes


def stationary_distribution(G: nx.Graph, nodes: list) -> np.ndarray:
    """
    Stationary distribution of a weighted random walk: pi_j proportional to
    node strength (sum of incident edge weights). This is NOT uniform unless
    the graph is regular -- here strengths span roughly 30x, so using a
    uniform pi would give badly wrong first-passage times.
    """
    strength = np.array(
        [sum(G[u][v].get("weight", 1.0) for v in G.neighbors(u)) for u in nodes],
        dtype=float,
    )
    total = strength.sum()
    if total <= 0:
        return np.full(len(nodes), 1.0 / len(nodes))
    return strength / total


def mfpt_from_transition(T: np.ndarray, nodes: list, student_nodes: set,
                          pi: np.ndarray) -> pd.DataFrame:
    """
    Exact pairwise MFPT via the fundamental matrix (Grinstead & Snell):

        Z    = (I - T + 1 pi^T)^-1
        H_ij = (Z_jj - Z_ij) / pi_j

    `pi` must be the true stationary distribution of T. Passing a uniform pi
    on a non-regular graph silently produces wrong -- and suspiciously flat --
    first-passage times.

    O(n^3) on n = 147 nodes, so exact inversion is fine.
    """
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    assert np.allclose(pi @ T, pi, atol=1e-10), "pi is not stationary for T"

    Pi = np.outer(np.ones(n), pi)
    try:
        Z = np.linalg.inv(np.eye(n) - T + Pi)
    except np.linalg.LinAlgError:
        s_list = sorted(student_nodes)
        return pd.DataFrame(np.full((len(s_list), len(s_list)), np.nan),
                            index=s_list, columns=s_list)

    H_full = np.zeros((n, n))
    for j in range(n):
        H_full[:, j] = (Z[j, j] - Z[:, j]) / pi[j]
    np.fill_diagonal(H_full, 0.0)

    s_list = [nd for nd in nodes if nd in student_nodes]
    s_idx = [node_idx[s] for s in s_list]
    return pd.DataFrame(H_full[np.ix_(s_idx, s_idx)], index=s_list, columns=s_list)


def relative_mfpt(H: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """
    MFPT normalised by the degree-only expectation.

    For a well-mixed weighted graph the first-passage time to j is about
    `vol / strength_j`, where vol is the total edge weight (x2). So

        R_ij = H_ij * strength_j / vol

    is ~1 when j is exactly as hard to reach as its strength predicts, > 1
    when the walker is structurally held away from j, and < 1 when it is
    funnelled toward j. R is dimensionless and -- unlike raw MFPT -- does not
    shrink automatically when we delete edges, so it is the fair quantity for
    the robustness experiment below.
    """
    vol = 2.0 * sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    strength = {n: sum(G[n][v].get("weight", 1.0) for v in G.neighbors(n)) for n in H.columns}
    return H.mul(pd.Series(strength)[H.columns] / vol, axis=1)


def compute_mfpt(G: nx.Graph) -> pd.DataFrame:
    """Full pipeline: bipartite graph -> MFPT matrix between students."""
    T, nodes = transition_matrix_bipartite(G)
    pi = stationary_distribution(G, nodes)
    students = {n for n, d in G.nodes(data=True) if d.get("node_type") == "student"}
    return mfpt_from_transition(T, nodes, students, pi)


# ---------------------------------------------------------------------------
# Robustness experiment
# ---------------------------------------------------------------------------

def robustness_experiment(G: nx.Graph, n_top_items: int = 5) -> pd.DataFrame:
    """
    Remove top-degree items one at a time and measure:
    - mean MFPT across all student pairs
    - fraction of student pairs that remain connected (finite MFPT)

    Returns DataFrame with one row per removal step.
    """
    itm_deg = degree_table(G, "item")
    top_items = list(itm_deg.head(n_top_items)["node"])

    rows = []

    def _stats(H: pd.DataFrame, Gx: nx.Graph) -> dict:
        """Off-diagonal statistics only -- the diagonal is 0 by definition and
        must not be counted as a 'disconnected' pair (the previous version did,
        which pinned frac_connected at 7482/7569 = 0.989 no matter what)."""
        vals = H.values[~np.eye(H.shape[0], dtype=bool)]
        finite = np.isfinite(vals)
        R = relative_mfpt(H, Gx).values[~np.eye(H.shape[0], dtype=bool)]
        return {
            "mean_mfpt": float(vals[finite].mean()) if finite.any() else np.nan,
            "mean_relative_mfpt": float(R[np.isfinite(R)].mean()) if finite.any() else np.nan,
            "max_mfpt": float(vals[finite].max()) if finite.any() else np.nan,
            "frac_connected": float(finite.mean()),
        }

    # Baseline
    rows.append({"removed_item": "none", "n_edges": G.number_of_edges(),
                 **_stats(compute_mfpt(G), G)})

    # Remove items sequentially
    G_current = G.copy()
    for item in top_items:
        G_current.remove_node(item)
        rows.append({"removed_item": item, "n_edges": G_current.number_of_edges(),
                     **_stats(compute_mfpt(G_current), G_current)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    clean = load_clean()
    z = zscore_items(clean)
    G = build_bipartite(z, cutoff=PRIMARY_CUTOFF)

    students = [n for n, d in G.nodes(data=True) if d.get("node_type") == "student"]
    print(f"Computing MFPT on bipartite graph: "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    H = compute_mfpt(G)
    off = H.values[~np.eye(H.shape[0], dtype=bool)]
    finite_vals = off[np.isfinite(off)]
    print(f"MFPT stats across {len(finite_vals)} connected student pairs "
          f"(of {len(off)} ordered pairs):")
    print(f"  mean  = {finite_vals.mean():.2f}")
    print(f"  median= {np.median(finite_vals):.2f}")
    print(f"  min   = {finite_vals.min():.2f}")
    print(f"  max   = {finite_vals.max():.2f}")

    H.to_csv(TABLES_DIR / "mfpt_matrix.csv")
    print(f"MFPT matrix saved.")

    # Robustness
    print("\nRunning robustness experiment (removing top-5 items)...")
    rob = robustness_experiment(G, n_top_items=5)
    print(rob.to_string(index=False))
    rob.to_csv(TABLES_DIR / "robustness_mfpt.csv", index=False)
