"""Dynamics on the statement graph: Laplacian consensus/diffusion and a
perturbation experiment giving an "opinion leverage" ranking.

Two Laplacians are used, deliberately:
- Unsigned (weight = |r|): standard consensus dynamics dx/dt = -Lx. On a
  connected component this always converges to a single shared value (the
  weighted average); it cannot show fragmentation, it only shows how fast
  agreement is reached, so it is used for the perturbation/leverage experiment
  where "does it converge" is not the question, "how far does a nudge travel
  and how fast" is.
- Signed (Altafini model, weight = r, degree = sum of |r|): dx/dt = -L_signed x.
  A classical result (Altafini 2013) ties its outcome directly to structural
  balance (balance.py). A connected signed graph reaches "consensus up to
  sign" (bipartite consensus, opinions split into two mirror-image factions)
  iff it is structurally balanced, if frustrated, every opinion decays to 0
  (no shared reference point survives). We use this as a dynamical readout of
  the balance.py finding rather than a second, independent claim.
"""

import networkx as nx
import numpy as np
from scipy.linalg import expm

# --- constants ---
T_HORIZON = 1.0  # time horizon for the perturbation / leverage experiment
N_TIME_POINTS = 60
T_MAX_TRAJECTORY = 120.0  # long enough to reveal the slow Altafini decay (see FINDINGS.md)


def unsigned_laplacian(G: nx.Graph, nodelist):
    return nx.laplacian_matrix(G, nodelist=nodelist, weight="weight").toarray().astype(float)


def signed_laplacian(G: nx.Graph, nodelist):
    """L = D - W, W_ij = r_ij (signed), D_ii = sum_j |r_ij| (Altafini's construction)."""
    n = len(nodelist)
    idx = {node: k for k, node in enumerate(nodelist)}
    W = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        W[i, j] = W[j, i] = d["r"]
    D = np.diag(np.abs(W).sum(axis=1))
    return D - W


def run_trajectory(L: np.ndarray, x0: np.ndarray, t_points: np.ndarray) -> np.ndarray:
    """x(t) = exp(-L t) x0 for each t in t_points. Returns array (len(t_points), n)."""
    return np.array([expm(-L * t) @ x0 for t in t_points])


def signed_consensus_demo(G: nx.Graph, nodelist, seed_value_map: dict, t_max: float = T_MAX_TRAJECTORY):
    """Run Altafini dynamics from real (deviation-from-neutral) initial opinions
    and report whether the giant component collapses to 0 (frustrated / no
    shared reference) or splits into two mirror factions (balanced).
    """
    L = signed_laplacian(G, nodelist)
    x0 = np.array([seed_value_map.get(n, 0.0) for n in nodelist])
    t_points = np.linspace(0, t_max, N_TIME_POINTS)
    traj = run_trajectory(L, x0, t_points)
    final = traj[-1]
    return {
        "t_points": t_points,
        "trajectory": traj,
        "final_state": final,
        "final_abs_mean": float(np.mean(np.abs(final))),
        "initial_abs_mean": float(np.mean(np.abs(x0))),
    }


def perturbation_leverage(G: nx.Graph, nodelist, t: float = T_HORIZON) -> np.ndarray:
    """M = exp(-L_signed * t). Column i is the response of every statement to a
    unit nudge on statement i. Leverage(i) = sum_j!=i |M[j, i]| - total signed
    magnitude of movement a nudge to statement i propagates through the rest
    of the issue graph within time t.
    """
    L = signed_laplacian(G, nodelist)
    M = expm(-L * t)
    leverage = np.abs(M).sum(axis=0) - np.abs(np.diag(M))
    return leverage


def algebraic_connectivity(G: nx.Graph, nodelist) -> float:
    L = unsigned_laplacian(G, nodelist)
    eigvals = np.linalg.eigvalsh(L)
    return float(sorted(eigvals)[1]) if len(eigvals) > 1 else 0.0


if __name__ == "__main__":
    import pandas as pd

    from balance import frustration_index, triad_census
    from build import (
        build_signed_graph,
        fdr_bh_threshold_mask,
        pairwise_complete_corr,
        permutation_edge_pvalues,
        permutation_null_pool,
    )
    from load import item_blocks, load_clean_responses

    coded = load_clean_responses()
    blocks = item_blocks(coded.columns)
    corr = pairwise_complete_corr(coded)
    null_pool = permutation_null_pool(coded)
    edge_p = permutation_edge_pvalues(corr, null_pool)
    mask = fdr_bh_threshold_mask(edge_p["p"].to_numpy())
    keep_pairs = set(zip(edge_p.loc[mask, "i"], edge_p.loc[mask, "j"]))
    G = build_signed_graph(corr, block_of=blocks, keep_pairs=keep_pairs)

    giant_nodes = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(giant_nodes).copy()
    nodelist = list(Gc.nodes())
    print(f"Giant component: {len(nodelist)} / {G.number_of_nodes()} statements")

    census = triad_census(Gc)
    frust = frustration_index(Gc)
    print(f"Giant component frustration: {frust['frustration_index']}/{frust['n_edges']} "
          f"({frust['frustration_fraction']:.1%}), unbalanced triads: {census['n_unbalanced']}")

    mean_dev = coded[nodelist].mean() - 3.0  # deviation from the neutral Likert midpoint
    demo = signed_consensus_demo(Gc, nodelist, mean_dev.to_dict())
    print(f"Altafini dynamics: |opinion| mean {demo['initial_abs_mean']:.3f} -> {demo['final_abs_mean']:.3f} "
          f"over t=[0,{T_MAX_TRAJECTORY}]")

    leverage = perturbation_leverage(Gc, nodelist)
    ranking = pd.DataFrame({"item": nodelist, "leverage": leverage}).sort_values(
        "leverage", ascending=False
    ).reset_index(drop=True)
    print("Top 5 opinion-leverage statements:")
    print(ranking.head())
