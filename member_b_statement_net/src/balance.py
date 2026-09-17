"""Structural balance analysis of the signed statement network.

Closed-triad balance (classic Heider/Cartwright-Harary theory): a triangle is
balanced if the product of its three edge signs is positive (+++ or +--),
unbalanced ("frustrated") otherwise (++- or ---). A frustrated triad among
three statements means there is no way to read them as "two camps agreeing
within, disagreeing across"; the class's stance on those three statements is
internally incoherent.

We also compute the frustration index: the minimum number of edges that
would need to flip sign for the whole graph to become perfectly balanced
(equivalently, the best 2-faction split of statements). Exact computation is
NP-hard; we use greedy local-search (a standard correlation-clustering /
Ising-ground-state heuristic) with multiple random restarts.
"""

import networkx as nx
import numpy as np

from load import SEED

# --- constants ---
N_RESTARTS = 50
MAX_ITERS = 200
N_NULL_SIGN_SHUFFLES = 500


def triad_census(G: nx.Graph) -> dict:
    """Count balanced vs unbalanced closed triads (each triangle counted once)."""
    seen = set()
    balanced, unbalanced = 0, 0
    frustrated_triads = []
    for u, v in G.edges():
        common = set(G[u]) & set(G[v])
        for w in common:
            tri = tuple(sorted((u, v, w)))
            if tri in seen:
                continue
            seen.add(tri)
            s1 = G[tri[0]][tri[1]]["sign"]
            s2 = G[tri[0]][tri[2]]["sign"]
            s3 = G[tri[1]][tri[2]]["sign"]
            product = s1 * s2 * s3
            if product > 0:
                balanced += 1
            else:
                unbalanced += 1
                frustrated_triads.append(tri)
    total = balanced + unbalanced
    return {
        "n_closed_triads": total,
        "n_balanced": balanced,
        "n_unbalanced": unbalanced,
        "frac_balanced": balanced / total if total else float("nan"),
        "frustrated_triads": frustrated_triads,
    }


def _frustration_for_assignment(G: nx.Graph, spin: dict) -> int:
    return sum(1 for u, v, d in G.edges(data=True) if d["sign"] * spin[u] * spin[v] < 0)


def _greedy_local_search(G: nx.Graph, rng: np.random.Generator) -> tuple[dict, int]:
    nodes = list(G.nodes())
    spin = {n: rng.choice([-1, 1]) for n in nodes}
    for _ in range(MAX_ITERS):
        changed = False
        order = rng.permutation(nodes)
        for n in order:
            neighbors = G[n]
            if not neighbors:
                # isolated node: no edges to satisfy, spin is unconstrained and
                # left untouched (never forced to a default faction)
                continue
            field = sum(d["sign"] * spin[m] for m, d in neighbors.items())
            new_spin = 1 if field >= 0 else -1
            if new_spin != spin[n]:
                spin[n] = new_spin
                changed = True
        if not changed:
            break
    return spin, _frustration_for_assignment(G, spin)


def frustration_index(G: nx.Graph, seed: int = SEED, n_restarts: int = N_RESTARTS) -> dict:
    """Best-of-n_restarts greedy local search for the minimum-frustration 2-faction split."""
    rng = np.random.default_rng(seed)
    best_spin, best_frustration = None, float("inf")
    for _ in range(n_restarts):
        spin, frustration = _greedy_local_search(G, rng)
        if frustration < best_frustration:
            best_frustration, best_spin = frustration, spin
    n_edges = G.number_of_edges()
    return {
        "frustration_index": best_frustration,
        "n_edges": n_edges,
        "frustration_fraction": best_frustration / n_edges if n_edges else float("nan"),
        "faction_assignment": best_spin,
    }


def frustration_null_distribution(
    G: nx.Graph, seed: int = SEED, n_shuffles: int = N_NULL_SIGN_SHUFFLES, n_restarts: int = 10
) -> np.ndarray:
    """Null model: reshuffle which existing edges are negative vs positive
    (topology and the 391/11 pos/neg split held fixed), recompute the minimum
    frustration for each shuffle. Tests whether the placement of negative
    edges (not just their count) drives the observed frustration.
    """
    rng = np.random.default_rng(seed)
    signs = [d["sign"] for _, _, d in G.edges(data=True)]
    edges = list(G.edges())
    null_frustrations = np.empty(n_shuffles)
    for s in range(n_shuffles):
        shuffled_signs = rng.permutation(signs)
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for (u, v), sign in zip(edges, shuffled_signs):
            H.add_edge(u, v, sign=int(sign))
        _, frustration = min(
            (_greedy_local_search(H, rng) for _ in range(n_restarts)), key=lambda t: t[1]
        )
        null_frustrations[s] = frustration
    return null_frustrations


if __name__ == "__main__":
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

    census = triad_census(G)
    print(f"Closed triads: {census['n_closed_triads']}, balanced: {census['n_balanced']}, "
          f"unbalanced (frustrated): {census['n_unbalanced']} ({1 - census['frac_balanced']:.1%})")

    frust = frustration_index(G)
    print(f"Frustration index: {frust['frustration_index']} / {frust['n_edges']} edges "
          f"({frust['frustration_fraction']:.1%})")

    null_dist = frustration_null_distribution(G)
    z = (frust["frustration_index"] - null_dist.mean()) / null_dist.std()
    print(f"Null (sign-shuffled) frustration: {null_dist.mean():.1f} +/- {null_dist.std():.1f}, z={z:.2f}")
