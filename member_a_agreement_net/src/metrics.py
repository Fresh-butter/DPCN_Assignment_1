"""
metrics.py -- member_a_agreement_net

Structural analysis of the built agreement network: degree distribution
(log binning), clustering, centralities, community detection, and
comparison against the n/mean-degree-matched Erdos-Renyi null model
(mean +/- SD across N_ER_REALIZATIONS seeded realizations).
"""

import networkx as nx
import numpy as np
import pandas as pd

from build import N_ER_REALIZATIONS, er_null_ensemble, graph_summary
from load import SEED

N_LOG_BINS = 10


def degree_distribution(G: nx.Graph) -> pd.DataFrame:
    degrees = [d for _, d in G.degree()]
    return pd.DataFrame({"node": list(G.nodes()), "degree": degrees})


def log_binned_degree_histogram(G: nx.Graph, n_bins: int = N_LOG_BINS) -> pd.DataFrame:
    """
    Logarithmic-binning degree histogram (Module 2d). Degree-0 nodes are
    reported separately since log(0) is undefined; positive degrees are
    binned into n_bins logarithmically spaced bins between 1 and max degree.
    """
    degrees = np.array([d for _, d in G.degree()])
    zero_count = int((degrees == 0).sum())
    pos = degrees[degrees > 0]
    rows = [{"bin_low": 0, "bin_high": 0, "count": zero_count, "density": np.nan}]
    if len(pos):
        edges = np.logspace(0, np.log10(pos.max()), n_bins + 1)
        edges[-1] *= 1.0001  # include the max value in the last bin
        counts, _ = np.histogram(pos, bins=edges)
        widths = np.diff(edges)
        density = counts / widths / len(degrees)
        for lo, hi, c, dens in zip(edges[:-1], edges[1:], counts, density):
            rows.append({"bin_low": lo, "bin_high": hi, "count": int(c), "density": float(dens)})
    return pd.DataFrame(rows)


def clustering_metrics(G: nx.Graph) -> dict:
    return {
        "global_clustering_transitivity": nx.transitivity(G),
        "average_local_clustering": nx.average_clustering(G),
        "n_connected_components": nx.number_connected_components(G),
        "avg_shortest_path_giant_component": _avg_path_giant(G),
    }


def _avg_path_giant(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return float("nan")
    giant_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(giant_nodes)
    if H.number_of_nodes() < 2:
        return float("nan")
    return nx.average_shortest_path_length(H)


def centralities(G: nx.Graph) -> pd.DataFrame:
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G, seed=SEED)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: np.nan for n in G.nodes()}
    degree = dict(G.degree())
    rows = []
    for n in G.nodes():
        rows.append({
            "node": n,
            "degree": degree[n],
            "closeness": closeness[n],
            "betweenness": betweenness[n],
            "eigenvector": eigenvector[n],
        })
    return pd.DataFrame(rows).sort_values("eigenvector", ascending=False).reset_index(drop=True)


def detect_communities(G: nx.Graph, seed: int = SEED) -> dict:
    """Louvain community detection on the (unsigned, positive-weight) agreement graph."""
    communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
    return {node: cid for cid, comm in enumerate(communities) for node in comm}, communities


def modularity(G: nx.Graph, communities) -> float:
    return nx.community.modularity(G, communities, weight="weight")


def null_model_comparison(G: nx.Graph, n_realizations: int = N_ER_REALIZATIONS, seed: int = SEED) -> pd.DataFrame:
    """
    Compare observed clustering / avg path length / n-components against
    an ER null matched on (n, mean_degree), reported as mean +/- SD over
    n_realizations seeded realizations (per CLAUDE.md: never a single
    stochastic realization).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    mean_deg = (2 * m / n) if n else 0.0
    null_graphs = er_null_ensemble(n, mean_deg, n_realizations, seed)

    null_clustering = [nx.average_clustering(H) for H in null_graphs]
    null_path = [_avg_path_giant(H) for H in null_graphs]
    null_gcc = [max((len(c) for c in nx.connected_components(H)), default=0) / n for H in null_graphs]

    obs = clustering_metrics(G)
    obs_summary = graph_summary(G)

    rows = [
        {"metric": "average_local_clustering", "observed": obs["average_local_clustering"],
         "null_mean": np.mean(null_clustering), "null_sd": np.std(null_clustering)},
        {"metric": "avg_shortest_path_giant_component", "observed": obs["avg_shortest_path_giant_component"],
         "null_mean": np.nanmean(null_path), "null_sd": np.nanstd(null_path)},
        {"metric": "giant_component_fraction", "observed": obs_summary["giant_component_fraction"],
         "null_mean": np.mean(null_gcc), "null_sd": np.std(null_gcc)},
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from load import load_clean, zscore_items
    from build import SIM_THRESHOLD, build_threshold_graph, pairwise_complete_cosine

    coded = load_clean()
    z = zscore_items(coded)
    sim, _ = pairwise_complete_cosine(z)
    G = build_threshold_graph(sim, SIM_THRESHOLD)

    print("Graph summary:", graph_summary(G))
    print("Clustering metrics:", clustering_metrics(G))

    cent = centralities(G)
    print("\nTop 5 by eigenvector centrality:")
    print(cent.head(5).to_string(index=False))
    print("\nTop 5 by betweenness centrality:")
    print(cent.sort_values("betweenness", ascending=False).head(5).to_string(index=False))

    comm_map, comms = detect_communities(G)
    print(f"\nLouvain communities: {len(comms)}, sizes: {sorted([len(c) for c in comms], reverse=True)}")
    print(f"Modularity: {modularity(G, comms):.4f}")

    null_cmp = null_model_comparison(G)
    print("\nVs. ER null model (mean +/- SD over 30 seeded realizations):")
    print(null_cmp.to_string(index=False))
