"""
dynamics.py -- member_a_agreement_net

Two dynamical processes run on the built agreement network:
  1. Bounded-confidence opinion dynamics (Deffuant-Weisbuch): pairwise
     interaction along edges, opinions converge only if within a
     confidence threshold epsilon. Swept over epsilon to find the critical
     value where the outcome flips from fragmentation to consensus.
  2. SI spreading, seeded from high-eigenvector-centrality nodes vs.
     random nodes, comparing time-to-full-coverage.
Every stochastic run is seeded and repeated N_TRIALS times; results are
reported as mean +/- SD, per CLAUDE.md.
"""

import networkx as nx
import numpy as np
import pandas as pd

from load import SEED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TRIALS = 30
MU_CONVERGENCE = 0.5            # Deffuant convergence parameter (how far apart pair moves)
EPSILON_SWEEP = np.round(np.arange(0.05, 1.01, 0.05), 3)  # confidence-threshold sweep
DW_MAX_STEPS = 20000
DW_CLUSTER_TOL = 0.01            # opinions within this distance are considered one cluster
SI_INFECTION_PROB = 0.10         # per-edge, per-timestep infection probability
SI_MAX_STEPS = 500
SI_N_SEED_NODES = 3               # number of seed nodes per trial (high-centrality set size)


def _rng_for_trial(seed: int, trial: int) -> np.random.Generator:
    return np.random.default_rng((seed, trial))


def deffuant_weisbuch(G: nx.Graph, epsilon: float, mu: float = MU_CONVERGENCE,
                       seed: int = SEED, trial: int = 0, max_steps: int = DW_MAX_STEPS) -> np.ndarray:
    """
    Run Deffuant-Weisbuch bounded-confidence dynamics to (near-)convergence.
    Initial opinions: uniform random in [0, 1] per node (agnostic starting
    point -- we are studying the *dynamics*, not re-simulating the survey).
    Each step: pick a random edge; if |opinion_i - opinion_j| < epsilon,
    both move mu of the way toward each other. Stops after max_steps or
    once no active edge remains that would change further.
    """
    rng = _rng_for_trial(seed, trial)
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    opinions = rng.uniform(0, 1, size=len(nodes))
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    if not edges:
        return opinions

    for _ in range(max_steps):
        e = edges[rng.integers(0, len(edges))]
        i, j = e
        d = opinions[i] - opinions[j]
        if abs(d) < epsilon:
            opinions[i] -= mu * d / 2
            opinions[j] += mu * d / 2
    return opinions


def count_opinion_clusters(opinions: np.ndarray, tol: float = DW_CLUSTER_TOL) -> int:
    """Count clusters of near-identical final opinions (sorted, gap > tol starts a new cluster)."""
    s = np.sort(opinions)
    if len(s) == 0:
        return 0
    gaps = np.diff(s) > tol
    return int(gaps.sum() + 1)


def bounded_confidence_sweep(G: nx.Graph, epsilons: np.ndarray = EPSILON_SWEEP,
                              n_trials: int = N_TRIALS, seed: int = SEED) -> pd.DataFrame:
    """
    Sweep the confidence threshold epsilon; for each value run n_trials
    seeded trials and report mean +/- SD number of final opinion clusters.
    The critical epsilon is where mean clusters drops to (near) 1.
    """
    rows = []
    for eps in epsilons:
        n_clusters = []
        for t in range(n_trials):
            final = deffuant_weisbuch(G, epsilon=eps, seed=seed, trial=t)
            n_clusters.append(count_opinion_clusters(final))
        rows.append({
            "epsilon": float(eps),
            "mean_n_clusters": float(np.mean(n_clusters)),
            "sd_n_clusters": float(np.std(n_clusters)),
            "frac_trials_consensus": float(np.mean(np.array(n_clusters) == 1)),
        })
    return pd.DataFrame(rows)


def find_critical_epsilon(sweep_df: pd.DataFrame, consensus_frac_threshold: float = 0.5) -> float | None:
    """First epsilon (ascending) at which >= consensus_frac_threshold of trials reach a single cluster."""
    hit = sweep_df[sweep_df["frac_trials_consensus"] >= consensus_frac_threshold]
    if hit.empty:
        return None
    return float(hit.iloc[0]["epsilon"])


def si_spread(G: nx.Graph, seed_nodes: list, infection_prob: float = SI_INFECTION_PROB,
              seed: int = SEED, trial: int = 0, max_steps: int = SI_MAX_STEPS) -> dict:
    """
    Simulate SI spreading from seed_nodes. At each step, every infected
    node attempts to infect each susceptible neighbor independently with
    probability infection_prob. Returns time-to-full-coverage (or max_steps
    if not fully covered) and the infection curve (cumulative infected count per step).
    """
    rng = _rng_for_trial(seed, trial)
    infected = set(seed_nodes)
    curve = [len(infected)]
    n = G.number_of_nodes()
    step = 0
    while len(infected) < n and step < max_steps:
        newly = set()
        # sorted(): iterating the set directly makes the order of rng draws
        # depend on PYTHONHASHSEED, which silently breaks reproducibility.
        for u in sorted(infected):
            for v in sorted(G.neighbors(u)):
                if v not in infected and rng.random() < infection_prob:
                    newly.add(v)
        infected |= newly
        curve.append(len(infected))
        step += 1
        if not newly:
            break
    return {
        "time_to_full_coverage": step if len(infected) >= n else None,
        "final_coverage_fraction": len(infected) / n,
        "curve": curve,
    }


def si_seeding_comparison(G: nx.Graph, high_centrality_nodes: list,
                           n_trials: int = N_TRIALS, seed: int = SEED,
                           n_seeds: int = SI_N_SEED_NODES) -> pd.DataFrame:
    """
    Compare SI spreading seeded from the top-`n_seeds` eigenvector-centrality
    nodes vs. `n_seeds` uniformly random nodes, over n_trials seeded trials
    each. Reports mean +/- SD final coverage fraction and time-to-full-coverage
    (excluding trials that never reached full coverage from the mean).
    """
    nodes = list(G.nodes())
    hc_seeds = high_centrality_nodes[:n_seeds]
    rows = []
    for label, fixed_seed_set in [("high_eigenvector_centrality", hc_seeds), ("random", None)]:
        coverage, times = [], []
        for t in range(n_trials):
            if fixed_seed_set is None:
                rng = _rng_for_trial(seed, t)
                seed_nodes = list(rng.choice(nodes, size=n_seeds, replace=False))
            else:
                seed_nodes = fixed_seed_set
            result = si_spread(G, seed_nodes, seed=seed, trial=t)
            coverage.append(result["final_coverage_fraction"])
            if result["time_to_full_coverage"] is not None:
                times.append(result["time_to_full_coverage"])
        rows.append({
            "seeding": label,
            "mean_final_coverage": float(np.mean(coverage)),
            "sd_final_coverage": float(np.std(coverage)),
            "frac_trials_full_coverage": float(len(times) / n_trials),
            "mean_time_to_full_coverage": float(np.mean(times)) if times else float("nan"),
            "sd_time_to_full_coverage": float(np.std(times)) if times else float("nan"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from load import load_clean, zscore_items
    from build import SIM_THRESHOLD, build_threshold_graph, pairwise_complete_cosine
    from metrics import centralities

    coded = load_clean()
    z = zscore_items(coded)
    sim, _ = pairwise_complete_cosine(z)
    G = build_threshold_graph(sim, SIM_THRESHOLD)

    sweep = bounded_confidence_sweep(G, n_trials=10)
    print(sweep.to_string(index=False))
    crit = find_critical_epsilon(sweep)
    print(f"\nCritical epsilon (fragmentation -> consensus): {crit}")

    cent = centralities(G)
    hc_nodes = cent.sort_values("eigenvector", ascending=False)["node"].tolist()
    si_cmp = si_seeding_comparison(G, hc_nodes, n_trials=10)
    print("\nSI spreading, high-centrality vs random seeding:")
    print(si_cmp.to_string(index=False))


SI_HORIZON = 30                   # fixed time horizon for the coverage-curve comparison


def si_curve_comparison(G: nx.Graph, high_centrality_nodes: list,
                        n_trials: int = N_TRIALS, seed: int = SEED,
                        n_seeds: int = SI_N_SEED_NODES,
                        horizon: int = SI_HORIZON) -> pd.DataFrame:
    """
    Same paired comparison as si_seeding_comparison, but reports the whole
    epidemic curve (mean +/- SD coverage at each step) instead of only the
    time to full coverage.

    Needed because on a sparse graph the SI process often never reaches
    every node, which makes "time to full coverage" undefined; coverage at
    a fixed horizon stays well defined on dense and sparse graphs alike.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    hc_seeds = high_centrality_nodes[:n_seeds]
    rows = []

    for label, fixed_seed_set in [("high_eigenvector_centrality", hc_seeds), ("random", None)]:
        curves = np.zeros((n_trials, horizon + 1))
        for t in range(n_trials):
            if fixed_seed_set is None:
                seed_nodes = list(_rng_for_trial(seed, t).choice(nodes, size=n_seeds, replace=False))
            else:
                seed_nodes = fixed_seed_set
            c = si_spread(G, seed_nodes, seed=seed, trial=t, max_steps=horizon)["curve"]
            c = np.array(c, dtype=float) / n
            # the walk stops early once no new node is infected; hold the final value
            curves[t] = np.pad(c, (0, horizon + 1 - len(c)), mode="edge")[: horizon + 1]
        for step in range(horizon + 1):
            rows.append({
                "seeding": label,
                "step": step,
                "mean_coverage": float(curves[:, step].mean()),
                "sd_coverage": float(curves[:, step].std()),
            })
    return pd.DataFrame(rows)


def seed_set_dispersion(G: nx.Graph, seed_nodes: list) -> float:
    """
    Mean shortest-path distance between the chosen seed nodes. A low value
    means the seeds sit on top of each other and largely duplicate each
    other's reach, which is the usual reason centrality-based seeding
    underperforms random seeding.
    """
    pairs = [(u, v) for i, u in enumerate(seed_nodes) for v in seed_nodes[i + 1:]]
    d = [nx.shortest_path_length(G, u, v) for u, v in pairs
         if nx.has_path(G, u, v)]
    return float(np.mean(d)) if d else float("nan")
