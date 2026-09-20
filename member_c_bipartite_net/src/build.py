"""
build.py -- member_c_bipartite_net

Bipartite deviation network:
  - Left nodes: 87 students (respondent IDs)
  - Right nodes: 60 statements (item codes T01..V15)
  - Edge (student s, item q): |z_sq| > cutoff  (student deviated from consensus on that item)

Also builds:
  - One-mode student-student projection (co-dissent network, edge weight = shared contrarian items)
  - One-mode statement-statement projection (issue co-deviation, edge weight = shared contrarian students)
  - Sensitivity sweep over cutoff values
  - GEXF export
"""

from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd

# Make load.py importable when run from repo root or src/
sys.path.insert(0, str(Path(__file__).parent))
from load import load_clean, zscore_items, zscore_mean_imputed, SEED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIMARY_CUTOFF = 1.0        # |z| > 1.0 is "notably non-consensus"
SENSITIVITY_CUTOFFS = [0.75, 1.0, 1.5]   # dense, primary, sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMBER_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = MEMBER_DIR / "figures"
TABLES_DIR = MEMBER_DIR / "tables"
GEXF_DIR = MEMBER_DIR / "gexf"


def build_bipartite(z: pd.DataFrame, cutoff: float = PRIMARY_CUTOFF) -> nx.Graph:
    """
    Build bipartite graph.
    - Student nodes: type='student'
    - Item nodes:   type='item', block=T/E/S/V
    - Edge: |z_sq| > cutoff, weight=|z_sq|
    """
    G = nx.Graph()

    students = list(z.index)
    items = list(z.columns)

    # Add student nodes
    for s in students:
        G.add_node(s, bipartite=0, node_type="student")

    # Add item nodes
    for q in items:
        block = q[0]  # T/E/S/V
        G.add_node(q, bipartite=1, node_type="item", block=block)

    # Add edges
    arr = z.to_numpy()
    for i, s in enumerate(students):
        for j, q in enumerate(items):
            val = arr[i, j]
            if np.isnan(val):
                continue
            abs_z = abs(val)
            if abs_z > cutoff:
                G.add_edge(s, q, weight=float(abs_z), z_signed=float(val))

    return G


def bipartite_summary(G: nx.Graph) -> dict:
    students = [n for n, d in G.nodes(data=True) if d.get("node_type") == "student"]
    items = [n for n, d in G.nodes(data=True) if d.get("node_type") == "item"]

    student_degs = [G.degree(s) for s in students]
    item_degs = [G.degree(q) for q in items]

    return {
        "n_students": len(students),
        "n_items": len(items),
        "n_edges": G.number_of_edges(),
        "density": G.number_of_edges() / (len(students) * len(items)) if students and items else 0,
        "student_deg_mean": float(np.mean(student_degs)) if student_degs else 0,
        "student_deg_max": int(max(student_degs)) if student_degs else 0,
        "item_deg_mean": float(np.mean(item_degs)) if item_degs else 0,
        "item_deg_max": int(max(item_degs)) if item_degs else 0,
        "isolated_students": sum(1 for d in student_degs if d == 0),
        "isolated_items": sum(1 for d in item_degs if d == 0),
    }


def sensitivity_sweep(z: pd.DataFrame, cutoffs: list = SENSITIVITY_CUTOFFS) -> pd.DataFrame:
    """Build the bipartite graph at each cutoff and record summary stats."""
    rows = []
    for c in cutoffs:
        G = build_bipartite(z, cutoff=c)
        s = bipartite_summary(G)
        s["cutoff"] = c
        rows.append(s)
    return pd.DataFrame(rows)[["cutoff", "n_edges", "density",
                                "student_deg_mean", "item_deg_mean",
                                "isolated_students", "isolated_items"]]


def student_projection(G: nx.Graph) -> nx.Graph:
    """
    Weighted one-mode projection onto students.
    Edge weight = number of items where BOTH students deviate (co-dissent count).
    Uses networkx's bipartite projection with weight='weight' summed over shared items.
    """
    # sorted(), not a set: a set's iteration order depends on PYTHONHASHSEED,
    # which changes the node insertion order of P and makes Louvain on the
    # projection non-reproducible between runs.
    students = sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "student")
    return nx.bipartite.weighted_projected_graph(G, students)


def item_projection(G: nx.Graph) -> nx.Graph:
    """
    Weighted one-mode projection onto items.
    Edge weight = number of students who deviate on BOTH items.
    """
    items = sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "item")
    return nx.bipartite.weighted_projected_graph(G, items)


def export_gexf(G: nx.Graph, name: str) -> Path:
    """Export graph to GEXF (Gephi-compatible). Returns path."""
    GEXF_DIR.mkdir(parents=True, exist_ok=True)
    path = GEXF_DIR / f"{name}.gexf"
    nx.write_gexf(G, str(path))
    return path


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    GEXF_DIR.mkdir(parents=True, exist_ok=True)

    clean = load_clean()
    z = zscore_items(clean)
    print(f"Loaded: {clean.shape[0]} respondents x {clean.shape[1]} items")
    print(f"Z-score NaN count: {z.isna().sum().sum()}")

    # Primary bipartite graph
    G = build_bipartite(z, cutoff=PRIMARY_CUTOFF)
    print(f"\nPrimary bipartite (cutoff={PRIMARY_CUTOFF}):")
    for k, v in bipartite_summary(G).items():
        print(f"  {k}: {v}")

    # Sensitivity
    sweep = sensitivity_sweep(z)
    print(f"\nSensitivity sweep:")
    print(sweep.to_string(index=False))
    sweep.to_csv(TABLES_DIR / "sensitivity_sweep.csv", index=False)

    # Projections
    Ps = student_projection(G)
    Pi = item_projection(G)
    print(f"\nStudent projection: {Ps.number_of_nodes()} nodes, {Ps.number_of_edges()} edges")
    print(f"Item projection:    {Pi.number_of_nodes()} nodes, {Pi.number_of_edges()} edges")

    # GEXF exports
    p1 = export_gexf(G, "bipartite_primary")
    p2 = export_gexf(Ps, "student_codissent_projection")
    p3 = export_gexf(Pi, "item_codeviation_projection")
    print(f"\nGEXF exported:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
