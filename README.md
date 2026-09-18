# DPCN_Assignment_1

## Harith - Signed Statement Network (`member_b_statement_net/`)

Nodes = 60 survey statements (T/E/S/V blocks). 
Edges = Pearson correlation of respondents' answers across each pair of statements, sign retained.

- **Construction**: fresh load from `data/raw/` (87 respondents, per shared
  conventions) → per-item z-score → edges kept via permutation test + FDR
  (q=0.05), with a fixed-threshold sensitivity check.
- **Structure**: Louvain communities vs. T/E/S/V blocks (NMI), hub/isolated
  statement ranking, structural balance (triad census, frustration index vs.
  a sign-shuffled null).
- **Dynamics**: Altafini signed-consensus simulation, perturbation-based
  "opinion leverage" ranking.
- **Run it**: `member_b_statement_net/notebook.ipynb` (already executed).
- **Outputs**: `figures/`, `tables/`, `statement_network.gexf`. 
- **Results Log**: `FINDINGS.md`.