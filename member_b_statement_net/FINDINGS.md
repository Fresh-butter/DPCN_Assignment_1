# Signed Statement Network: Findings

**Author: Harith**  ·  Team: A Shortest Path to an A Grade

See `notebook.ipynb` for the full pipeline. Figures in `figures/`, tables in `tables/`.

## Pipeline / data

- Loaded via `common/contract.py`:
  `Survey_Results_UC.csv` → Likert 1-5 → drop rows with >30 of 60
  missing → **87 respondents**. `src/load.py` just re-indexes the result 0..86,
  since this network's nodes are statements and we don't use respondent identity.
- 60 items kept: T01-T15, E01-E15, S01-S15, V01-V15.
- Missing-data sanity check: pairwise-complete correlation vs.
  mean-imputed z-scored correlation are nearly identical (mean |Δr| = 0.0033,
  max = 0.069). There's so little missing data that the handling choice
  doesn't matter much. Used pairwise-complete as the primary. (`src/build.py`)
- Raw response distribution confirms the ceiling effect:
  **80.0% of the 5158 answers actually given are Agree/Strongly Agree**
  (4127 of 5158). Per-item means cluster between 3.5 and 4.5;
  only a few items (E02 and E03 mainly) sit near or below the midpoint.
  (`fig01_construction.png`)

## Distributions

- **Correlation distribution vs permutation null** (`fig01_construction.png`):
  observed |r| distribution across all 1770 pairs sits clearly to the
  right of the null (each item independently permuted), which is
  direct evidence the network picks up real signal.
  The FDR-significant subset (402 pairs) is the right tail, cut where it stops
  separating from the null (~|r|=0.27).
- **Degree**: mean 13.4, range 0-32, 2 isolates. No power-law hub,
  which isn't surprising at 60 nodes. E03 is the one statement with a lot of
  negative edges (8 of them); every other statement has at most 1-2.
- **Node strength** is mildly bimodal: a low-strength group (0-3, mostly
  T-block) and a broader higher-strength spread (5-12, mainly S/V),
  matching the hub ranking. Both are shown in `fig03_structure.png`.

## Network construction (edge-rule sensitivity)

- Primary rule: permutation-null + Benjamini-Hochberg FDR (q=0.05) on
  per-pair empirical p-values → 402 edges (391 positive, 11 negative),
  density 0.227, 2 isolated nodes (T08, T09). Smallest |r| retained is 0.274.
- We first tried a family-wise max-statistic permutation threshold
  (alpha=0.01, controlling the max |r| across all 1770 pairs). That gave
  |r| > 0.50 and kept zero negative edges, which would've made the whole
  signed network idea pointless. So we switched to per-edge FDR.
- Sensitivity check: fixed |r| > 0.30 cutoff → 316 edges (312 positive, 4
  negative), density 0.179, 4 isolates. The fixed threshold keeps way fewer
  negative edges (4 vs 11). The construction choice matters most right where
  the interesting signed structure is.
  (`tables/table00_edge_rule_sensitivity.csv`, `fig02_signed_network.png`)

## Communities vs the survey's T/E/S/V design

- Louvain on the positive-edge backbone finds 8 communities — so the match
  with the survey's T/E/S/V labels is pretty weak, NMI = 0.277.
- The class doesn't organise its opinions along the survey's topic axes.
  E.g. community 4 mixes 5 Education + 8 Society + 2 Environment items;
  community 3 pulls from all four blocks.
  (`fig03_structure.png`, `tables/table02*.csv`)

## Hubs and isolated statements

- Highest total agreement-strength (hubs): S09 (inclusive university
  environment), V07 (protecting biodiversity), S14 (ethical
  leadership), V09, V10 — broad consensus statements that are most
  strongly connected to the rest of the survey.
- Fully isolated (no significant correlation with anything): T08
  ("AI-assisted diagnosis should become routine in healthcare") and T09
  ("Autonomous vehicles will ultimately make transportation safer"). These
  two specific-technology items just don't connect to anything.
  Near-isolated: T15, E13, E06, T07, T05.
  (`fig03_structure.png`, `tables/table03_hub_ranking.csv`)

## Structural balance / frustration

- Only 11/402 edges (2.7%) are negative, but they're not random: 8 of
  the 11 touch E03 ("Class attendance should be compulsory for all courses").
  E03 anti-correlates with a cluster of progressive-education/innovation items
  (E04, E07, E10, E11, E12, T06, S06) — a clear "traditionalist vs progressive
  pedagogy" axis.
- Closed-triad census: 0 frustrated triangles out of 1298 closed triads.
  Every triangle that exists is locally balanced.
- Frustration index (minimum edge-sign flips for a two-faction split, greedy
  local search, 50-300 restarts, stable across restart counts):
  2 / 402 edges (0.5%). Since the triangle census is already perfect,
  those 2 edges must sit on a longer cycle that triangles can't catch.
- Null model (reshuffle which of the 402 edges carry the 11 negative signs,
  keep topology fixed): mean frustration 10.98 ± 0.16 → observed 2
  is 56 SD below chance (z = -56.2). The disagreement in this class is
  placed very coherently.
  (`fig03_structure.png`, `tables/table05_balance_summary.csv`)
- Minority faction in the best 2-coloring (excluding T08/T09, isolated):
  T01, E02, E03 vs the other 55 connected statements.
  T01 ("AI will improve society more than create problems") sits opposite
  T12 ("Governments should introduce stricter regulations for AI", r=-0.28) —
  tech-optimism trades off against wanting regulation. E02 and E03
  cluster together (r=+0.28) as the "traditionalist" minority view.
  (`tables/table04_factions.csv`)

### E02 / T10 reverse-coding check

- **E02** ("Traditional written examinations accurately measure a student's
  knowledge") is a genuine contrarian item, exactly as flagged. It
  anti-correlates with E01 and sits in the 3-item minority faction.
- **T10** ("Protection of personal data is more important than
  technological convenience") is not an outlier here. It sits in
  the majority faction, correlating positively with T11, T12, T14, S05, S12,
  V10 (all regulation/privacy items). It reads as a natural part of the
  "cautious about tech" cluster rather than the opposite of it.

## Dynamics

- **Altafini signed consensus** (`dx/dt = -L_signed x`, seeded with each
  item's real deviation from the Likert midpoint): mean |opinion| decays
  from 1.215 to 0.080 over t ∈ [0, 120] — slow decay toward zero rather
  than a stable split or fast collapse. Makes sense: a graph that's almost
  perfectly balanced (frustration 2/402) is predicted by Altafini's theory to
  eventually decay to zero, and the decay rate depends on how much frustration
  there is — very little here, so very slow.
  (`fig04_dynamics.png`)
- **Perturbation / opinion-leverage ranking** (nudge one item, measure
  how much signed movement it propagates at t=1): leverage
  is mainly about how central a statement is in the positive backbone.
  Top: S09, V10, S14, V13, V15 (all 0.966-0.967). Bottom: T15 (0.233),
  T01 (0.253), E04 (0.340), E13 (0.399), E02 (0.416). Being well-connected
  matters more for spreading influence than being a source of disagreement.
  (`fig04_dynamics.png`, `tables/table06_leverage_ranking.csv`)

### Leverage grouped by block

(`fig04_dynamics.png`, `tables/table07_leverage_by_block.csv`)

| Block | n | mean leverage | SD | min | max |
|---|---|---|---|---|---|
| Technology | 13 | 0.714 | 0.249 | 0.233 | 0.953 |
| Education | 15 | 0.762 | 0.235 | 0.340 | 0.963 |
| Society/Ethics | 15 | **0.933** | 0.054 | 0.785 | 0.967 |
| Environment | 15 | **0.957** | **0.014** | 0.914 | 0.967 |

(Technology shows 13, not 15: T08 and T09 are isolates.)

- **S and V dominate leverage, and it's a big gap** — mean 0.93/0.96 vs
  0.71/0.76 for T and E.
- **The spread is interesting too.** Environment items have an SD
  of just **0.014** — all fifteen fall between 0.914 and 0.967.
  Technology's SD is **18× larger** (0.249), ranging from 0.233 to 0.953.
  Environment behaves as one tightly-coupled bloc; Technology is a bunch of
  statements that happen to share a label.
- Member A finds the same thing from the other side — two opinion blocs
  separated almost entirely by V and S items (mean |z-gap|: V 0.986,
  S 0.835 vs E 0.536, T 0.434). Two different graphs, same ordering:
  **V > S > E > T**. Quantified in `FINDINGS_CROSS.md` §1 (Spearman ρ = +0.784
  across the 58 statements both measures score).

## Reproducibility

`notebook.ipynb` runs end to end and reproduces every number here. Re-running
writes identical CSVs. Both graphs are exported to `gexf/` with block,
community, degree, strength and faction as node attributes.

## Figures

4 multi-panel figures, all using the shared palette
(`src/viz.py`, blocks always T = blue, E = orange, S = aqua, V = yellow).

| Figure | What it shows |
|---|---|
| `fig01_construction.png` | How we built the network. The 80% ceiling, and the permutation-null + FDR cut at \|r\| = 0.274 that picks the 402 edges. |
| `fig02_signed_network.png` | What it looks like. Full layout (S/V dense core, T on the rim) and the 11 negative edges drawn alone. |
| `fig03_structure.png` | How it's organised. Block vs community (NMI = 0.277), hubs vs isolates, and frustration against a sign-shuffled null. |
| `fig04_dynamics.png` | Dynamics. Altafini decay 1.21 -> 0.08, and leverage by block. |
