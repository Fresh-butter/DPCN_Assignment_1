# Agreement Network Between Students — Findings

**Author: Vishal**  ·  Team: A Shortest Path to an A Grade

Nodes = 87 students, edges = similarity of per-item z-scored opinion vectors.
Full runnable pipeline in `notebook.ipynb`; figures in `figures/`, tables in
`tables/`. All numbers below come from the current execution (N = 87).

## Pipeline / data

- Loads via `common/contract.py`, the single shared data contract: read
  `Survey_Results_UC.csv`, then drop any respondent missing more than
  30 of 60 answers → **87 respondents**. Members B and C import the same
  function, so the three networks are provably describing the same cohort and
  are directly comparable. (An earlier version of this analysis dropped any row
  with a single missing answer, which left only 85 and discarded two
  respondents for no reason; that rule is retired.)
- 60 Likert items (T01–T15, E01–E15, S01–S15, V01–V15) mapped to 1–5;
  `"No Comments"` and blanks → NaN, never Neutral. 62 cells remain missing.
  Pairwise-complete similarity handles them: each pair is scored on the items
  both students answered, requiring at least 20 shared items.
- **Per-item z-scoring is the load-bearing step.** Raw 1–5 vectors have a ~80%
  Agree/Strongly-Agree ceiling, so every pair of students looks alike: raw
  pairwise correlation is **mean 0.267** (min −0.359). After z-scoring,
  pairwise-complete cosine similarity spreads out to **mean −0.010**
  (min −0.638, max 0.758) — a usable signal instead of a near-clique.
  (`fig01_similarity_distribution.png`)

## Network construction (edge-rule sensitivity)

| Rule | Edges | Mean degree | Density | Giant comp. | Isolates |
|---|---|---|---|---|---|
| **threshold (cos > 0.15)** | **822** | **18.9** | **0.220** | 100% | 0 |
| mutual-kNN (k = 8) | 207 | 4.8 | 0.055 | 100% | 0 |
| disparity backbone (α = 0.10) | 1 | 0.02 | 0.000 | 2% | 85 |

- Threshold graph is the primary network for everything below.
- **Jaccard edge overlap between the threshold and kNN rules is only 0.252** —
  the two rules agree on a quarter of their edges while both produce a single
  connected component. *Which* student is "similar" to whom is highly
  sensitive to the construction choice, even though the network's coarse
  properties are not. Worth saying plainly in the report.
- The disparity-filter backbone collapses to a single edge. Each student's
  ~19 edges carry near-uniform weight, so no student has a standout "opinion
  twin". The agreement structure is diffuse and graded, not hub-and-spoke.
  (`table01_edge_rule_comparison.csv`)

## Percolation: where the network falls apart

Sweeping the similarity threshold (`table02_percolation_sweep.csv`,
`fig02_percolation_sweep.png`): the giant component holds at 100% down to
threshold 0.225, loses its first node at 0.25, and drops below 50% between
0.325 and 0.350. The primary threshold 0.15 sits comfortably below the
fragmentation edge — the network is not an artifact of a knife-edge cutoff.

## Structure: degree, clustering, paths

- **Degree** (`fig03_degree_distribution_logbinned.png`): unimodal, range
  **6–32**, median 19, sd 7.3, no isolates, 11 students with degree < 10.
  Not a power law — expected for a threshold graph on roughly normal
  similarity values.
- **Clustering**: average local clustering **0.505**, global transitivity
  **0.586**. Average shortest path **2.06**; one connected component.
- **Degree assortativity +0.278** — well-connected students preferentially
  connect to other well-connected students, consistent with the community
  structure below.

## Communities

- Louvain (seeded) finds **4 communities of sizes [48, 35, 2, 2]**,
  **modularity Q = 0.432**. The two size-2 groups (students 84/43 and 26/116)
  are negligible; the effective structure is **two large opinion blocs**.
  (`table05_communities.csv`, `fig04_network_communities.png`)
- **Community attribute assortativity = +0.821** — edges overwhelmingly stay
  inside their own bloc. **746 of 822 edges (90.8%) are within-community**,
  and within-community edges are stronger (mean weight 0.286 vs 0.212
  between). The blocs are genuinely segregated, not a partition imposed on
  a homogeneous blob.
- Community 1 (35 students) is internally much denser than community 0
  (48 students): mean degree **24.3 vs 15.9**.

  (Community ids are assigned canonically — largest first — so they are stable
  across runs. See the reproducibility note at the end.)

### What the two blocs actually believe

Profiled by comparing per-item mean z-scores between the two blocs
(`table09_community_profile.csv`, **`fig07_community_profile.png`**).

- The split is **not topical — it is a split in intensity**. Both blocs rank
  the same statements highest; they differ in how hard they agree.
  - **Community 1 (n = 35, "high-conviction")**: overall mean z **+0.371**.
  - **Community 0 (n = 48, "moderate agreement")**: overall mean z **−0.283**.
- **The gap is concentrated in Environment and Society/Ethics**: of the 20
  most discriminating items, **11 are V and 7 are S** — only 1 E and 1 T.
  Mean |z-gap| per item by block: **V 0.986, S 0.835, E 0.536, T 0.434**.
- Largest gaps: S09 (inclusive university discourse) 1.30, V10 (consumer
  environmental responsibility) 1.19, S14 (ethical leadership) 1.17,
  V07 (biodiversity) 1.15.
- Framing for the report: these are not "pro- vs anti-sustainability" camps —
  given the ceiling effect everyone leans agree. They are a **fervent
  idealist bloc versus a tepid agreement bloc**, and the divide is almost
  entirely about how strongly one feels about environmental and social
  questions, not about technology or education.

## Hubs and bridges (centrality)

- **Eigenvector hubs** (all in community 1): students 110, 114, 94, 21, 54 —
  degree 30–32, eigenvector 0.205–0.216.
- **Betweenness bridges**: 72, 57, 55, 19, 32. Almost no overlap with the
  hubs — **hubs and bridges are different people**.
- **Student 72 is the standout bridge**: highest betweenness (0.042) on a
  low degree of 15 and a low eigenvector score (0.065). Its neighbours split
  **8 in community 1 and 7 in community 0** — an almost perfectly balanced
  inter-bloc connector. (`table04_centralities.csv`)

## Null model comparison (Erdős–Rényi, matched n and mean degree)

Mean ± SD over 30 seeded realizations (`table06_null_model_comparison.csv`).

| Metric | Observed | Null | z |
|---|---|---|---|
| Average local clustering | **0.505** | ER 0.220 ± 0.009 | **+31.6** |
| Avg shortest path (giant comp.) | 2.054 | ER 1.792 ± 0.011 | +22.8 |
| Giant component fraction | 1.00 | ER 1.00 ± 0.00 | — |
| **Louvain modularity Q** | **0.432** | config. 0.178 ± 0.007 | **+35.5** |

- **Clustering is 2.3× the random-graph null, about 32 SD above chance.**
  This is the single strongest structural result: agreement is strongly
  transitive — if A agrees with B and with C, then B and C tend to agree too.
  Real homophily, not a threshold artifact.
- Average path length is slightly *longer* than the ER null. The network is
  clustered but not shortcut-rich — locally cliquey rather than small-world.
- **The two blocs are not an artifact of running Louvain.** Louvain returns
  *some* partition for any graph, so Q = 0.432 only means something next to
  what a structureless graph with the same degree sequence scores. Rewiring
  the network as a configuration model gives Q = 0.178 ± 0.007, putting the
  observed value **35.5 SD above chance** (`table11_modularity_null.csv`).
  This is the number to quote against member C's co-dissent projection, which
  scores *below* its own null.

## Dynamics 1: bounded-confidence opinion dynamics (Deffuant–Weisbuch)

Sweep over the confidence threshold ε, 30 seeded trials each
(`table07_bounded_confidence_sweep.csv`, `fig05_bounded_confidence_transition.png`).

- ε = 0.05 → **21.7 final opinion clusters**; the population shatters.
- Cluster count falls steadily; **the fragmentation→consensus transition band
  is ε ≈ 0.30–0.55**. The fraction of trials reaching a single cluster goes
  0.13 (ε=0.30) → 0.20 → **0.57 (ε=0.40)** → 0.83 → 0.97 → **1.00 (ε=0.55)**.
- Report it as a *band*, not a single critical value: the transition is
  gradual over ~0.25 in ε, not a sharp threshold.
- Consensus arrives at a moderate ε rather than requiring near-total
  open-mindedness — the network's own dense local structure (clustering 2.3×
  the null) does part of the work of pulling opinions together.

## Dynamics 2: SI spreading — does it matter whom you seed?

3 seeds, 30 seeded trials per condition, per-edge infection probability 0.10.
(`table08_si_seeding_comparison.csv`, `table10_si_coverage_curves.csv`,
`fig06_si_seeding_comparison.png`, **`fig08_si_seeding_density.png`**)

- **On the threshold graph, seeding the three most central students gives no
  meaningful advantage.** Coverage at t = 30 is 0.9985 for high-eigenvector
  seeds vs 0.9950 for random — a 0.35 percentage-point edge on a network where
  both strategies reach essentially everyone. Random seeding actually reaches
  *full* coverage faster when it gets there (7.75 vs 9.11 steps).
- **On the mutual-kNN graph, centrality seeding is actively worse**:
  coverage at t = 30 is **0.244 for high-eigenvector seeds vs 0.487 for
  random** — random seeding reaches twice as many students.
- **Mechanism — seed redundancy.** The top-3 eigenvector students are
  mutually adjacent on both graphs (**mean pairwise seed distance 1.00 hop**,
  vs 2.00 hops for random seeds on the threshold graph and **3.33 hops** on
  the kNN graph). Eigenvector centrality by construction rewards being in the
  same dense core, so the three "best" seeds largely duplicate each other's
  reach.
- **Why density hides it**: on the dense threshold graph every student is
  ~2 hops from everyone, so redundant seeds cost nothing. Sparsify the same
  data to mean degree 4.8 and the redundancy becomes the dominant effect.
  The textbook "seed the hubs" heuristic fails here for a reason that is
  invisible until the network is sparse — a good discussion point for the
  report's dynamics section.

## E02 / T10 / E03 reverse-coding check

- **E02** ("Traditional written examinations accurately measure knowledge"):
  mean 2.84, sd 1.12 — low and unusually spread. Consistent with being
  phrased against its block's grain; treat correlation signs involving E02
  with care.
- **T10** ("Protection of personal data over convenience"): mean 4.18,
  sd 0.88 — high and tight like most T items. No sign of reverse phrasing.
- **E03** ("Class attendance should be compulsory"): mean **2.06**, the
  lowest in the survey. Also member B's main contrarian item.

## Cross-member connections

1. **S/V items dominate.** They drive the two-bloc split here (18 of the top
   20 discriminating items) and are the top strength hubs in member B's
   statement network. Independent confirmation from two different graphs.
2. **Construction choice matters.** Threshold vs kNN Jaccard overlap 0.252
   here; member B's FDR vs fixed-threshold rule changes negative edges from
   11 to 4. Both members found that the coarse structure is robust while the
   specific edges are not — a methodological finding in its own right.
3. **Coherence.** Clustering 2.3× the ER null here and near-zero frustration
   in member B both say the same thing: this class's opinions form an
   unusually consistent system.
4. **Bridges vs deviants.** Student 72 is the structural bridge in *agreement*
   space; member C's deviation network finds a different set of students at
   the top. Being a bridge between opinion blocs is not the same as being
   a contrarian.

## Summary table

| Quantity | Value |
|---|---|
| Respondents (N) | 87 |
| Raw pairwise correlation, mean (min) | 0.267 (−0.359) |
| Z-scored cosine similarity, mean (min, max) | −0.010 (−0.638, 0.758) |
| Threshold graph: edges / mean degree / density | 822 / 18.9 / 0.220 |
| Threshold vs kNN edge Jaccard overlap | 0.252 |
| Disparity backbone edges (of 822) | 1 |
| Giant component ≥ 50% up to threshold | 0.325 |
| Avg local clustering (vs ER null) | 0.505 (vs 0.220 ± 0.009), z = +31.6 |
| Global transitivity | 0.586 |
| Avg shortest path (vs ER null) | 2.054 (vs 1.792 ± 0.011) |
| Degree assortativity | +0.278 |
| Louvain communities / modularity | 4 → [48, 35, 2, 2] / Q = 0.432 |
| Modularity vs configuration null | 0.432 vs 0.178 ± 0.007, z = **+35.5** |
| Community attribute assortativity | +0.821 |
| Within-community edges | 746 / 822 (90.8%) |
| Edge weight within vs between | 0.286 vs 0.212 |
| Bloc mean z (high-conviction / moderate) | +0.371 / −0.283 |
| Mean \|z-gap\| per item by block (T/E/S/V) | 0.434 / 0.536 / 0.835 / 0.986 |
| Deffuant–Weisbuch transition band | ε ≈ 0.30–0.55 |
| SI coverage at t=30, threshold graph (central / random) | 0.9985 / 0.9950 |
| SI coverage at t=30, kNN graph (central / random) | **0.244 / 0.487** |
| Mean pairwise seed distance (central / random, kNN) | 1.00 / 3.33 hops |


## Reproducibility

Every number in this file is produced by `notebook.ipynb` with `SEED = 42`, and
the notebook is byte-for-byte reproducible: two consecutive executions write
identical CSVs.

Getting there required fixing two silent non-determinism bugs that `SEED = 42`
did *not* cover, because they came from Python's hash randomisation rather than
from any RNG:

1. **`si_spread` iterated a `set` of infected nodes.** The order of that
   iteration sets the order of the `rng.random()` draws, so the SI results
   changed between runs even with the seed fixed. Now iterates `sorted(...)`.
2. **Louvain returned sets whose order varied.** The *partition* was stable
   ([48, 35, 2, 2], Q = 0.4320 every time), but the community ids and the row
   order of the exported table shuffled, and re-ordering the rows changed the
   float summation in the per-community means. `detect_communities` now sorts
   the members inside each community and orders communities largest-first.

Worth a sentence in the report's methodology: a fixed seed is necessary but not
sufficient for reproducibility once set iteration order touches a stochastic
path.
