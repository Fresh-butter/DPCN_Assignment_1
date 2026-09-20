# Agreement Network Between Students — Findings

**Author: Vishal**  ·  Team: A Shortest Path to an A Grade

Nodes = 87 students, edges = similarity of per-item z-scored opinion vectors.
Full runnable pipeline in `notebook.ipynb`; figures in `figures/`, tables in
`tables/`. All numbers below come from the current execution (N = 87).

## Pipeline / data

- Loads via `common/contract.py`, the shared data contract: read
  `Survey_Results_UC.csv`, then drop any respondent missing more than
  30 of 60 answers → **87 respondents**. Members B and C import the same
  function, so the three networks describe the same cohort and
  can be directly compared. (An earlier version dropped any row
  with a single missing answer, which left only 85 and threw away two
  respondents for no good reason; we changed that.)
- 60 Likert items (T01–T15, E01–E15, S01–S15, V01–V15) mapped to 1–5;
  `"No Comments"` and blanks → NaN, never Neutral. 62 cells remain missing.
  Pairwise-complete similarity handles them: each pair is scored on the items
  both students answered, needing at least 20 shared items.
- **Z-scoring is what makes this whole analysis work.** Raw 1–5 vectors have a ~80%
  Agree/Strongly-Agree ceiling, so every pair of students looks nearly identical: raw
  pairwise correlation is **mean 0.267** (min −0.359). After z-scoring,
  pairwise-complete cosine similarity spreads out to **mean −0.010**
  (min −0.638, max 0.758) — now we have a usable signal instead of basically a complete graph.
  (`fig01_similarity_distribution.png`)

## Network construction (edge-rule sensitivity)

| Rule | Edges | Mean degree | Density | Giant comp. | Isolates |
|---|---|---|---|---|---|
| **threshold (cos > 0.15)** | **822** | **18.9** | **0.220** | 100% | 0 |
| mutual-kNN (k = 8) | 207 | 4.8 | 0.055 | 100% | 0 |
| disparity backbone (α = 0.10) | 1 | 0.02 | 0.000 | 2% | 85 |

- Threshold graph is the primary network for everything below.
- **Jaccard edge overlap between the threshold and kNN rules is only 0.252** —
  the two rules only agree on about a quarter of their edges, even though both
  give a single connected component. *Which* student is "similar" to whom
  changes a lot depending on the construction choice, even though the big-picture
  properties don't.
- The disparity-filter backbone collapses to a single edge. Each student's
  ~19 edges carry roughly equal weight, so no student has a clear "opinion
  twin". The agreement structure is spread out and gradual, not hub-and-spoke.
  (`table01_edge_rule_comparison.csv`)

## Percolation: where the network falls apart

Sweeping the similarity threshold (`table02_percolation_sweep.csv`,
`fig02_percolation_sweep.png`): the giant component stays at 100% down to
threshold 0.225, loses its first node at 0.25, and drops below 50% between
0.325 and 0.350. Our primary threshold of 0.15 sits comfortably below where
fragmentation starts — the network isn't just barely holding together.

## Structure: degree, clustering, paths

- **Degree** (`fig03_degree_distribution_logbinned.png`): unimodal, range
  **6–32**, median 19, sd 7.3, no isolates, 11 students with degree < 10.
  Not a power law — which makes sense for a threshold graph on roughly normal
  similarity values.
- **Clustering**: average local clustering **0.505**, global transitivity
  **0.586**. Average shortest path **2.06**; one connected component.
- **Degree assortativity +0.278** — well-connected students tend to connect
  to other well-connected students, which fits with the community
  structure below.

## Communities

- Louvain (seeded) finds **4 communities of sizes [48, 35, 2, 2]**,
  **modularity Q = 0.432**. The two size-2 groups (students 84/43 and 26/116)
  are tiny; the real structure is **two large opinion blocs**.
  (`table05_communities.csv`, `fig04_network_communities.png`)
- **Community attribute assortativity = +0.821** — edges mostly stay
  inside their own bloc. **746 of 822 edges (90.8%) are within-community**,
  and within-community edges are stronger (mean weight 0.286 vs 0.212
  between). The blocs are genuinely separated, not a random partition
  imposed on a uniform blob.
- Community 1 (35 students) is internally much denser than community 0
  (48 students): mean degree **24.3 vs 15.9**.

  (Community ids are assigned canonically — largest first — so they're stable
  across runs. See the reproducibility note at the end.)

### What the two blocs actually believe

Profiled by comparing per-item mean z-scores between the two blocs
(`table09_community_profile.csv`, **`fig07_community_profile.png`**).

- The split is **not by topic — it's a split in intensity**. Both blocs rank
  the same statements highest; they differ in how strongly they agree.
  - **Community 1 (n = 35, "high-conviction")**: overall mean z **+0.371**.
  - **Community 0 (n = 48, "moderate agreement")**: overall mean z **−0.283**.
- **The gap is concentrated in Environment and Society/Ethics**: of the 20
  most discriminating items, **11 are V and 7 are S** — only 1 E and 1 T.
  Mean |z-gap| per item by block: **V 0.986, S 0.835, E 0.536, T 0.434**.
- Largest gaps: S09 (inclusive university discourse) 1.30, V10 (consumer
  environmental responsibility) 1.19, S14 (ethical leadership) 1.17,
  V07 (biodiversity) 1.15.
- These aren't "pro- vs anti-sustainability" camps — given the ceiling effect
  everyone leans agree. It's more like a fervent bloc and a temperate one, and the
  divide is almost entirely about how strongly one holds environmental and
  social positions, not about technology or education.

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

- Clustering is 2.3× the random-graph null, about 32 SD above chance.
  Agreement is strongly transitive — if A agrees with B and with C, then B and
  C usually agree too. That's real homophily, not just an artifact of the threshold.
- Average path length is slightly *longer* than the ER null. The network is
  clustered but doesn't have many shortcuts — locally cliquey rather than small-world.
- **The two blocs aren't just an artifact of Louvain.** Louvain finds
  *some* partition for any graph, so Q = 0.432 only means something compared to
  what a random graph with the same degree sequence scores. Rewiring the
  network as a configuration model gives Q = 0.178 ± 0.007, putting the
  observed value **35.5 SD above chance** (`table11_modularity_null.csv`).
  Member C's co-dissent projection, on the same 87 students, scores *below*
  its own null — see `FINDINGS_CROSS.md` §3.

## Dynamics 1: bounded-confidence opinion dynamics (Deffuant–Weisbuch)

Sweep over the confidence threshold ε, 30 seeded trials each
(`table07_bounded_confidence_sweep.csv`, `fig05_bounded_confidence_transition.png`).

- ε = 0.05 → **21.7 final opinion clusters**; the population shatters.
- Cluster count falls steadily; **the transition from fragmentation to consensus
  happens around ε ≈ 0.30–0.55**. The fraction of trials reaching a single cluster goes
  0.13 (ε=0.30) → 0.20 → **0.57 (ε=0.40)** → 0.83 → 0.97 → **1.00 (ε=0.55)**.
- It's a band, not a sharp threshold: the transition is spread over ~0.25 in ε.
- Consensus happens at a moderate ε rather than needing near-total
  open-mindedness — the network's dense local structure (clustering 2.3×
  the null) helps pull opinions together on its own.

## Dynamics 2: SI spreading — does it matter whom you seed?

3 seeds, 30 seeded trials per condition, per-edge infection probability 0.10.
(`table08_si_seeding_comparison.csv`, `table10_si_coverage_curves.csv`,
**`fig08_si_seeding_density.png`**)

- **On the threshold graph, seeding the top-centrality students gives basically
  no advantage.** Coverage at t = 30 is 0.9985 for high-eigenvector
  seeds vs 0.9950 for random — a tiny difference on a network where
  both strategies reach essentially everyone. Random seeding actually reaches
  *full* coverage faster when it gets there (7.75 vs 9.11 steps).
- **On the mutual-kNN graph, centrality seeding is actually worse**:
  coverage at t = 30 is **0.244 ± 0.266 for high-eigenvector seeds vs
  0.487 ± 0.401 for random** (mean ± SD over 30 trials) — random seeding
  reaches roughly twice as many students on average.
  **The spread matters as much as the gap.** On a graph this sparse an
  outbreak either catches or dies out, so the SD is as large as the mean
  (coefficient of variation 1.09 and 0.82) and individual trials range from
  near-zero to near-full coverage. The difference in means is still
  solid: Welch t = 2.76 on 30 trials per arm, p = 0.008. But the ±SD bands in
  `fig08_si_seeding_density.png` overlap heavily and the figure should be
  read that way.
- **Why this happens — seed redundancy.** The top-3 eigenvector students are
  right next to each other on both graphs (**mean pairwise seed distance 1.00 hop**,
  vs 2.00 hops for random seeds on the threshold graph and **3.33 hops** on
  the kNN graph). Eigenvector centrality rewards being in the
  same dense core, so the three "best" seeds basically cover the same
  area.
- **Why density hides it**: on the dense threshold graph every student is
  ~2 hops from everyone, so having redundant seeds doesn't cost anything. Make the
  graph sparser and the redundancy becomes the main effect.
  The textbook "seed the hubs" idea fails here, and it fails for a reason
  that's invisible until the network is sparse.

## E02 / T10 / E03 reverse-coding check

- **E02** ("Traditional written examinations accurately measure knowledge"):
  mean 2.84, sd 1.12 — low and unusually spread. Consistent with being
  phrased against its block's grain; be careful with correlation signs involving E02.
- **T10** ("Protection of personal data over convenience"): mean 4.18,
  sd 0.88 — high and tight like most T items. No sign of reverse phrasing.
- **E03** ("Class attendance should be compulsory"): mean **2.06**, the
  lowest in the survey. Also member B's main contrarian item.

## Cross-member connections

1. **S/V items dominate.** They drive the two-bloc split here (18 of the top
   20 discriminating items) and are the top strength hubs in member B's
   statement network. Two different graphs, same conclusion.
2. **Construction choice matters.** Threshold vs kNN Jaccard overlap 0.252
   here; member B's FDR vs fixed-threshold rule changes negative edges from
   11 to 4. Both of us found that the big picture is stable while the
   specific edges aren't — which is a finding in itself.
3. **Coherence.** Clustering 2.3× the ER null here and near-zero frustration
   in member B both point to the same thing: this class's opinions are
   unusually consistent.
4. **Bridges vs deviants.** Student 72 is the structural bridge in *agreement*
   space; member C's deviation network finds completely different students at
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
| SI coverage at t=30, kNN graph (central / random) | **0.244 ± 0.266 / 0.487 ± 0.401** (Welch t = 2.76, p = 0.008) |
| Mean pairwise seed distance (central / random, kNN) | 1.00 / 3.33 hops |


## Reproducibility

Every number here is produced by `notebook.ipynb` with `SEED = 42`, and two consecutive
executions write identical CSVs. Getting that to work required fixing two places where a Python
`set` was iterated — in `si_spread` and in the Louvain wrapper — because per-process hash
randomisation changed the order of random draws and of float summation even with the seed
fixed. Both now iterate `sorted(...)`, and communities are ordered largest-first so their
ids are stable across runs.
