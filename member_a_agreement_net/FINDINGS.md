# Agreement Network Between Students: Findings

Running log, most recent at bottom of each section. See `notebook.ipynb` for
the full runnable pipeline. Figures in `figures/`, tables in `tables/`.

## Pipeline / data

- By explicit user instruction, this analysis reads
  `data/clean/cleaned_Survey_Results_UC.csv` (NOT `data/raw/`). The cleaning
  script `data/clean/clean_data.py` drops any row with a single blank cell,
  leaving **85 respondents** — not the 87 from the team's >30-missing rule
  used by member_b. This is a genuine, documented divergence; it must be
  called out in the report's cross-member comparison. See `src/load.py`.
- 60 Likert items kept: T01–T15, E01–E15, S01–S15, V01–V15. Likert text
  mapped to 1–5; `"No Comments"` mapped to NaN (never to Neutral/3).
  37 `"No Comments"` cells remain across 26 items; pairwise-complete
  similarity handles them (only jointly-answered items used for each pair).
- Per-item z-scoring (population std, ddof=0) is essential: raw 1–5 vectors
  suffer from a strong ceiling effect (~79 % Agree/Strongly Agree). Raw
  pairwise correlation between respondents: **mean 0.274, min −0.359** —
  compressed toward agreement. After z-scoring, pairwise-complete cosine
  similarity spreads to **mean −0.010, min −0.634, max 0.752**.
  (`fig01_similarity_distribution.png`)

## Network construction (edge-rule sensitivity)

- Primary rule: cosine similarity on z-scored vectors > 0.15 → **775 edges**,
  density 0.217, mean degree 18.2, single connected component (100 % giant
  component). (`table01_edge_rule_comparison.csv`, `fig01_similarity_distribution.png`)
- Sensitivity check #1: **mutual-kNN** (k = 8) → only 203 edges, mean degree
  4.8, still one connected component. Jaccard edge overlap with the threshold
  graph is only **0.262**, i.e. the two rules agree on barely a quarter of
  their edges. Which student is "similar" to whom is fairly sensitive to
  construction choice.
- Sensitivity check #2: **disparity-filter backbone** (Serrano et al. 2009,
  α = 0.10, applied to the threshold graph) → collapses to just **2 edges**,
  isolating 81 of 85 nodes. Each student's ~18 edges have fairly homogeneous
  weight — no student has one standout "opinion twin." The agreement
  network's weight structure is diffuse/graded, not hub-and-spoke.
- Jaccard overlap threshold vs. backbone: **0.003**.
  (`table01_edge_rule_comparison.csv`)

## Percolation: giant-component emergence

- Sweeping the similarity threshold from −0.30 to 0.525
  (`table02_percolation_sweep.csv`, `fig02_percolation_sweep.png`):
  the giant component remains at 100 % down to threshold 0.225, begins
  fragmenting around 0.25, and drops below 50 % between 0.325 (GC = 74.1 %)
  and 0.350 (GC = 38.8 %). The primary threshold 0.15 is deliberately below
  this fragmentation edge — worth noting in the report.

## Structure: degree, clustering, paths

- **Degree distribution** (`fig03_degree_distribution_logbinned.png`,
  `table03_degree_histogram_logbinned.csv`): unimodal, range 6–32 (min: node
  26, max: node 110), median 18, std 6.96. No isolates. Not a power law —
  consistent with a threshold graph on approximately normal similarity values.
  11 nodes have degree < 10.
- **Clustering**: average local clustering **0.495**, global transitivity
  **0.574**. Average shortest path in the (whole) giant component: **2.06**.
  Single connected component.
- **Degree assortativity**: **+0.278** — positively assortative, i.e.
  high-degree students preferentially connect to other high-degree students.
  Consistent with the community structure (community 2 has mean degree 23.3
  vs. community 1's 15.4).

## Communities

- Louvain finds **4 communities**, sizes **[46, 35, 2, 2]**. Modularity
  **Q = 0.428**, a solid community signal. The two size-2 groups are
  negligible; the effective structure is two large, roughly balanced opinion
  blocs plus 4 peripheral students.
  (`table05_communities.csv`, `fig04_network_communities.png`)
- **Edge-weight patterns**: 702/775 edges (90.6 %) are within-community;
  within-community mean weight 0.284 vs. between-community 0.211 (35 %
  stronger within). All 5 strongest edges — (94,110) at 0.752 down to
  (41,50) at 0.658 — are within community 2.

### What the two communities actually believe

- Profiled the two large communities by comparing per-item z-score means.
  The split is not topical — both communities' top-agreement items are the
  same statements. It is a split in **intensity of agreement**. 19 of the top
  20 most divisive items (by z-score gap) come from the V (Environment) and S
  (Society/Ethics) blocks; essentially none from T or E.
- **Community 2 (35 students, "high-conviction" bloc)**: mean z-score +0.365.
  Near-ceiling Strongly Agree responses (raw means 4.6–4.9/5) on V07
  (biodiversity), V10 (consumer environmental responsibility), S09 (inclusive
  university discourse), S14 (ethical leadership), S15 (public trust in tech).
- **Community 1 (46 students, "moderate agreement" bloc)**: mean z-score
  −0.286. Agrees with the same statements but noticeably less strongly (raw
  means 3.6–4.2, Agree/Neutral-leaning).
- Framing for the report: the two Louvain communities are not "pro- vs.
  anti-sustainability" camps — everyone leans agree given the ceiling effect —
  they are a "fervent idealist" bloc vs. a "tepid agreement" bloc, and the gap
  is concentrated almost entirely in the Environment and Society/Ethics blocks.

## Hubs and bridges (centrality)

- Highest **eigenvector centrality** (hubs, all in community 2):
  110, 114, 94, 81, 54 — degree 29–32, eigenvector scores 0.210–0.223.
  These are the students most embedded in the high-conviction community.
- Highest **betweenness centrality** (bridges): 72, 57, 55, 19, 82 —
  almost no overlap with the eigenvector top 5. Hubs and bridges are
  different people. (`table04_centralities.csv`)
- **Node 72** is the standout bridge: highest betweenness (0.040) despite a
  low degree (14) and low eigenvector score (0.061). It belongs to community 1
  and has exactly 7 neighbors in community 1 and 7 in community 2 — a nearly
  perfect 50/50 inter-community bridge. This distinction (bridges vs. hubs)
  is exactly the kind of observation the report's centrality section should
  highlight.
- **Community 2** (35 students, "high conviction") has mean degree 23.3 vs.
  community 1's 15.4 — the fervent bloc is internally much denser, consistent
  with strong opinions creating more mutual similarity.

## Null model comparison (vs. Erdős–Rényi, matched n and mean degree)

- Mean ± SD over 30 seeded ER realizations.
  (`table06_null_model_comparison.csv`)
- **Observed clustering (0.495) is more than double the ER null
  (0.218 ± 0.009)** — about 30 standard deviations above chance. This is the
  single strongest structural finding: real opinion similarity produces much
  more triadic closure ("my two agreement-partners tend to agree with each
  other too") than random-graph chance alone. Clear evidence of non-random,
  homophily-driven structure.
- Observed average shortest path (2.06) is slightly *longer* than the ER null
  (1.80 ± 0.01) — small-world-ish but not dramatically so; the network is a
  bit more "clustered but not shortcut-rich" than a comparable random graph.
- Giant component fraction matches the null exactly (both 1.0) at this
  density — unsurprising, since mean degree ≈ 18 on n = 85 is well above the
  ER connectivity threshold (~ln 85 ≈ 4.4).

## Dynamics 1: bounded-confidence opinion dynamics (Deffuant–Weisbuch)

- Sweep over ε, mean ± SD final opinion-cluster count over 30 seeded trials
  per ε. (`table07_bounded_confidence_sweep.csv`,
  `fig05_bounded_confidence_transition.png`)
- At low confidence (ε = 0.05) the network fragments heavily: mean **21.9
  final clusters**. As ε increases, cluster count falls steadily.
- **Critical ε ≈ 0.40**: this is the first threshold at which ≥50 % of trials
  converge to a single consensus cluster (fraction of consensus trials climbs
  0.33 → 0.60 between ε = 0.35 and ε = 0.40, then 0.87 → 0.97 → 1.0 by
  ε = 0.60). Below ε ≈ 0.30 essentially no trial reaches consensus; above
  ε ≈ 0.60 every trial does. This gives a fairly narrow **transition band
  (≈ 0.30–0.60)** rather than a knife-edge single critical value — worth
  reporting as a *range*, not a single number.
- Because the underlying agreement network already has dense local structure
  (high clustering vs. the ER null), consensus is reached at a moderate ε
  rather than requiring near-total open-mindedness (ε close to 1) — the
  network's structure itself facilitates convergence.

## Dynamics 2: SI spreading, high-eigenvector-centrality vs. random seeding

- 30 seeded trials per condition, 3 seed nodes each.
  (`table08_si_seeding_comparison.csv`, `fig06_si_seeding_comparison.png`)
- **Counter-intuitive result: no meaningful advantage for high-centrality
  seeding.** Mean time-to-full-coverage was effectively the same for both
  strategies (9.0 vs. 8.6 steps), and final coverage fraction was
  statistically indistinguishable (both 0.997). Random seeding even
  converged marginally faster.
- This makes sense given the network's structure: mean degree ≈ 18 out of 84
  possible neighbors (density 0.217) and average path length ≈ 2.06 — the
  network is dense and compact enough that *any* 3 seed nodes are already
  close to everyone, so seed choice barely matters. High-eigenvector seeding
  would likely show a larger effect on the sparser mutual-kNN graph (mean
  degree 4.8) — a natural sensitivity check for the report's discussion.

## E02 / T10 reverse-coding check

- **E02** ("Traditional written examinations accurately measure a student's
  knowledge"): mean = 2.81, sd = 1.12 — noticeably lower and more spread than
  most items (most items skew high per the ceiling effect). Consistent with
  E02 being phrased against the grain of its block. Correlation signs
  involving E02 should be treated with caution.
- **T10** ("Protection of personal data…"): mean = 4.19, sd = 0.89 — high and
  low-variance like most technology items, i.e. it does *not* show obvious
  signs of reverse phrasing. Correlation signs involving T10 can likely be
  interpreted at face value.
- **E03** ("Class attendance should be compulsory for all courses"):
  mean = 2.02 — the *lowest* mean in the entire survey. If reverse-coding is
  a concern for E02, E03 deserves the same attention. This is also flagged as
  a key contrarian item in member_b's signed statement network.

## Open questions / next steps

- Run the SI spreading comparison on the sparser mutual-kNN network to test
  whether high-centrality seeding shows a larger advantage when the network
  is less dense.
- Cross-check the two communities against member_b's block-level clusters:
  do the same S/V items that drive the community split here also anchor the
  hub/strength ranking in the statement network?
- The two size-2 mini-communities (nodes 84/43 and 116/26) may be worth
  profiling: are these students genuinely distinctive, or artifacts of the
  threshold?

## Summary of concrete numbers for the report

| Quantity | Value |
|---|---|
| Respondents (N) | 85 (cleaned-file path, not 87) |
| Raw pairwise correlation, mean (min) | 0.274 (−0.359) |
| Z-scored cosine similarity, mean (min, max) | −0.010 (−0.634, 0.752) |
| Threshold graph edges / mean degree / density | 775 / 18.2 / 0.217 |
| Threshold vs. kNN edge Jaccard overlap | 0.262 |
| Disparity backbone edges (of 775 candidates) | 2 |
| Giant component remains ≥ 50 % up to threshold | 0.325 |
| Avg local clustering (vs. ER null) | 0.495 (vs. 0.218 ± 0.009) |
| Global transitivity | 0.574 |
| Avg shortest path (vs. ER null) | 2.06 (vs. 1.80 ± 0.01) |
| Degree assortativity | +0.278 |
| Louvain communities / modularity | 4 [46, 35, 2, 2] / Q = 0.428 |
| Within-community edge weight mean | 0.284 |
| Between-community edge weight mean | 0.211 |
| Critical bounded-confidence ε (consensus) | ≈ 0.40 (transition band 0.30–0.60) |
| SI seeding effect (high-centrality vs. random) | Negligible (dense/compact network) |
