# Bipartite Deviation Network — Findings

**Author: Navneet**  ·  Team: A Shortest Path to an A Grade

Nodes = 87 students **and** 60 statements. An edge joins student *s* to
statement *q* when *s* answered notably against the class consensus on *q*
(|z| > 1 after per-item z-scoring), weighted by |z|.

Full pipeline in `notebook.ipynb`; figures in `figures/`, tables in `tables/`.

**Why deviation rather than agreement**: ~80% of all answers are Agree or
Strongly Agree, so raw responses are nearly identical between students.
Deviation from the per-item consensus is where the information is. Members A
and B both z-score to *remove* that ceiling; this network makes the residual
the object of study directly.

## Network construction

- **Primary graph (|z| > 1.0)**: 147 nodes, **1200 edges**, density 0.230.
  Student mean degree **13.8** of 60 items; item mean degree **20.0** of 87
  students. No isolates on either side.
- **Sensitivity** (`tables/table01_sensitivity_sweep.csv`):

  | cutoff | edges | student mean deg | isolated students |
  |---|---|---|---|
  | 0.75 | 2856 | 32.8 | 0 |
  | **1.0** | **1200** | **13.8** | **0** |
  | 1.5 | 442 | 5.1 | 4 |

  A 6× swing in edge count across the range. Cutoff 1.0 is the operating
  point: sparse enough to mean something, dense enough that every student
  still participates.

## Who deviates, and on what

(`tables/table02_student_degree.csv`, `tables/table03_item_degree.csv`, `fig01_degree_distributions.png`)

- **Most contrarian students**: 120 (35 items), 119 (31), 63 (26), 58 and 36
  (25 each), against a mean of 13.8. Students 120 and 119 are genuine outliers.
- **Most deviated-on statements**: **T13 (60 of 87 students)**, V03 (50),
  V10 (49), T15 (48), S13 (48), E05 (47), S10 (46), T03 (46).
- **T13 is the single biggest flashpoint in the survey** — 69% of the class
  answered it against the consensus. Note this is 60 out of 87 possible, not a
  saturated node.
- **The lowest-mean items are not the highest-deviation items.** E03 and E02,
  which member B finds at the centre of the class's disagreement structure,
  rank only 11th and below here. Low mean means "the class agrees it's false";
  high deviation degree means "the class can't settle". Those are different
  things, and the survey's most contentious items by mean are not the ones the
  class is most split on.
- **Block mix of the top 8**: T 3, V 2, S 2, E 1. Society and Environment are
  over-represented relative to Education — the same S/V prominence member A
  finds driving the two-bloc split.
- `fig02_bipartite_layout.png` shows the whole graph with students on the left
  and statements on the right, coloured by block and sized by degree — useful
  for seeing at a glance that deviation edges are spread across the class
  rather than concentrated on a few students.

## Nestedness

- **NODF = 41.8** (rows 44.5, cols 39.1) on a 0–100 scale. Moderately nested.
- Interpretation: the contrarian items of a moderate deviant tend to be a
  *subset* of a strong deviant's. There is a **hierarchy of dissent** — people
  do not each pick an idiosyncratic set of issues to disagree on; they work
  down a shared ordering of how contentious each statement is.
  (`fig03_nestedness_matrix.png`)

## Core-periphery

- 17 students (top 20% by degree) form a core of habitual dissenters; the
  other 70 are selectively contrarian.
- Most of the class picks its battles; a fifth of it is broadly out of step.

## One-mode projections

(`tables/table06_student_communities.csv`, `tables/table07_item_communities.csv`)

- **Student co-dissent** (87 nodes, 3699 edges, weight = shared contrarian
  items): Louvain gives 2 communities of 44 and 43 at **Q = 0.0368**.
- **Item co-deviation** (60 nodes, 1684 edges): 2 communities (40 / 20) at
  **Q = 0.0761**.

### Is that "weak structure" or *no* structure?

Louvain returns a partition for any graph, so a small Q on its own proves
nothing. We compared both projections against a **degree-preserving
(configuration-model) null**, 30 rewirings each (`tables/table05_modularity_null.csv`):

| Projection | Q observed | Q null | z |
|---|---|---|---|
| **Student co-dissent** | **0.0368** | 0.0567 ± 0.0032 | **−6.14** |
| Item co-deviation | 0.0761 | 0.0671 ± 0.0041 | +2.21 |

- **The co-dissent network is not weakly clustered — it is *less* clustered
  than chance**, by 6.1 SD. A random graph with exactly the same degree
  sequence partitions *better* than the real one.
- The item co-deviation projection is marginal (+2.2 SD): barely distinguishable
  from a structureless graph, and not something to build an argument on.
- Member A's agreement network on the *same 87 students* scores **Q = 0.432 vs
  a configuration null of 0.178 ± 0.007, z = +35.5**.

- **This is the headline result: consensus has structure, dissent does not.**
  Agreement is 35 SD above chance; co-dissent is 6 SD *below* it. Students who
  agree form two clear, highly assortative blocs. Students who deviate form no
  groups at all — less than random, in fact. There is no "contrarian clique",
  and disagreeing with the class on one issue tells you nothing about what else
  you disagree about.

## Dynamics: random walk and mean first-passage time

A walker hops student → item → student, so it can only travel between two
students via an issue they both deviate on. MFPT(s → t) is therefore an
**issue-mediated distance**. Computed exactly from the fundamental matrix
`Z = (I − T + 1π^T)^{-1}`, `H_ij = (Z_jj − Z_ij)/π_j`, with the true
stationary distribution `π_j ∝ strength_j`.

> **Correction.** An earlier version used a *uniform* π, which is only valid
> on a regular graph. Strengths here span ~30×, and the uniform approximation
> collapsed the MFPT range from 27× to 1.2×, producing the false conclusion
> that "the network is homogeneous, no student is special". All numbers below
> are from the corrected computation.

- **All 7482 ordered student pairs are connected.** Mean MFPT **240.6** steps,
  median 210.7, min **39.8**, max **1063.5** — a **27× spread**.
- **That spread is almost entirely degree.** Spearman ρ between a student's
  mean incoming MFPT and their deviation degree is **−0.971**. The walker
  reaches the loud deviants (120: 42 steps, 119: 63) almost immediately and
  takes 25× longer to find the quiet ones (99: 1048 steps on 3 deviations;
  67: 691 on 5).
- **Dividing strength out leaves almost nothing.** The relative MFPT
  `R = H·strength/vol` spans only **1.005–1.118 (1.11×)**, hugging the
  degree-only expectation of 1.0. (`fig04_mfpt_structure.png`)
- **So the deviation network behaves like a configuration model.** Once you
  know how often each student deviates, you know essentially everything about
  how the dynamics reach them. There are no structural pockets of dissent, no
  bottlenecks, no gatekeepers. This is the *same* conclusion the buggy version
  reached, but for an honest reason and with the real quantities: the
  interesting fact is not that MFPT is flat (it is not — it spans 27×), it is
  that **all of the variation is accounted for by volume of dissent alone**.

## Robustness: are the flashpoint issues load-bearing?

Remove the 8 highest-deviation statements one at a time
(`tables/table10_robustness_mfpt.csv`, `fig05_mfpt_robustness.png`).

- **100% of student pairs stay connected** throughout — even with T13, V03,
  V10, T15, S13, E05, S10 and T03 all deleted.
- Raw mean MFPT rises 240.6 → 291.2 (**+21.0%**), but most of that is
  mechanical: deleting edges shrinks the graph volume that MFPT scales with.
- **Relative MFPT rises only +1.4%.** After correcting for volume, removing
  the eight biggest flashpoints in the survey changes the structure of dissent
  by almost nothing.
- No single issue — not T13, not E03 — is the bridge holding dissent together.
  Deviation is genuinely diffuse, which is the dynamical counterpart of the
  near-zero modularity above.

## Cross-check: is deviating together the same as agreeing?

Member A's agreement similarity (pairwise-complete cosine of z-scored vectors)
recomputed here from our own `load.py`, correlated against the symmetrised
MFPT over all 3741 comparable pairs (`tables/table11_crosscheck_vs_agreement.csv`,
`fig06_crosscheck_agreement.png`).

- **Pearson r = +0.036, Spearman ρ = +0.026.** Essentially zero.
- **Agreement distance and issue-mediated deviation distance are unrelated
  measurements.** Knowing that two students think alike overall tells you
  nothing about whether they break from the class on the same questions.
- This is the quantitative backing for the qualitative contrast above, and the
  clearest argument in the report for why three different network
  constructions on one dataset were worth building: they are not three views
  of the same structure, they are measurements of genuinely different things.

## Cross-member connections

1. **Consensus is structured, dissent is not** — A's Q = 0.432 (**+35.5 SD**
   above a configuration null) vs C's Q = 0.0368 (**−6.14 SD**, i.e. below its
   own null), on the same 87 students. Backed by r = +0.036 between the two
   distance measures. Both halves of the contrast are now tested against a
   null rather than asserted from the raw Q values.
2. **S/V prominence** confirmed independently: S and V are over-represented
   among the highest-deviation items, and they drive A's bloc split and top
   B's leverage ranking.
3. **T13 vs E03** — C's top flashpoint (T13) is not B's disagreement axis
   (E03), and B's axis is not near the top here. Raw spread and patterned
   anti-correlation are different phenomena; the report should say so rather
   than pretend the three networks agree on everything.
4. **Bridges vs deviants** — A's student 72 is the structural bridge between
   agreement blocs; C's top deviants (120, 119) are different people entirely.
   Being between the blocs is not the same as being outside the consensus.

## Summary table

| Quantity | Value |
|---|---|
| Nodes / edges (|z| > 1.0) | 87 students + 60 items / 1200 |
| Density | 0.230 |
| Student mean / max degree | 13.8 / 35 (student 120) |
| Item mean / max degree | 20.0 / 60 (T13, of 87 possible) |
| Cutoff sensitivity (0.75 / 1.0 / 1.5) | 2856 / 1200 / 442 edges |
| NODF nestedness | 41.8 / 100 |
| Core dissenters (top 20%) | 17 of 87 |
| Co-dissent projection: edges / Q | 3699 / **0.0368** |
| Co-dissent Q vs configuration null | 0.0368 vs 0.0567 ± 0.0032, z = **−6.14** |
| Item co-deviation projection: edges / Q | 1684 / 0.0761 (z = +2.21) |
| MFPT mean / median / range | 240.6 / 210.7 / 39.8–1063.5 (**27×**) |
| Spearman ρ(incoming MFPT, deviation degree) | **−0.971** |
| Relative MFPT range | 1.005–1.118 (**1.11×**) |
| Connected student pairs | 7482 / 7482 (**100%**) |
| Robustness: remove top 8 items | raw MFPT +21.0%, **relative +1.4%**, 100% connected |
| Correlation with A's agreement similarity | **r = +0.036**, ρ = +0.026 |


## Reproducibility

`notebook.ipynb` is byte-for-byte reproducible with `SEED = 42`: two consecutive
executions write identical CSVs.

One non-determinism bug had to be fixed to get there. `student_projection` and
`item_projection` passed a **`set`** of node names to
`nx.bipartite.weighted_projected_graph`. Set iteration order depends on Python's
per-process hash randomisation, so the projection's node insertion order — and
therefore the Louvain result on it — changed between runs even with
`random_state` fixed. Observed across three runs before the fix: 3 communities
[36, 44, 7], then 2 communities [45, 42], [50, 37], [49, 38], with Q drifting
0.037–0.039. Both functions now pass a sorted list, and the result is a stable
2 communities [44, 43] at Q = 0.0368.

That instability was itself diagnostic — a partition that reshuffles on every
run is a partition that is not there — but it is only reportable once it is
measured deliberately against a null instead of observed by accident.
