# Bipartite Deviation Network — Findings

**Author: Navneet**  ·  Team: A Shortest Path to an A Grade

Nodes = 87 students **and** 60 statements. An edge joins student *s* to
statement *q* when *s* answered notably against the class consensus on *q*
(|z| > 1 after per-item z-scoring), weighted by |z|.

Full pipeline in `notebook.ipynb`; figures in `figures/`, tables in `tables/`.

**Why deviation rather than agreement**: ~80% of all answers are Agree or
Strongly Agree, so raw responses are nearly identical between students.
Deviation from the per-item consensus is where the interesting stuff lives.
Members A and B both z-score to *remove* that ceiling; this network makes
the residual the thing we actually study.

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

  A 6× swing in edge count. Cutoff 1.0 is our main
  operating point: sparse enough to mean something, dense enough that every
  student still participates.

## Who deviates, and on what

(`tables/table02_student_degree.csv`, `tables/table03_item_degree.csv`, `fig01_degree_distributions.png`)

- **Most contrarian students**: 120 (35 items), 119 (31), 63 (26), 58 and 36
  (25 each), against a mean of 13.8. Students 120 and 119 are clear outliers.
- **Most deviated-on statements**: **T13 (60 of 87 students)**, V03 (50),
  V10 (49), T15 (48), S13 (48), E05 (47), S10 (46), T03 (46).
- **T13 is the single biggest flashpoint in the survey** — 69% of the class
  answered it against the consensus. Note this is 60 out of 87 possible.
- **Low mean ≠ high deviation.** E03 and E02,
  which member B finds at the centre of the disagreement structure,
  rank only 11th and below here. Low mean means "the class agrees it's false";
  high deviation degree means "the class can't settle." Different
  things, and the items with the most extreme means aren't the ones
  the class is most split on.
- **Block mix of the top 8**: T 3, V 2, S 2, E 1. S and V are
  over-represented — same prominence member A finds driving the two-bloc split.
- `fig02_bipartite_layout.png` shows the full graph, students on the left,
  statements on the right, coloured by block, sized by degree.

## Nestedness — and why the obvious reading doesn't hold

- **NODF = 41.8** (rows 44.5, cols 39.1) on a 0–100 scale. Sorted by degree,
  the incidence matrix does look nested: the items a moderate deviant breaks on
  are largely a subset of those a heavy deviant breaks on.
- The tempting conclusion is a hierarchy of dissent — that students work
  down a shared ordering of how contentious each statement is. **But that
  doesn't hold up against a null.**
- NODF goes up automatically with heterogeneous degrees, and degrees here are
  very heterogeneous (students deviate on 3 to 35 items). Against a curveball
  null that fixes *both* degree sequences (30 replicates):

  | | NODF |
  |---|---|
  | Observed | **41.79** |
  | Degree-preserving null | 41.00 ± 0.26 |
  | z | +3.08 |
  | Excess over null | **+0.79 points of 100** |

- The nesting is real but almost entirely explained by the degree
  sequence. Technically above chance; practically 0.8 points on a 100-point
  scale. **No real hierarchy of dissent beyond "some students deviate more
  often than others."** (`fig03_nestedness_matrix.png`,
  `tables/table04_nodf.csv`)
- Same verdict from two other angles: co-dissent modularity is *below*
  its null (z = −6.14), and MFPT correlates with deviation degree at
  ρ = −0.971. Three different structural questions, same answer: **once you
  know how often a student deviates, you know basically everything about where
  they sit in this network.**

## Core-periphery

- 17 students (top 20% by degree) are habitual dissenters; the
  other 70 are more selective about when they disagree.

## One-mode projections

(`tables/table06_student_communities.csv`, `tables/table07_item_communities.csv`)

- **Student co-dissent** (87 nodes, 3699 edges, weight = shared contrarian
  items): Louvain gives 2 communities of 44 and 43 at **Q = 0.0368**.
- **Item co-deviation** (60 nodes, 1684 edges): 2 communities (40 / 20) at
  **Q = 0.0761**.

### Is that "weak structure" or *no* structure?

Louvain returns a partition for any graph, so a small Q doesn't tell us
much on its own. Compared against a degree-preserving
(configuration-model) null, 30 rewirings each (`tables/table05_modularity_null.csv`):

| Projection | Q observed | Q null | z |
|---|---|---|---|
| **Student co-dissent** | **0.0368** | 0.0567 ± 0.0032 | **−6.14** |
| Item co-deviation | 0.0761 | 0.0671 ± 0.0041 | +2.21 |

- **The co-dissent network isn't weakly clustered — it's *less* clustered
  than chance**, by 6.1 SD. A random graph with the same degree
  sequence partitions *better* than the real one.
- The item projection is borderline (+2.2 SD): basically like a
  structureless graph.
- Compare with member A's agreement network on the same 87 students:
  Q = 0.432 vs a configuration null of 0.178 ± 0.007, z = +35.5.

- So: consensus has structure; dissent does not. Agreement is 35 SD above
  chance; co-dissent is 6 SD *below* it. Students who agree form two clear
  blocs. Students who deviate form no groups at all — actually less than
  random. There's no "contrarian clique."

## Dynamics: random walk and mean first-passage time

A walker hops student → item → student, so it can only travel between two
students through an issue they both deviate on. MFPT(s → t) is an
**issue-mediated distance**. Computed exactly from the fundamental matrix
`Z = (I − T + 1π^T)^{-1}`, `H_ij = (Z_jj − Z_ij)/π_j`, with the true
stationary distribution `π_j ∝ strength_j`.

> **Correction.** An earlier version used a *uniform* π, which only works
> on a regular graph. Strengths here span ~30×, and the uniform version
> collapsed the MFPT range from 27× to 1.2×, giving the wrong conclusion
> that "the network is homogeneous." All numbers below are from the
> corrected computation.

- **All 7482 ordered student pairs are connected.** Mean MFPT **240.6** steps,
  median 210.7, min **39.8**, max **1063.5** — a **27× spread**.
- **That spread is almost entirely degree.** Spearman ρ between incoming MFPT
  and deviation degree is **−0.971**. The walker reaches the loud deviants
  (120: 42 steps, 119: 63) almost immediately and takes 25× longer to find the
  quiet ones (99: 1048 steps on 3 deviations; 67: 691 on 5).
- **Dividing strength out leaves almost nothing.** Relative MFPT
  `R = H·strength/vol` spans only **1.005–1.118 (1.11×)**, right around the
  degree-only expectation of 1.0. (`fig04_mfpt_structure.png`)
- **The deviation network behaves like a configuration model.** Once you
  know how often each student deviates, you know basically everything about
  how the dynamics reach them. No hidden pockets of dissent, no bottlenecks.

## Robustness: are the flashpoint issues load-bearing?

Remove the 8 highest-deviation statements one at a time
(`tables/table10_robustness_mfpt.csv`, `fig05_mfpt_robustness.png`).

- **100% of student pairs stay connected** throughout — even after deleting
  T13, V03, V10, T15, S13, E05, S10 and T03.
- Raw mean MFPT rises 240.6 → 291.2 (**+21.0%**), but most of that is
  mechanical: fewer edges → larger MFPT.
- **Relative MFPT rises only +1.4%.** After correcting for graph size, removing
  the eight biggest flashpoints barely changes anything.
- No single issue is the bridge holding dissent together. Deviation is
  genuinely spread out.

## Cross-check: is deviating together the same as agreeing?

Member A's agreement similarity recomputed here from our own `load.py`,
correlated against symmetrised MFPT over all 3741 comparable pairs
(`tables/table11_crosscheck_vs_agreement.csv`,
`fig06_crosscheck_agreement.png`).

- **Pearson r = +0.036, Spearman ρ = +0.026.** Basically zero.
- **Agreement distance and issue-mediated deviation distance are unrelated.**
  Knowing two students think alike tells you nothing about whether they break
  from the class on the same questions.

## Cross-member connections

1. **Consensus is structured, dissent is not** — A's Q = 0.432 (+35.5 SD)
   vs C's Q = 0.0368 (−6.14 SD), on the same 87 students. Backed by
   r = +0.036 between the two distance measures.
2. **S/V prominence** shows up independently in all three networks: S and V
   over-represented among highest-deviation items here, drive A's bloc split,
   and top B's leverage ranking.
3. **T13 vs E03** — C's top flashpoint (T13) is not B's disagreement axis
   (E03), and B's axis isn't near the top here. Raw spread and patterned
   anti-correlation are different things.
4. **Bridges vs deviants** — A's student 72 (structural bridge between
   agreement blocs) and C's top deviants (120, 119) are completely different
   people. Being between the blocs is not the same as being outside the
   consensus.

## Summary table

| Quantity | Value |
|---|---|
| Nodes / edges (|z| > 1.0) | 87 students + 60 items / 1200 |
| Density | 0.230 |
| Student mean / max degree | 13.8 / 35 (student 120) |
| Item mean / max degree | 20.0 / 60 (T13, of 87 possible) |
| Cutoff sensitivity (0.75 / 1.0 / 1.5) | 2856 / 1200 / 442 edges |
| NODF nestedness (vs degree-preserving null) | 41.79 vs 41.00 ± 0.26, z = +3.08 — **+0.8 points, i.e. degree explains it** |
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

`notebook.ipynb` is reproducible with `SEED = 42`: two consecutive runs
produce identical CSVs. Projections use sorted node lists for
deterministic output; stable at 2 communities [44, 43] at Q = 0.0368.
