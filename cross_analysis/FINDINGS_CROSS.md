# Cross-Network Findings

**Team: A Shortest Path to an A Grade**
Vishal — agreement network · Harith — signed statement network · Navneet — bipartite deviation network

Each member's own results are in their `FINDINGS.md`. This file is about what the three networks say when you compare them.

Sections 1 and 2 are computed by `cross_analysis/notebook.ipynb`, which reads the three
members' exported CSVs and should be run after all three member notebooks.
Sections 3 and 4 quote the members' own tables.

## Why the comparison is fair

All three networks start from the same data contract (`common/contract.py`): same
raw CSV, same Likert encoding, same >30-missing rule, same per-item z-score.
The three `load.py` modules produce bit-identical 87 × 60 matrices, verified with
`np.allclose(..., equal_nan=True)`.

This matters because A and C describe the same 87 students, and B and C describe
the same 60 statements. Every comparison below is one population measured two ways, not
two populations compared. Where the members differ is in how they handle the
62 remaining missing cells — pairwise-complete cosine with ≥20 shared items
(A), pairwise-complete Pearson with `min_periods=10` (B), no edge for a NaN cell (C).
Each member also reports a mean-imputed variant as a sanity check.

| | Agreement (A) | Signed statement (B) | Bipartite deviation (C) |
|---|---|---|---|
| Nodes | 87 students | 60 statements | 87 students + 60 statements |
| Edge | cosine of z-vectors > 0.15 | Pearson *r*, permutation + FDR, sign kept | student–statement, \|z\| > 1.0 |
| Edges | 822 | 402 (391 +, 11 −) | 1200 |
| Asks | who thinks alike? | which opinions move together? | who breaks from consensus, on what? |

---

## 1. Two different constructions rank the same statements almost identically

Member A measures, for each statement, the gap in mean z-score between the two student
communities — how much that statement separates the class (`abs_gap`,
`table09_community_profile.csv`). Member B measures how far a perturbation to a
statement propagates through the signed correlation network — its dynamical
leverage (`table06_leverage_ranking.csv`). Different graphs, different node sets,
different math, no shared code beyond the data contract.

Across the 58 statements both can score, they agree at **Spearman ρ = +0.784**
(p = 3.5 × 10⁻¹³; Pearson r = +0.617). A statement that splits the class into two camps
is almost always also one whose influence spreads furthest through the opinion
network. → `cross_analysis/fig01_item_level_crosscheck.png` panel (a),
`table02_item_level_correlations.csv`

Grouped by blocks, the two measures give the same ordering, **V > S > E > T**:

| Block | A: mean \|z-gap\| | B: mean leverage | C: mean deviation degree |
|---|---|---|---|
| Environment (V) | 0.986 | 0.957 ± 0.014 | 14.13 |
| Society/Ethics (S) | 0.835 | 0.933 ± 0.054 | 19.33 |
| Education (E) | 0.536 | 0.762 ± 0.235 | 23.13 |
| Technology (T) | 0.434 | 0.714 ± 0.249 | 23.40 |

The spread is worth looking at too. Environment's leverage has an SD of 0.014 — all
fifteen V statements fall between 0.914 and 0.967. Technology's is eighteen times
larger at 0.249, ranging from 0.233 to 0.953. Environment acts as one tightly coupled
bloc; Technology is a loose collection of statements that just share a label. This
fits with B finding T08 and T09 completely isolated.

One caveat: leverage saturates near 0.96, so the positive relationship is mostly
driven by the low-leverage tail.

## 2. Influence and deviation run in opposite directions

The fourth column above goes backwards. Member C's deviation degree — how many students
answered a statement more than one SD from the mean (`table03_item_degree.csv`) —
anti-correlates with both other measures:

| Comparison | Spearman ρ | p | n |
|---|---|---|---|
| B leverage vs C deviation degree | −0.390 | 0.0025 | 58 |
| A \|z-gap\| vs C deviation degree | −0.349 | 0.0063 | 60 |

→ `cross_analysis/fig01_item_level_crosscheck.png` panel (b), `fig02_block_inversion.png`

The statements with the most influence and the sharpest class-dividing power are
the ones individuals deviate from *least*. Environment is where the class is most
tightly coupled, so V statements sort students into camps by *how strongly* they
agree while attracting almost no individual dissent. Technology and Education are
where people actually break ranks.

There are also two different senses of "controversial" here, and they point in opposite
directions. B identifies **E03** as the disagreement axis (8 of 11 negative edges).
C identifies **T13** as the biggest flashpoint (60 of 87 students deviating). Both
are correct: E03 produces patterned anti-correlation, splitting the class into opposed
sides. T13 produces raw spread, where everyone answers differently but without any
particular pattern.

## 3. Consensus is structured; dissent is not

Louvain returns a partition for any graph, so raw modularity doesn't prove anything.
Compared against degree-preserving configuration-model nulls (30 rewirings each):

| Network | Q observed | Q null | z |
|---|---|---|---|
| A — agreement, 87 students | 0.4320 | 0.1779 ± 0.0072 | **+35.5** |
| C — student co-dissent, same 87 students | 0.0368 | 0.0567 ± 0.0032 | **−6.14** |

When students are linked by what they agree about, they fall into two clear camps, 35
SD above chance. When the same students are linked by what they dissent about, a
random graph with the same degrees clusters *better* than the real one. There are no
contrarian cliques in this class.

Network C reaches the same conclusion from two other angles:

- **Nestedness.** The incidence matrix looks nested (NODF = 41.79),
  but against a curveball null fixing both degree sequences, the null is 41.00 ± 0.26.
  The observed value beats it by 0.79 points on a 0–100 scale (z = +3.08). Technically
  above chance, practically nothing.
- **Random walk.** MFPT spans 27× (39.8 to 1063.5 steps), which looks like
  structure, but correlates with deviation degree at ρ = −0.971;
  dividing node strength out leaves a relative range of just 1.005–1.118.

Three different questions, same answer: once you know how often a student
deviates, you know basically everything about where they sit in that network.

That agreement and dissent really are different measurements is confirmed directly.
Correlating A's pairwise similarity against C's symmetrised issue-mediated distance
over 3741 student pairs gives **Pearson r = +0.036** (ρ = +0.026) — 0.1% of variance.
The p-value only clears 0.05 because 3741 pairs make tiny effects detectable.
→ `fig06_crosscheck_agreement.png`, `table11_crosscheck_vs_agreement.csv`

## 4. Running a process reveals things static metrics don't

Each member ran a dynamical process, and in each case it added something the
static structure alone didn't show.

- **A — seeding hubs backfires.** SI spreading from the three highest-eigenvector
  students reaches 0.244 ± 0.266 of the class by t = 30 on the mutual-kNN graph, vs
  0.487 ± 0.401 for random seeds (Welch t = 2.76, p = 0.008). The
  reason: eigenvector centrality rewards being in the same dense core, so the top
  three seeds are right next to each other (1.00 hops apart, vs 3.33 for random) and
  cover the same area. On the dense threshold graph the redundancy doesn't matter;
  on the sparse kNN graph it becomes the dominant effect. Also worth noting: on a
  graph this sparse, outcomes are close to all-or-nothing (SD ≈ mean).
  → `fig08_si_seeding_density.png`
- **B — near-perfect balance produces slow, monotone decay.** Altafini signed consensus
  decays 1.21 → 0.08 without oscillation, consistent with a nearly balanced network
  (frustration 2/402). → `fig04_dynamics.png`
- **C — the walk is pure degree**, as in section 3. → `fig04_mfpt_structure.png`

## 5. The class's beliefs are internally consistent

Two independent measures on two different graphs say the same thing. Average local
clustering in A is 0.505 against an ER null of 0.220 ± 0.009 (z ≈ +32): if you
agree with two people, they very probably agree with each other. In B, zero of 1298
closed triads are frustrated, and the frustration index is 2 of 402 edges against a
sign-shuffled null of 10.978 ± 0.160 (z = −56.2).

There are essentially no cases of holding contradictory positions, and almost no
triples of statements whose signs can't be sorted into two consistent camps.

This also works against the survey's own design. The communities Louvain finds in B
don't match the T/E/S/V blocks well (NMI = 0.277), and A reaches the same conclusion
from the other side: its two student communities differ by intensity, not topic
(mean z −0.283 vs +0.371), with 18 of the 20 most discriminating items being V or S.
The survey designer's grouping is not how this class actually thinks.

## 6. How much do the construction choices drive the conclusions?

Every network needed a threshold that could've been set differently.
Each member swept theirs, and the answer is consistent: big-picture structure is
stable, fine detail is not.

| Member | Choice tested | Effect |
|---|---|---|
| A | threshold vs mutual-kNN edge rule | 822 vs 207 edges; edge Jaccard only 0.252 — yet both give the same two-bloc partition |
| B | permutation+FDR vs fixed \|r\| > 0.3 | 402 → 316 edges; negative edges 11 → 4, isolates 2 → 4 |
| C | \|z\| cutoff 0.75 / 1.0 / 1.5 | 2856 / 1200 / 442 edges — a 6.5× swing |
| C | removing the 8 highest-deviation statements | 100% of pairs stay connected; relative MFPT +1.4% |

Which specific pairs get an edge depends a lot on the rule — three-quarters of A's
edges change between its two rules. What doesn't change is the overall shape: same
communities, same block ordering, same conclusions. We only state results that
survive these sweeps.

Construction choice matters most where the interesting signed structure is:
B's fixed-threshold alternative keeps way fewer negative edges (4 vs 11), which is
why the FDR rule is the primary one.
