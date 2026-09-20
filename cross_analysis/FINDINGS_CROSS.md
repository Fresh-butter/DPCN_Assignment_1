# Cross-Network Findings

**Team: A Shortest Path to an A Grade**
Vishal — agreement network · Harith — signed statement network · Navneet — bipartite deviation network

Each member's own results are in their `FINDINGS.md`. This file is about what the three networks say when you compare them.

Sections 1 and 2 are computed by `cross_analysis/notebook.ipynb`, which reads the three
members' exported CSVs (their published outputs, never their code) and needs to be run after
all three member notebooks. Sections 3 and 4 quote the members' own tables directly.

## Why the comparison is fair

All three networks are built through one data contract (`common/contract.py`): same
raw CSV, same Likert encoding, same >30-missing rule, same per-item z-score
computed over the full sample. The three `load.py` modules produce bit-identical 87 × 60
matrices, verified pairwise with `np.allclose(..., equal_nan=True)`.

This matters because A and C describe the same 87 students, and B and C describe
the same 60 statements. Every comparison below is one population measured two ways, not
two populations set side by side. Where the members differ is in what they
do with the 62 remaining missing cells — pairwise-complete cosine with ≥20 shared items
(A), pairwise-complete Pearson with `min_periods=10` (B), no edge for a NaN cell (C).
That's a design choice rather than a cleaning one, and each member also reports a
mean-imputed variant as a sensitivity check.

| | Agreement (A) | Signed statement (B) | Bipartite deviation (C) |
|---|---|---|---|
| Nodes | 87 students | 60 statements | 87 students + 60 statements |
| Edge | cosine of z-vectors > 0.15 | Pearson *r*, permutation + FDR, sign kept | student–statement, \|z\| > 1.0 |
| Edges | 822 | 402 (391 +, 11 −) | 1200 |
| Asks | who thinks alike? | which opinions move together? | who breaks from consensus, on what? |

---

## 1. Two different constructions rank the same statements almost identically

Member A measures, for each statement, the gap in mean z-score between the two student
communities — how sharply that statement separates the class (`abs_gap`,
`table09_community_profile.csv`). Member B measures, for each statement, how far a
perturbation to it propagates through the signed correlation network — its dynamical
leverage (`table06_leverage_ranking.csv`). Different graphs, different node sets,
different math, no shared code beyond the data contract.

Across the 58 statements both can score, they agree at **Spearman ρ = +0.784**
(p = 3.5 × 10⁻¹³; Pearson r = +0.617). A statement that splits the class into two camps
is almost always also a statement whose influence spreads furthest through the opinion
network. → `cross_analysis/fig01_item_level_crosscheck.png` panel (a),
`table02_item_level_correlations.csv`

Grouped by blocks, the two measures give the exact same ordering, **V > S > E > T**:

| Block | A: mean \|z-gap\| | B: mean leverage | C: mean deviation degree |
|---|---|---|---|
| Environment (V) | 0.986 | 0.957 ± 0.014 | 14.13 |
| Society/Ethics (S) | 0.835 | 0.933 ± 0.054 | 19.33 |
| Education (E) | 0.536 | 0.762 ± 0.235 | 23.13 |
| Technology (T) | 0.434 | 0.714 ± 0.249 | 23.40 |

The spread is as interesting as the means. Environment's leverage has a standard
deviation of 0.014 — all fifteen V statements fall between 0.914 and 0.967 — while
Technology's is eighteen times larger at 0.249, ranging from 0.233 to 0.953. Environment
behaves as one tightly coupled bloc; Technology is a loose collection of statements that
happen to share a label. This fits with B finding T08 and T09 completely isolated,
correlating with nothing in the survey including the rest of Technology.

One caveat: leverage saturates near 0.96 for well-connected statements, so
the positive relationship is mostly driven by the low-leverage tail.

## 2. Influence and deviation run in opposite directions

The fourth column above goes backwards, and that's actually a finding in itself. Member C's
deviation degree — how many students answered a statement more than one SD from the mean
(`table03_item_degree.csv`) — anti-correlates with both other measures:

| Comparison | Spearman ρ | p | n |
|---|---|---|---|
| B leverage vs C deviation degree | −0.390 | 0.0025 | 58 |
| A \|z-gap\| vs C deviation degree | −0.349 | 0.0063 | 60 |

→ `cross_analysis/fig01_item_level_crosscheck.png` panel (b), `fig02_block_inversion.png`

The statements with the most influence and that most sharply divide the class are
the ones individuals deviate from *least*. Environment is where the class is most
tightly coupled and most consensual, so V statements sort students into camps by *how
strongly* they agree while attracting almost no individual dissent. Technology and
Education are where individuals actually break ranks.

Two senses of "controversial" turn out to be almost opposite, and the concrete case makes
it clear. B identifies **E03** ("class attendance should be compulsory") as the
disagreement axis, carrying 8 of the network's 11 negative edges. C identifies **T13** as
the biggest flashpoint, with 60 of 87 students deviating by more than one SD. Both are
correct, because they measure different things: E03 produces patterned
anti-correlation, splitting the class into opposed sides, whereas T13 produces raw
spread, making everyone answer differently from everyone else with no particular pattern.
You need two networks to even see the distinction.

## 3. Consensus is structured; dissent is not

Louvain returns a partition for any graph, so raw modularity proves nothing.
Compared against degree-preserving configuration-model nulls (30 rewirings each):

| Network | Q observed | Q null | z |
|---|---|---|---|
| A — agreement, 87 students | 0.4320 | 0.1779 ± 0.0072 | **+35.5** |
| C — student co-dissent, same 87 students | 0.0368 | 0.0567 ± 0.0032 | **−6.14** |

When students are linked by what they agree about, they fall into two sharp camps, 35
standard deviations more clearly than chance. When the very same students are
linked by what they dissent about, a random graph with identical degrees clusters
*better* than the real one. Dissent here isn't weakly organised; it's organised worse
than random. There are no contrarian cliques in this class.

Network C reaches the same conclusion from two other angles:

- **Nestedness.** Sorted by degree the incidence matrix looks nested (NODF = 41.79),
  which suggests students work down a shared ordering of contentiousness.
  Against a curveball null fixing both degree sequences, the null is 41.00 ± 0.26 — the
  observed value beats it by 0.79 points on a 0–100 scale (z = +3.08). Technically above
  chance, but practically nothing.
- **Random walk.** Mean first-passage time spans 27× (39.8 to 1063.5 steps), which looks
  like strong structure, but correlates with deviation degree at Spearman ρ = −0.971;
  dividing node strength out leaves a relative range of just 1.005–1.118.

Three different structural questions, one answer: once you know how often a student
deviates, you know basically everything about where they sit in that network.

That agreement and dissent really are different measurements and not just two views of
one structure is confirmed directly. Correlating A's pairwise similarity against C's
symmetrised issue-mediated distance over 3741 student pairs gives **Pearson r = +0.036**
(ρ = +0.026) — explaining 0.1% of variance. The p-value only clears 0.05 because 3741
pairs make tiny effects detectable; there's basically no relationship here.
→ `fig06_crosscheck_agreement.png`, `table11_crosscheck_vs_agreement.csv`

## 4. Running a process reveals what just measuring the graph doesn't

Each member ran a dynamical process, and in each case the process added something that
the static structure alone didn't show.

- **A — hub seeding backfires.** Seeding SI spreading at the three highest-eigenvector
  students reaches 0.244 ± 0.266 of the class by t = 30 on the mutual-kNN graph, against
  0.487 ± 0.401 for random seeds (mean ± SD, 30 trials; Welch t = 2.76, p = 0.008). The
  reason is seed redundancy: eigenvector centrality rewards sitting in the same dense
  core, so the top three seeds are right next to each other (1.00 hops apart, against 3.33
  for random seeds) and basically cover the same area. On the dense threshold graph
  every student is ~2 hops from everyone, so the redundancy doesn't matter and the effect
  is invisible. It only shows up once the network is sparse. Also: on a graph this
  sparse an outbreak either catches or dies, so the SD is as large as the mean and
  individual trials scatter across nearly the whole range. → `fig08_si_seeding_density.png`
- **B — near-perfect balance produces slow, monotone decay.** Altafini signed consensus
  decays 1.21 → 0.08 without oscillation, which is the predicted signature of a graph that
  is almost perfectly balanced (frustration 2/402). → `fig04_dynamics.png`
- **C — the walk is pure degree**, as in section 3. → `fig04_mfpt_structure.png`

## 5. The class's belief system is internally consistent

Two independent measures on two different graphs say the same thing. Average local
clustering in A is 0.505 against an Erdős–Rényi null of 0.220 ± 0.009 (z ≈ +32): if you
agree with two people, they very probably agree with each other. In B, zero of 1298
closed triads are frustrated, and the frustration index is 2 of 402 edges against a
sign-shuffled null of 10.978 ± 0.160 (z = −56.2).

There are essentially no cases of holding two positions that contradict each other, and
almost no triples of statements whose agreements and disagreements can't be sorted
into two consistent camps.

This also works against the survey's own design. The communities Louvain finds in B's
statement network match the T/E/S/V blocks only weakly (NMI = 0.277), and A reaches the
same conclusion from the other side: its two student communities differ by intensity
rather than topic (mean z −0.283 against +0.371), with 18 of the 20 most discriminating
items being V or S. The survey designer's idea of what groups together is not how this
class actually thinks.

## 6. How much do the construction choices drive the conclusions?

Every network required a threshold that could've been set differently.
Each member swept theirs, and the answer is consistent: big-picture structure is stable,
fine detail is not.

| Member | Choice tested | Effect |
|---|---|---|
| A | threshold vs mutual-kNN edge rule | 822 vs 207 edges; edge Jaccard only 0.252 — yet both give the same two-bloc partition |
| B | permutation+FDR vs fixed \|r\| > 0.3 | 402 → 316 edges; negative edges 11 → 4, isolates 2 → 4 |
| C | \|z\| cutoff 0.75 / 1.0 / 1.5 | 2856 / 1200 / 442 edges — a 6.5× swing |
| C | removing the 8 highest-deviation statements | 100% of pairs stay connected; relative MFPT +1.4% |

Which specific pairs get an edge depends heavily on the rule — three-quarters of
A's edges change between its two rules. What doesn't change is the shape: the same two
communities, the same block ordering, the same conclusions. Results that survive this are
the ones we state; results that wouldn't survive aren't claimed.

Construction choice matters most exactly where the interesting signed structure is:
B's fixed-threshold alternative keeps way fewer negative edges (4 vs 11), which is why the
FDR rule is the primary one.
