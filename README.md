# DPCN Assignment 1: Opinion Network Formation

**Team: A Shortest Path to an A Grade**

One survey of 60 Likert statements answered by a class; three different ways of
turning it into a network; one report that triangulates across all three.

The survey has 4 topic blocks of 15 statements each - **T**echnology,
**E**ducation, **S**ociety/Ethics, en**V**ironment; answered on a
Strongly Disagree → Strongly Agree scale.

## The three networks

| Dir | Owner | Network | Nodes | Edge rule | Question it answers |
|---|---|---|---|---|---|
| `member_a_agreement_net/` | Vishal | Agreement | 87 students | cosine of z-scored response vectors > 0.15 | Who thinks alike? |
| `member_b_statement_net/` | Harith | Signed statement | 60 statements | Pearson *r*, permutation test + FDR, sign retained | Which opinions co-occur, and which clash? |
| `member_c_bipartite_net/` | Navneet | Bipartite deviation | 87 students + 60 statements | student–statement edge when \|z\| > 1.0 | Who deviates from consensus, and on what? |

A and C share the same 87 student nodes; B and C share the same 60 item nodes.
That overlap is deliberate. The strongest results in the report are
cross-network comparisons, and they are only valid because all three are built
from one shared data contract.

## Layout

```
common/                  the one module every member imports
  contract.py            the data contract: load, encode, drop, z-score
  viz.py                 the shared palette and matplotlib style
Survey_Results_UC.csv    the survey data, read-only, 96 x 61
member_*/src/*.py        that member's logic (load, build, metrics, dynamics)
member_*/notebook.ipynb  orchestrates and narrates; contains no logic
member_*/figures/        figNN_description.png
member_*/tables/         tableNN_description.csv
member_*/gexf/           one file per analysed graph, named for its construction
member_*/FINDINGS.md     results log: technical claim, then plain English
cross_analysis/          comparisons between the three networks (joint work)
FINDINGS_CROSS.md        what the three networks say when compared
report/                  the submitted report
development_guidelines.md  the conventions all three members agreed on
```

## Running it

```bash
source .env/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `common` importable from any directory

python common/contract.py   # self-test: should print 87 x 60

jupyter nbconvert --to notebook --execute --inplace \
    member_a_agreement_net/notebook.ipynb \
    --ExecutePreprocessor.timeout=1800
```

Each member's notebook runs standalone and regenerates every figure, table and
GEXF it references. No member's code imports another member's; `common/` is the
only shared dependency.

`cross_analysis/notebook.ipynb` is run **last**: it reads the three members'
exported CSVs (their published outputs, not their code) and produces the
cross-network comparisons written up in `FINDINGS_CROSS.md`.

## Conventions that matter

- **`Survey_Results_UC.csv` at the repo root is the only data source.** All cleaning lives in
  `common/contract.py` and nowhere else: Likert → 1–5, `No Comments`/blank →
  NaN (never Neutral), drop rows missing more than 30 of 60 answers → 87
  respondents, then per-item z-score across the full 87.
- **Per-item z-scoring is load-bearing.** 80% of the 5158 answers given are
  Agree or Strongly Agree, so raw cosine similarity sits near 0.96 for every
  pair of students and no structure survives without it.
- **`SEED = 42`, and always iterate over `sorted(...)`.** The real test is that
  two consecutive full re-runs produce identical CSVs; a seed alone does not
  guarantee that.
- **Every structural result is measured against a null**: Modularity against a
  degree-preserving configuration model, clustering against Erdős–Rényi,
  frustration against a sign-shuffled null. A bare value proves nothing.
- **One palette, in `common/viz.py`.** Blocks are always T blue `#2a78d6`,
  E orange `#eb6834`, S aqua `#1baf7a`, V yellow `#eda100`.

See `development_guidelines.md` for the full agreement.

## Individual contribution

Grading is individual, so each member owns one network end to end. Its `src/`
modules, its notebook, its figures, tables and GEXF exports, and its
`FINDINGS.md`:

| Member | Directory | Network |
|---|---|---|
| Vishal | `member_a_agreement_net/` | Agreement network over the 87 students |
| Harith | `member_b_statement_net/` | Signed correlation network over the 60 statements |
| Navneet | `member_c_bipartite_net/` | Bipartite student–statement deviation network |

`common/` (the data contract and the shared palette), `cross_analysis/` and the
report are joint work. The report's final section gives the detailed per-task
breakdown.

## The report

`report/report.pdf` (8 pages, built from `report/report.tex` with `pdflatex`) is
the submission. It is deliberately much shorter than the combined `FINDINGS.md`
files: it carries 8 of the repo's 19 figures and the results that change a
conclusion. The `FINDINGS.md` files remain the full working record behind it.
