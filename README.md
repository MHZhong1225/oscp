# OSCP implementation

This repository implements synthetic experiments for **OSCP:
Operation-Selection Conditional Conformal Prediction**.

The original synthetic experiment is designed around one phenomenon: ordinary marginal
CP can be valid overall, while the subset selected by a downstream batch
operation, especially the bottom-risk `routine` operation, can under-cover.
OSCP calibrates conditionally on whether a calibration point would have been
selected by the same operation in the same realized batch.

The newer implementation uses the edge-based formulation:

> Relational Operation-Selection Conditional CP.

It maps label probabilities through a many-to-many label-action relation matrix,
selects capacity-constrained unit-action edges in each batch, and returns an
edge-specific label prediction set `C_j^(a)` for every selected edge. The
synthetic relational experiment uses per-action top-capacity selection so the
main OSCP implementation has a fast closed-form reference set.

## Run OSCP on synthetic data

Use the existing `cp` conda environment:

```bash
conda run -n ucp python scripts/run_relational_synthetic.py --seeds 5 --n 30000
```

For a quick smoke test:

```bash
conda run -n ucp python scripts/run_relational_synthetic.py --seeds 1 --n 10000
```

To choose the CUDA device explicitly, add `--cuda 0` or `--cuda 1`.
If you omit `--cuda`, the script uses CPU.

```bash
conda run -n ucp python scripts/run_relational_synthetic.py --seeds 1 --n 10000 --cuda 0
```

For small runs (`--n <= 5000`), the script also includes the generic swap
selection-conditional implementation as a sanity check against the closed-form
top-capacity OSCP.

```bash
conda run -n ucp python scripts/run_relational_synthetic.py --seeds 1 --n 3000
```

Outputs are written to `results/relational_synthetic/raw.csv`,
`results/relational_synthetic/summary.csv`, and
`results/relational_synthetic/diagnostics.csv`.

The relational experiment reports edge coverage for `routine`, `labs`,
`imaging`, `urgent`, and `expert_review`, plus average set size, average
reference set size, action coverage gaps, explain rate, and the routine critical
miss rate.

## Run OSCP on MIMIC-IV

The MIMIC-IV runner builds an ICU-stay-level triage task from
`dataset/mimic-iv-3.1` using demographics, admission metadata, ICU care unit,
and diagnosis-chapter counts. The five labels are derived from ICU length of
stay and hospital mortality.

```bash
conda run -n ucp python scripts/run_relational_mimic.py --seeds 5
```

To choose the CUDA device explicitly, add `--cuda 0` or `--cuda 1`.
If you omit `--cuda`, the script uses CPU.

```bash
conda run -n cp python scripts/run_relational_mimic.py --seeds 1 --n 10000 --cuda 0
```

To sample a fraction of the MIMIC dataset, add `--frac`:

```bash
conda run -n ucp python scripts/run_relational_mimic.py --seeds 1 --frac 0.1
```

Use only one of `--n` or `--frac` in the same run.

For a quicker smoke test, sample fewer ICU stays:

```bash
conda run -n ucp python scripts/run_relational_mimic.py --seeds 1 --n 10000
```

Outputs are written to `results/relational_mimic/raw.csv`,
`results/relational_mimic/summary.csv`,
`results/relational_mimic/summary_full.csv`, and
`results/relational_mimic/diagnostics.csv`. The default `summary.csv` keeps the
displayed percentage columns for easier inspection; `summary_full.csv` keeps the
complete aggregate table with standard deviations.

## Relational methods

- Marginal CP
- JOMI (unit-selected focal-unit baseline)
- OSCP with closed-form top-capacity reference sets
- Generic swap selection-conditional CP API for arbitrary edge selection rules
- Bonferroni conservative CP

