# DQN Randomised Control Trial

Honours research comparing DQN, Double DQN, Prioritised Experience Replay,
n-step DQN, and Averaged DQN across 30 sampled hyperparameter configurations.
Each configuration and training seed is paired across the five variants. The
primary outcome is area under the evaluation learning curve (AUC).

## Results and figures

Start with the [results and methodology review](docs/RESULTS_REVIEW.md) or the
[figure index](analysis_output/review/README.md).

The [n-step/PER audit and final-return analysis](RESEARCH_AUDIT.md) documents a
confirmed PER priority-initialization fix, implementation checks, published
hyperparameter comparisons, and DQN variability. The analysis now also produces
paired final-return tests and figures, plus a last-ten-evaluations sensitivity
analysis. Saved results describe the original training code; corrected PER
performance requires separate reruns.

The [follow-up design and implementation audit](DESIGN_AND_IMPLEMENTATION_AUDIT.md)
explains how the shared hyperparameter population relates to published tuned
results and checks all five arms. The training CLI now runs algorithm/integrity
preflight checks. New results include source/dependency fingerprints, and
training refuses incompatible or legacy resumes; use a fresh output directory
for corrected experiments. Historical results remain readable for analysis.

```sh
python -m pip install -r requirements-analysis.txt
python -m analysis.review
```

The analysis reads `results/`, `slurm/logs/`, `new_logs/`, and `logs/` when present.
It validates pairing and saved AUCs, audits duplicate/restarted jobs, and writes
figures, tables, and coverage information to `analysis_output/review/`.
It expects the reviewed dataset: four completed games, five variants, 30
individuals per cell, and 5 million frames.

Figures include learning curves, paired effects, AUC distributions, Pretorius-style
density curves, Gaussian diagnostics of paired differences, and loss curves for
all four completed games. Partial Seaquest histories are separate.

For a single-game paired AUC test:

```sh
python analyze_rct.py --results-root results --game-slug breakout
```

`review_training_data.py` remains a compatibility entry point for the full review.
The earlier unpaired final-score analysis and historical CSV are preserved in
[`archive/legacy_analysis/`](archive/legacy_analysis/README.md).

## Repository layout

```text
Research_Proposal.pdf     Research design and planned methodology
rct/                      Training, replay, targets, and evaluation
run_rct.py                Training CLI
analysis/                 Multi-game analysis, log audit, and plotting
analyze_rct.py             Single-game paired AUC statistics
review_training_data.py    Compatibility entry point for analysis.review
check_env.py               Training environment checks
verify_arms.py             Algorithm smoke checks
slurm/                    CPU job scripts and historical logs
tests/                    Regression checks
docs/                     Research review, references, and project guide
archive/legacy_analysis/   Superseded analysis preserved for reference
results/                  Raw result JSONs (local, gitignored)
new_logs/                 Additional downloaded logs (local, gitignored)
analysis_output/review/    Figure index and generated figures/tables/audit
```

Raw data are kept locally and are not included by Git. The analysis reports
missing loss histories explicitly; result coverage and log coverage differ.

## Training on the Wits CPU cluster

The SLURM scripts cover **five variants × five games × 30 paired individuals**
(750 full training runs). W&B defaults to **offline**, project `dqn-rct` under
`2716924-wits`, while retaining local result JSONs and SLURM text logs. No API key
is needed on compute nodes; authenticate only when syncing later.

Follow the [cluster launch, storage and sync guide](CLUSTER_RUN_GUIDE.md) for
one-time setup, the 25-task pilot, full submission and continuation. New runs use
`experiments/<experiment-id>/` to keep corrected data separate from historical
results. The original three proposal games are retained; Space Invaders and
Seaquest are documented extensions. AUC remains primary and final return secondary.

The guide also records the methodology qualifications: independent uniform
sampling rather than Pretorius's balanced sampling, the MinAtar network adaptation,
and Seaquest's nonterminating greedy-episode guard. W&B logging does not retune
algorithms or change the paired cohort.

## Validation

Data-integrity checks run during figure generation. Focused parser tests:

```sh
python -m unittest discover -s tests -p 'test_analysis_logs.py'
```

Training regression tests additionally need PyTorch and pytest. See the
[implementation references](docs/REFERENCES.md) and
[cleanup record](docs/REPOSITORY_CLEANUP.md).
