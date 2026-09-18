# DQN Randomised Control Trial

Honours research comparing DQN, Double DQN, Prioritised Experience Replay,
n-step DQN, and Averaged DQN across 30 sampled hyperparameter configurations.
Each configuration and training seed is paired across the five variants. The
primary outcome is area under the evaluation learning curve (AUC).

## Results and figures

Start with the [results and methodology review](docs/RESULTS_REVIEW.md) or the
[figure index](analysis_output/review/README.md).

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

Current SLURM scripts use **bigbatch**, launching **two array tasks**:
Seaquest (0) and Space Invaders (1). The proposal's Breakout, Asterix, and Freeway
results came from earlier runs. Current training CLI choices follow
`rct/constants.py`; changing the games is a separate experiment configuration.

Set up the `mltest` environment with a suitable PyTorch build and dependencies
from `requirements.txt`. From the project root:

```sh
module load anaconda
~/.conda/envs/mltest/bin/python check_env.py
~/.conda/envs/mltest/bin/python verify_arms.py
mkdir -p logs/rct-bigbatch-test logs/rct-bigbatch-variant results

# Short pilot, then full experiment. Repeat for each variant.
sbatch --export=RCT_VARIANT=control slurm/run_rct_bigbatch_test.slurm
sbatch --export=RCT_VARIANT=control slurm/run_rct_bigbatch_variant.slurm
```

Variant names: `control`, `ddqn`, `per`, `nstep`, `averaged`.
Monitor with `squeue -u "$USER"` and stdout/stderr files in `logs/`.
The current code logs to stdout; no W&B setup is required.

Full experiments train each individual for 5 million frames, evaluating every
50,000 frames over 10 fixed-seed greedy episodes. Resubmission skips completed
individuals; a partial individual's training restarts from the beginning.

## Validation

Data-integrity checks run during figure generation. Focused parser tests:

```sh
python -m unittest discover -s tests -p 'test_analysis_logs.py'
```

Training regression tests additionally need PyTorch and pytest. See the
[implementation references](docs/REFERENCES.md) and
[cleanup record](docs/REPOSITORY_CLEANUP.md).
