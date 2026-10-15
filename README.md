# DQN Randomised Control Trial

Honours research comparing DQN, Double DQN, Prioritised Experience Replay,
n-step DQN, and Averaged DQN across 30 paired hyperparameter configurations and
five MinAtar games: **750 full training runs**. AUC is the primary outcome.

## Prepare the cluster transfer

```sh
python3 prepare_cluster.py
```

Transfer only `dist/rct-cluster.tar.gz`. It contains the training package,
launch scripts, required preflight tests, environment check, W&B sync helper,
single-game analysis helper and cluster guide. The archive excludes the local
Python environment, Git history, research documents, historical analysis code,
results, logs, figures and caches. Rebuild it after changing the training code.

On the cluster, extract it into a fresh directory:

```sh
tar -xzf rct-cluster.tar.gz
cd research_project
```

Follow [CLUSTER_RUN_GUIDE.md](CLUSTER_RUN_GUIDE.md) for environment setup, the
25-task pilot and full submission. Use the cluster's CPU PyTorch environment;
the laptop's `.venv` is not portable. W&B defaults to offline, with new results
under `experiments/<experiment-id>/`. Review the pilot before the full run.

## Local cleanup

Historical `results/`, `new_logs/`, `slurm/logs/` and generated
`analysis_output/` were removed for the fresh mass run on 20 September 2026,
following confirmation that the results are already on GitHub. They remain
available in earlier Git history. New generated outputs are ignored by Git.

Research documents, analysis source, tests and the local Python environment
remain available on the laptop, outside the transfer archive. Historical
reviews describe the previous experiment and require its archived data.

For paired AUC analysis of a completed new game:

```sh
python analyze_rct.py --results-root experiments/corrected-rct-01/full/results --game-slug breakout
```

The historical four-game review is still available as `python -m analysis.review`
after restoring its old data and installing `requirements-analysis.txt`.

## Validation

```sh
python verify_arms.py
python -m unittest discover -s tests
bash -n slurm/common.sh slurm/run_rct_bigbatch_variant.slurm slurm/run_rct_bigbatch_test.slurm
```

See the [implementation audit](DESIGN_AND_IMPLEMENTATION_AUDIT.md),
[research audit](RESEARCH_AUDIT.md) and [proposal](Research_Proposal.pdf)
for the research design and methodological qualifications.
