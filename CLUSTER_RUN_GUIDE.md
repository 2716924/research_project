# Five-game cluster runs and offline W&B

The cluster scripts now cover **5 variants × 5 games × 30 paired individuals =
750 completed training runs**, each with 5 million training frames. W&B defaults
to **offline**, project `dqn-rct`, entity `2716924-wits`. Nothing is uploaded by
training. Result JSON and SLURM text logs remain local and authoritative.

## Transfer only the runtime files

On the laptop, run `python3 prepare_cluster.py` and transfer only
`dist/rct-cluster.tar.gz`. Extract it on the cluster with
`tar -xzf rct-cluster.tar.gz`, then `cd research_project`. Extract into a fresh
directory so old files from a previous transfer do not remain alongside it.
The archive includes the tests required by training preflight. It excludes
`.venv`, `.git`, historical results/logs/figures, and research documents.
Proposal and audit links below refer to documents retained in the full laptop
repository. Rebuild the archive whenever you change the training code.

## Install once, then pilot

Run these commands on the cluster from the project root. The scripts retain the
existing `bigbatch`, CPU-only setup: one exclusive node, four CPU threads and
16 GB per task. Set `RCT_PYTHON` to another absolute Python path if needed.

```bash
module load anaconda
export RCT_PYTHON="$HOME/.conda/envs/mltest/bin/python"
# Use your existing CPU PyTorch environment. Install once, never concurrently
# inside array tasks, and keep this environment unchanged during a cohort.
"$RCT_PYTHON" -m pip install -r requirements.txt
"$RCT_PYTHON" check_env.py
"$RCT_PYTHON" verify_arms.py
mkdir -p logs/rct-bigbatch-test logs/rct-bigbatch-variant

export RCT_EXPERIMENT_ID=corrected-rct-01
export WANDB_MODE=offline
export WANDB_ENTITY=2716924-wits
export WANDB_PROJECT=dqn-rct
mkdir -p "experiments/$RCT_EXPERIMENT_ID"
"$RCT_PYTHON" -m pip freeze > "experiments/$RCT_EXPERIMENT_ID/environment.txt"

# 25 short pilot tasks; pilot outputs cannot enter the full results directory.
sbatch slurm/run_rct_bigbatch_test.slurm
```

Pilot settings remain 3 individuals, 15,000 frames, evaluation every 3,000 frames,
5 fixed evaluation episodes. Review pilot logs, numerical stability and learning
before submitting the full experiment. A short pilot is a plumbing check; it
does not establish convergence, as requested by proposal §5.2. Seaquest may hit
the nonterminating-evaluation guard described below. Full runs are not submitted
automatically after the pilot.

```bash
# Once the pilot has been reviewed, submit all 25 full cells with one command.
sbatch slurm/run_rct_bigbatch_variant.slurm
squeue -u "$USER"
```

The initial array is `0-24%5`: up to five tasks run concurrently. To request a
different initial concurrency, use e.g. `sbatch --array=0-24%10 ...`.
Automatic continuation submits individual child jobs after clean time-budget
stops. Those jobs are separate arrays, so the original `%5` is not a global cap.
Set `RCT_AUTO_RESUBMIT=0` before submission if you need to manage that cap manually.

| Variant | Breakout | Asterix | Freeway | Space Invaders | Seaquest |
|---|---:|---:|---:|---:|---:|
| control | 0 | 1 | 2 | 3 | 4 |
| ddqn | 5 | 6 | 7 | 8 | 9 |
| per | 10 | 11 | 12 | 13 | 14 |
| nstep | 15 | 16 | 17 | 18 | 19 |
| averaged | 20 | 21 | 22 | 23 | 24 |

`python -m rct.tasks 12` prints the assignment for a task. Every task runs the
same 30 configuration/seed pairs sequentially. These are population replicates,
not 30 seeds of one fixed configuration. Allocation order does not change seeds.

## Storage, failures and continuation

```text
experiments/<experiment-id>/
  environment.txt
  full/
    results/<variant>/<game>/*results*.json
    tracking/<variant>/<game>/attempt-<id>.jsonl
    tracking/<variant>/<game>/wandb/offline-run-*/run-<id>.wandb
  pilot/                         # same structure, separate data
logs/rct-bigbatch-variant/*.out    # training loss and evaluation text retained
logs/rct-bigbatch-variant/*.err
logs/rct-bigbatch-test/*
```

Use shared persistent storage, not node-local `/tmp`. To relocate the experiment,
set `RCT_EXPERIMENT_ROOT` to an absolute shared-filesystem path before submitting.
Without an explicit experiment ID, the initial array job ID is used and exported
to continuation jobs. For a manual continuation, reuse the original experiment
ID/root and select the unfinished array indices. Do not edit the code or update
the Python environment while a cohort is running.

Completed individuals are skipped. Partial individuals restart at frame zero
with the same configuration and seed, and receive a new W&B attempt ID. There is
no model/optimizer/replay checkpoint resume. Result metadata rejects mismatched
source, dependencies, protocol or completed cohort rows. Corrected experiments
must use a new directory rather than append to historical results. An advisory
file lock prevents simultaneous SLURM writers for the same variant/game.

The full script retains automatic resubmission for clean time-budget stops
(exit 2). Failures such as nonfinite learning values or nonterminating evaluation
raise an error and are not automatically resubmitted forever. A hard scheduler
kill may require manual resubmission. The 15-minute exit buffer is a best effort;
a long evaluation can still outlast it. If a single individual cannot finish
within the three-day allocation, restarting it will not make progress; training
checkpoint support would be a separate change.

## Metrics and syncing later

There is one W&B run per individual **attempt**, with experiment/game grouping,
variant, individual ID, training/evaluation seeds, sampled configuration, fixed
treatment settings, source/dependency fingerprints and SLURM job identifiers.
Training loss, epsilon, sampled MSE and absolute TD error are recorded every
1,000 frames once learning starts; PER also records importance weights and beta.
Evaluation return is recorded at the existing evaluation checkpoints.

Final summaries include `auc`, `final_return`, `last_10_mean_return`,
`completed_frames`, `completed` and `status`. Final return is the mean of the ten
fixed-seed evaluation episodes at frame 5,000,000, not a best score. AUC uses the
existing trapezoidal calculation over recorded checkpoints, from 50,000 to
5,000,000; it does not invent a frame-zero score.

The JSON result is saved before W&B receives a completed status. Interrupted and
failed attempts remain available for debugging, but are not experimental
replicates. In W&B, filter the intended `experiment_id`, `pilot=false` and
`completed=true`; use local result IDs to verify one completed row per pair.
Logging preserves Python, NumPy and CPU Torch RNG states. Tracking errors are
raised rather than silently claiming W&B data were saved; local result JSONs
already written remain intact.

After jobs stop, use a machine with internet access and a compatible W&B SDK.
If syncing from your laptop, first copy the whole experiment directory, retaining
the relative paths. Authenticate interactively; the API key is not stored in this
repository or passed as a command argument:

```bash
# Activate the Python environment with W&B installed first.
python -m wandb login

# Preview finalized, unsynced offline attempts (no upload).
python sync_wandb.py experiments/corrected-rct-01/full

# Upload to 2716924-wits/dqn-rct, retaining all local files.
python sync_wandb.py experiments/corrected-rct-01/full --execute
```

The helper skips active/unfinalized attempts and files marked as already synced.
It uploads finalized failed/interrupted attempts too, preserving their status.
Abruptly killed runs require manual inspection and, if desired, `wandb sync`
on their individual directories. Credentials and account write access are only
validated at login/sync time; offline creation cannot validate them. No cloud sync
or cluster submission was performed during implementation.

W&B's official [offline/environment configuration](https://docs.wandb.ai/models/track/environment-variables)
describes local mode and credentials. The sync flags were checked against the
pinned SDK's `wandb sync --help`. SLURM's [sbatch documentation](https://slurm.schedmd.com/sbatch.html)
defines array throttling and the Unix timestamp used for job end time.

## Methodology check

The central research question is unchanged: effects of each isolated DQN
modification across a shared hyperparameter population, using paired comparisons.
W&B is an observation layer, not a sweep or tuning mechanism.

| Item | Implementation and interpretation |
|---|---|
| Proposal §§4.1.1–4.1.2 | Same Table 4.1 candidate values, population seed 2024, 30 sampled individuals, training seeds 1337–1366; identical pairs across all five arms. |
| Isolated treatments | Shared network, optimizer, MSE, exploration, budget and evaluation; only the established DDQN, PER, n-step or averaged-target treatment differs. Fixed PER α=.6/β₀=.4, n=3, K=10 are unchanged. |
| Proposal §4.3 | Full runs: 5 million frames, evaluation every 50,000 frames, ten greedy episodes with seeds 1000–1009. |
| Proposal §4.4 | AUC remains primary; pair each treatment with control and retain four-comparison Bonferroni correction per game. Final return is a secondary endpoint requested subsequently; do not select whichever endpoint is significant. |
| Games | Breakout, Asterix and Freeway are the original proposal games. Space Invaders and Seaquest are an explicit scope extension; report those separately as additional environments. |
| Pretorius correspondence | Conceptual adaptation of sampled designs cloned across interventions. Its balanced sampling procedure is not reproduced by the repo's independent uniform draws. |
| Proposal §5.2 sampling qualification | Marginal stratification is not implemented. Existing cohort coverage is audited: 30 draws, 29 unique configurations; duplicate configurations have different seeds. This does not break between-arm pairing but limits population coverage. |
| Architecture qualification | Uses the established MinAtar-sized CNN, not literal Nature-DQN image architecture. Shared across arms; describe it accurately. |
| Proposal §5.2 validation | Algorithm/integrity preflight runs before training. Unit tests and short integration checks are not evidence of full-scale convergence; inspect the cluster pilots. |
| Seaquest evaluation | A 100,000-step guard raises if a greedy episode never terminates. No partial score is substituted. This prevents silent hanging, but can leave missing pairs. A finite-horizon evaluation amendment would need to be specified consistently across arms and rerun in a fresh cohort before claiming complete Seaquest results. |

[Pretorius et al. (2020), §3.1 and Appendix A](https://www.raillab.org/publication/pretorius-2020-if/pretorius-2020-if.pdf)
is the design inspiration, not a prescription for DQN hyperparameters. The
[local proposal](Research_Proposal.pdf) specifies the actual population and
budget. See [the implementation audit](DESIGN_AND_IMPLEMENTATION_AUDIT.md) for
algorithm/published-setting comparisons and remaining limitations.

The historical `analysis.review` pipeline still deliberately checks the original
four-game dataset. It does not automatically mix these new results with old
runs. For new per-game AUC comparisons use, once complete:

```bash
python analyze_rct.py --results-root experiments/corrected-rct-01/full/results --game-slug breakout
```

## Verification

```bash
python verify_arms.py
python -m unittest discover -s tests
# Optional real SDK test: local sockets are needed, but no credentials/network.
RCT_TEST_WANDB_SDK=1 python -m unittest discover -s tests -p 'test_cluster_tracking.py' -k RealOfflineSDKTest
bash -n slurm/common.sh slurm/run_rct_bigbatch_variant.slurm slurm/run_rct_bigbatch_test.slurm
```

Tracking regressions exercise all 25 shell assignments, identical learned weights
with/without RNG-consuming tracking in all five arms, failure/interruption
statuses, completed-result-before-W&B ordering, resume skips, separate pilot paths,
stable continuation IDs, offline key removal, sync selection and SLURM epoch time.

Validated locally on 2026-09-20: 43 standard unittest checks passed; the two
opt-in real W&B SDK checks also passed (offline artifact creation and identical
PER weights/returns with tracking enabled). The CLI completed an additional
two-frame offline plumbing run with local result JSON, evaluation text and a
syncable W&B file. Algorithm preflight and shell syntax checks passed. All 20
historical result files still match the recorded SHA-256 hashes. These checks do
not certify full-budget convergence or execution on the actual cluster.
