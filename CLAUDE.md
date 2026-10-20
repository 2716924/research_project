# CLAUDE.md

Honours research project (Wits, student 2716924): a **paired randomised control trial of DQN
extensions on MinAtar**. Five arms (DQN control, Double DQN, PER, n-step DQN, Averaged DQN) each
train the *same* 30 sampled hyperparameter configurations (each with its own seed). The primary
outcome is AUC of the greedy evaluation curve. Training runs on the Wits HPC `bigbatch` partition
(CPU only, SLURM). Analysis runs on the laptop.

## Where the project stands (updated 2026-10-18)

- **Main dataset is complete and analysed:** `corrected-rct-01`, 4 games × 5 arms × 30 = 600 runs.
  Results draft: `overview.tex` → `overview.pdf` (dated 16 Oct 2026).
- **Headline results** (AUC vs DQN, paired t-test, Bonferroni α = 0.05/16 across all 16 comparisons, as in
  the report; a per-game α = 0.05/4 gives the same nine):
  - Averaged DQN is significantly better in Asterix (+57.5%), Freeway (+20.9%) and Space Invaders (+19.5%).
  - Double DQN is significantly better only in Space Invaders (+11.4%).
  - PER is significantly worse in Freeway (−35.3%) and Space Invaders (−50.0%).
  - n-step is significantly worse in Asterix (−80.0%), Freeway (−59.4%) and Space Invaders (−21.0%).
  - Breakout: no significant difference for any arm.
- **PER bug fix did not explain PER's deficit:** PER AUC before vs after the max-priority fix shows
  no significant change in any game (`analysis_output/corrected-rct-01/tables/per_fix_effect.csv`).
- **In progress: PER loss-history rerun.** PER individuals 0–12 have results but no loss logs.
  Submitted 2026-10-16 on the cluster from `dist/per-rerun.zip` (`RCT_EXPERIMENT_ID=per-loss-rerun-02`,
  `RCT_NUM_INDIVIDUALS=13`, `--array=10-13`). It needs about 3.5 days with one self-resubmission per
  game. Accept it only if `check_per_rerun.py final` prints `ALL 52 MATCH`. Then bring back
  `per-rerun-output.zip` and *move* it into `Results/` without rewriting anything.
- **Next steps** (from `overview.tex`):
  1. PER loss diagnostics: weighted loss, unweighted TD error and importance weights. These are
     already logged in the tracking JSONL.
  2. Exploratory analyses that need no new training: effect vs hyperparameters, ranking stability
     across endpoints, paired bootstrap.
  3. Decide the inference family (per game or across all 16 tests) and how to report Seaquest, then
     freeze the analysis plan and write the results chapter.
- **Seaquest is excluded:** every arm hit the 100k-step greedy-evaluation guard, which aborts on purpose.

## Data provenance (important for any analysis)

- `Results/corrected-rct-01/` holds DDQN, PER, n-step and Averaged DQN (Oct 2026, two cluster
  partitions: A = control/ddqn/per, B = nstep/averaged).
- **This experiment contains no control results.** Cluster A's array started at task 4.
  Control comes from `Results/original-run-2026-09/` (Sept run, results only).
- n-step Freeway individuals 18–29 also come from the original run. Corrected has only 0–17.
- This mix is valid because the corrected rerun reproduced DDQN, n-step and averaged trajectories
  **bit for bit**. The only learning-relevant code change between the runs is the PER max-priority
  fix in `rct/replay.py`. `analysis/corrected_rct01.py` asserts the overlap matches and that
  seed/config pairing holds across all arms.
- Loss coverage is incomplete. Loss histories exist only for attempts tracked by the logged jobs;
  for example, DDQN Breakout has 2 of 30. `Results/corrected-rct-01/tables/attempts.csv` →
  `in_final_results` shows which result rows have one. DQN loss logs exist only in git history,
  before commit `d697076`.

## Hard rules

- **Never edit, merge or delete raw results, logs or tracking files** under `Results/`.
  Move files only, and record each move in `Results/corrected-rct-01/tables/file_moves.csv`
  (with SHA-256). Put derived outputs in `Results/<exp>/tables/` or `analysis_output/`.
- After any filesystem cleanup, re-verify the SHA-256s in `file_moves.csv`.
- **Cross-check every deliverable.** Recompute numbers independently from raw JSON, test checkers on
  good and deliberately broken input, dry-run cluster scripts, and render LaTeX to look at it.
  Report what was verified and how.
- Changing anything in `rct/` changes `source_sha256` provenance in result metadata, and resuming an
  existing result file then refuses to continue (by design). Don't touch `rct/` while a cohort is
  running unless a new experiment ID is intended.
- Hand-built cluster bundles must contain every file `prepare_cluster.py` lists. The preflight
  `verify_arms.py` imports `sync_wandb.py` through the tests; the first PER bundle failed because it
  left that file out.

## Layout

```
rct/                     training package (single source of truth for the experiment)
  constants.py           games, population space (Table 4.1), seeds, shared recipe, treatment defaults
  sampling.py            cohort = fixed-seed sequential draws → N individuals = prefix 0..N-1
  tasks.py               SLURM array task → (variant, game): task = 5*variant_idx + game_idx
  trainer.py             train_individual(); targets.py (control/DDQN/n-step/averaged targets)
  replay.py              Uniform, Prioritized (segment_tree.py), NStepCollector
  evaluation.py          greedy eval on fixed seeds 1000–1009; compute_auc (trapezoid, x=(i+1)*50k)
  results.py             atomic JSON save, strict resume validation, metadata + provenance hashes
  tracking.py            offline W&B + per-attempt JSONL; slurm_utils.py time-budget stop
run_rct.py               CLI entry point (runs verify_arms.py preflight in a subprocess first)
verify_arms.py           11 analytic checks of every arm + test suite preflight
slurm/                   common.sh helpers, run_rct_bigbatch_variant.slurm (full), _test.slurm (pilot)
prepare_cluster.py       builds dist/rct-cluster.tar.gz (explicit file list)
analysis/corrected_rct01.py   current analysis → analysis_output/corrected-rct-01/{tables,figures}
analyze_rct.py           quick single-game paired AUC check (cluster-side helper)
Results/<exp>/           per-experiment, every data folder keyed <arm>/<game>/; see its README.md
  corrected-rct-01/scripts/build_tables.py (stdlib), extract_wandb.py (needs .venv wandb)
overview.tex/.pdf        current results draft (figures from analysis_output/corrected-rct-01/figures)
archive/unused-2026-10-16.zip  Sept analysis pipeline, old bundles; ARCHIVE_MANIFEST.csv inside
dist/per-rerun.zip       PER loss rerun bundle (README + check_per_rerun.py inside)
```

**Stale documents:** `docs/RESULTS_REVIEW.md`, `docs/EXPLAINER.md`, `RESEARCH_AUDIT.md` and
`DESIGN_AND_IMPLEMENTATION_AUDIT.md` describe the September run and pipeline. Their methodology and
audit reasoning still hold, but their numbers, coverage and file paths do not. `docs/REFERENCES.md`
is the citation list. `Research_Proposal.pdf` is the authoritative design (Table 4.1 population,
Table 4.2 arms, §5.2 pilot).

## Commands

Local `.venv` is Python 3.9 with torch, wandb, pandas, scipy and matplotlib. It is not portable to the cluster.

```sh
.venv/bin/python -m unittest discover -s tests          # 38 tests, 2 skipped — all pass as of 2026-10-18
.venv/bin/python verify_arms.py                         # 11 arm checks
bash -n slurm/common.sh slurm/run_rct_bigbatch_variant.slurm slurm/run_rct_bigbatch_test.slurm
.venv/bin/python -m analysis.corrected_rct01            # regenerate analysis tables + figures
python3 Results/corrected-rct-01/scripts/build_tables.py   # rebuild derived CSVs after data changes
.venv/bin/python run_rct.py --variant control --pilot   # local smoke test (3 indiv, 15k frames)
python3 prepare_cluster.py                              # build cluster transfer archive
```

Cluster: Python is `~/.conda/envs/mltest/bin/python` (`RCT_PYTHON`). Outputs go to
`experiments/<RCT_EXPERIMENT_ID>/{pilot,full}/{results,tracking}/<arm>/<game>/`. Logs go to
`logs/rct-bigbatch-variant/`. Exit code 2 means the job stopped for its time budget and resubmits
itself. W&B is offline (entity `2716924-wits`, project `dqn-rct`); sync later with `sync_wandb.py`.
See `CLUSTER_RUN_GUIDE.md`.

## Experiment constants (don't drift)

5M frames, evaluation every 50k frames (100 points), 10 greedy episodes. Population seed 2024,
training seed = 1337 + individual_id. Recipe: γ=0.99, ε 1.0→0.1, MSE loss, RMSprop. PER uses
α=0.6 and β₀=0.4→1. n-step uses n=3. Averaged DQN uses K=10. Game order is Breakout, Asterix,
Freeway, SpaceInvaders, Seaquest. Arm order is control, ddqn, per, nstep, averaged. Both orders
define the SLURM array mapping, so keep them stable.

## Git

Working branch is `Before_PER`, 6 commits ahead of `main`. `main` (Aug 2026) is stale. Other
branches (`4games`, `Full_Results`, `after`) are historical snapshots. Commit only when asked.
