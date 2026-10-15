# corrected-rct-01 — results, logs and tracking

Full-mode runs (5M frames, evaluation every 50k frames, 10 greedy episodes per evaluation)
on MinAtar. They ran on two cluster partitions from the same code:

| Cluster | Arms | SLURM job name | Array tasks | Originally at |
|---|---|---|---|---|
| A | control, ddqn, per | `rct-five-games-A` | 4–14 | `Results/partition/research_project/` |
| B | nstep, averaged | `rct-five-games-B` | 15–24 | `Results/research_project/` |

Array task = `5 × arm + game`, with arms ordered control, ddqn, per, nstep, averaged and games ordered
breakout, asterix, freeway, spaceinvaders, seaquest.

## Layout

Every folder that holds data is keyed `<arm>/<game>/`.

```
corrected-rct-01/
├── results/<arm>/<game>/<arm>_results_MinAtar_<Game>-v1.json   final results (source of truth)
├── logs/<arm>/<game>/rct-five-games-{A|B}_<arrayjob>_<task>.{out,err}   SLURM stdout/stderr
├── tracking/<arm>/<game>/attempt-<id>.jsonl                    per-attempt loss/epsilon/TD metrics
├── tracking/<arm>/<game>/wandb/                                offline W&B runs (sync with sync_wandb.py)
├── tables/                                                     derived CSVs for plotting (see below)
├── scripts/build_tables.py                                     regenerates tables/ (standard library only)
├── provenance/cluster-{A|B}/                                   code snapshot, slurm scripts, tests,
│                                                               environment.txt, task locks, as deployed
└── archive/rct-cluster-A-ddqn-per.zip                          early Cluster A snapshot (see below)
```

## Integrity

The reorganisation only moved files: none were edited, merged or deleted. `tables/file_moves.csv`
records each file's old path, new path, SHA-256 and size. Every file was checksummed before and after
its move, and git records all 1,742 as exact (100%) renames. The two `environment.txt` files differ
(the clusters used different conda environments), so both are kept under `provenance/`.

## Tables for plotting

Rebuild after any change to results/logs/tracking: `python3 Results/corrected-rct-01/scripts/build_tables.py`.

| File | One row per | Use for |
|---|---|---|
| `coverage.csv` | arm × game | What exists. Status: complete / partial / failed / missing |
| `runs.csv` | arm × game × individual | AUC, final return, sampled hyper-parameters, seed |
| `auc_wide.csv` | game × individual | Paired comparisons: one AUC column per arm |
| `learning_curves.csv` | arm × game × individual × evaluation | Learning curves. `frame = (eval_index + 1) × 50,000`, as in `rct/evaluation.py:compute_auc` |
| `training_metrics.csv.gz` | logged training step (every 5k frames) | Loss, epsilon, sampled MSE, sampled abs-TD curves. Covers tracked attempts only (see gaps) |
| `attempts.csv` | tracked attempt | Links attempt ID ↔ individual, seed, SLURM job, and whether the attempt is in the final results |
| `slurm_jobs.csv` | SLURM job | Node, start/finish, wall time, status, exit code, last stderr line |
| `file_moves.csv` | original file | Old path → new path, SHA-256 |

Checks done when the tables were built: all 468 stored AUCs recompute exactly from
`learning_curves.csv`, and each individual has the same seed and configuration in every arm.

```python
import pandas as pd
curves = pd.read_csv("Results/corrected-rct-01/tables/learning_curves.csv")
runs = pd.read_csv("Results/corrected-rct-01/tables/runs.csv")
loss = pd.read_csv("Results/corrected-rct-01/tables/training_metrics.csv.gz")
```

## Coverage and known gaps

| Game | control | ddqn | per | nstep | averaged |
|---|---:|---:|---:|---:|---:|
| Breakout | — | 30 | 30 | 30 | 30 |
| Asterix | — | 30 | 30 | 30 | 30 |
| Freeway | — | 30 | 30 | **18** | 30 |
| Space Invaders | — | 30 | 30 | 30 | 30 |
| Seaquest | failed | failed | failed | not started | failed |

- **Control has no results in this folder.** Cluster A's array started at task 4, so control tasks 0–3
  never ran in these jobs. Their results, if they exist, live elsewhere. Without control, the
  treatment-vs-DQN comparison cannot be computed from this folder alone.
- **Seaquest failed for every arm** that started: `RuntimeError: Evaluation episode exceeded 100000 steps`
  (greedy policy stalls; the guard aborts on purpose). n-step Seaquest (task 19) never started.
- **n-step Freeway is partial (18/30, individuals 0–17).** Its job (B task 17) died at start-up with
  `mkdir: Already exists` and produced no log or tracking. The 18 saved individuals come from an
  earlier run whose logs are not here.
- **Tracking covers only some attempts.** Per-attempt JSONLs and W&B runs exist only for the attempts
  run by the jobs logged here. Individuals finished by earlier jobs (e.g. 28 of 30 DDQN Breakout
  individuals) have results but no loss history in this folder. `attempts.csv` → `in_final_results`
  shows which result rows have one. Some tracked attempts are partial runs cut by the time budget
  and later resumed; they are kept but are not in the final results.
- **PARTIAL (time budget)** jobs are normal: the job saved progress, resubmitted itself, and a later
  job for the same cell finished it (see `slurm_jobs.csv`).

## Archive

`archive/rct-cluster-A-ddqn-per.zip` is the Cluster A deployment bundle from 2026-10-03 00:22. Its
DDQN/PER result JSONs are partial (13–28 individuals) and are **superseded** by `results/`. It is
kept unmodified for provenance.
