# Follow-up runs (lecturer feedback, Oct 2026)

Self-contained bundle. Unzip it in its **own folder** next to (not inside) any running
experiment; the PER loss rerun in `per-rerun/` is untouched by this bundle.

Every manifest row is one 5M-frame run (one arm, one game, one individual). Each SLURM
array task runs `RCT_RUNS_PER_TASK` rows (default 4) side by side on one node.

| Group | Runs | What it answers |
|---|---|---|
| `recipe-sampled` | 42 | Positive control: DQN, PER, n-step on Asterix + Freeway under the Obando-Ceron & Castro (2021) recipe, on the 7 individuals with lr 1e-4 (12, 13, 18, 20, 22, 26, 28). Paired with their existing runs. |
| `recipe-reference` | 40 | Their exact published MinAtar DQN configuration (5 seeds): DQN, PER, n-step on Asterix + Freeway, DQN on Breakout + Space Invaders. |
| `lr-low-dqn` | 56 | DQN on the same 7 individuals at lr 3e-5 and 1e-5 (all 4 games). |
| `seaquest-capped` | 150 | All 5 arms x 30 individuals on Seaquest; a greedy evaluation episode is cut at 100,000 steps and keeps its partial return instead of aborting the run. |
| `lr-low-arms` | 224 | Optional: the 4 treatments at lr 3e-5 and 1e-5. |

Recipe `obando2021`: Adam (eps 3.125e-4), MSE loss, one gradient step every 4 frames,
epsilon 1.0 -> 0.01. The reference configuration adds lr 2.5e-4, replay 100k, batch 32,
target period 1000, epsilon decay 250k steps, learning starts after 1000 steps, MinAtar
network with one hidden layer. PER (alpha 0.6, beta 0.4 -> 1) and n = 3 are unchanged.
Everything else (5M frames, evaluation every 50k frames, 10 greedy episodes on seeds
1000-1009) is as in the main trial.

## 1. Set up (login node)

```sh
unzip followup.zip && cd followup
mkdir -p logs/rct-followup
export RCT_PYTHON=~/.conda/envs/mltest/bin/python
export RCT_EXPERIMENT_ID=followup-02
$RCT_PYTHON followup.py prepare --root $PWD/experiments/$RCT_EXPERIMENT_ID   # REQUIRED, once
sinfo -p batch -o "%P %D %c %m %l"     # partition exists? cores and memory per node
$RCT_PYTHON followup.py tasks          # array range of each group (4 runs per task)
```

`prepare` creates every output folder before any job starts. Jobs that start at the
same moment and create the same folder on shared storage can fail with
`mkdir: Already exists`; with the folders already there, nothing is created at start-up.

The script defaults to the `batch` partition. To use another one, add `-p bigbatch`
(or any other partition) to `sbatch`. If the nodes have many cores, more runs per task
use them better, for example `export RCT_RUNS_PER_TASK=6`, then rerun
`followup.py tasks` to get the new ranges. **Keep one value of RCT_RUNS_PER_TASK for the
whole experiment**, because it decides which rows each array index trains.

## 2. Submit (priority order)

Use the ranges printed by `followup.py tasks`. With the default of 4 runs per task:

```sh
sbatch --array=0-10%6   slurm/run_followup.slurm   # recipe-sampled   (highest priority)
sbatch --array=11-20%6  slurm/run_followup.slurm   # recipe-reference
sbatch --array=21-34%6  slurm/run_followup.slurm   # lr-low-dqn
sbatch --array=35-72%8  slurm/run_followup.slurm   # seaquest-capped
sbatch --array=73-128%8 slurm/run_followup.slurm   # lr-low-arms (optional)
```

`%6` caps how many nodes run at once (the cluster guidelines ask users not to flood it).
Before training, each run executes the preflight (`verify_arms.py` and its tests); a
failed preflight stops the run.

Rough wall time per task (4 runs side by side, from the main trial's timings, which
used one update per frame): recipe groups about 1–3 h (one update every 4 frames);
lr groups about 4–10 h; Seaquest about 5–12 h, more while greedy episodes hit the cap.

## 3. Monitor and resume

```sh
squeue -u $USER
$RCT_PYTHON followup.py status --root $PWD/experiments/$RCT_EXPERIMENT_ID
tail logs/rct-followup/rows/row0_*.log
```

**Auto-resubmit.** When a task ends with unfinished rows, for whatever reason (a
time-budget stop, a crashed run, or a failure during start-up), it queues the same array
index again, to run after the current job ends with the same partition and time limit.
Completed runs are skipped. A run cannot resume mid-way, so an interrupted run restarts
from frame 0. Each task is resubmitted at most `RCT_RESUBMIT_MAX` times (default 3),
so a run that fails the same way every time stops after 3 attempts; check its row log.
A task you cancel with `scancel` is never resubmitted. `export RCT_AUTO_RESUBMIT=0`
before `sbatch` turns resubmission off. SLURM itself requeues tasks hit by a node
failure (`--requeue`).

`status` lists the array tasks that still have unfinished rows; resubmitting those
indices by hand is safe, because completed runs are skipped and a row that is already
running refuses to start twice.

## 4. Bring the results back

```sh
$RCT_PYTHON followup.py collect --root $PWD/experiments/$RCT_EXPERIMENT_ID --out followup_runs.csv
zip -r followup-output.zip experiments/$RCT_EXPERIMENT_ID logs/rct-followup followup_runs.csv
```

`collect` also prints mean AUC per group, game and arm, and paired differences to DQN.

Layout: `experiments/<id>/<group>/results/<arm>/<game>/<tag>/<arm>_results_<game>.json`,
with the loss and evaluation logs (JSONL + offline W&B) under `.../tracking/...`. Tags
are `iNN` (individual) or `iNN-lr3e-05` / `iNN-lr1e-05`. Seaquest results also record
`eval_truncated_episodes`: how many of the 10 episodes hit the cap at each checkpoint.
