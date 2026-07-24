# DQN Randomised Control Trial (Honours Research)

Rigorous benchmarking of DQN modifications using a Randomised Control Trial
(RCT) methodology, as described in the research proposal.

## Repository layout

```
rct/                    Core library (trainer, replay, targets, evaluation)
run_rct.py              Main CLI — run one algorithmic arm
analyze_rct.py          Paired t-tests + Bonferroni (Phase 4)
verify_arms.py          Automated checks for all five algorithmic arms
check_env.py            Cluster environment check
slurm/                  Wits HPC job scripts
docs/EXPLAINER.md       Non-technical project guide
docs/REFERENCES.md      Trusted implementation sources
results/                JSON outputs (gitignored)
logs/                   SLURM logs (gitignored)
wandb/                  Local wandb runs (gitignored)
```

## Algorithmic arms

| `--variant` | Description |
|-------------|-------------|
| `control` | Canonical Nature-DQN (baseline) |
| `ddqn` | Double DQN — Van Hasselt et al. 2016 |
| `per` | Prioritised Experience Replay — Schaul et al. 2015 |
| `nstep` | n-step learning — Hernandez-Garcia & Sutton 2019 |
| `averaged` | Averaged DQN — Anschel et al. 2017 |

## Cluster setup (Wits HPC — bigbatch CPU)

### 1. Sync code and install dependencies

```bash
cd ~/research_project
git pull
module load anaconda
~/.conda/envs/mltest/bin/pip install -r requirements.txt
```

### 2. Set Weights & Biases API key (once)

**Do not commit your API key.** Add this to `~/.bashrc` on the cluster:

```bash
export WANDB_API_KEY="your-key-here"
export WANDB_ENTITY="juniormatimu572-wits"
export WANDB_PROJECT="DQN modifications"
```

Then run `source ~/.bashrc`.

### 3. Verify environment

```bash
~/.conda/envs/mltest/bin/python check_env.py
~/.conda/envs/mltest/bin/python verify_arms.py
```

### 4. Create log directories (from project root)

```bash
cd ~/research_project
mkdir -p logs/rct-bigbatch-test logs/rct-bigbatch-variant results
```

### 5. CPU smoke test on bigbatch (run every variant first)

Each submission launches **3 array tasks** (Breakout, Asterix, Freeway) for one variant:

```bash
cd ~/research_project
sbatch --export=RCT_VARIANT=control  slurm/run_rct_bigbatch_test.slurm
sbatch --export=RCT_VARIANT=ddqn     slurm/run_rct_bigbatch_test.slurm
sbatch --export=RCT_VARIANT=per      slurm/run_rct_bigbatch_test.slurm
sbatch --export=RCT_VARIANT=nstep    slurm/run_rct_bigbatch_test.slurm
sbatch --export=RCT_VARIANT=averaged slurm/run_rct_bigbatch_test.slurm
```

Monitor:

```bash
squeue -u $USER
tail -f logs/rct-bigbatch-test/rct-bigbatch-test_<JOBID>_0.out
```

### 6. Full CPU experiment (one variant at a time, 3 games in parallel)

```bash
cd ~/research_project
sbatch --export=RCT_VARIANT=control  slurm/run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=ddqn     slurm/run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=per      slurm/run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=nstep    slurm/run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=averaged slurm/run_rct_bigbatch_variant.slurm
```

Each job uses the **bigbatch** partition (CPU, 3-day limit). Array tasks 0–2 run Breakout, Asterix, and Freeway simultaneously for the chosen variant. If a task hits the time limit, it saves progress and auto-resubmits.

Results: `results/{variant}/{game}/`  
W&B dashboard: https://wandb.ai/juniormatimu572-wits/DQN%20modifications

### Optional: GPU runs on biggpu

See `slurm/run_rct_gpu_test.slurm` and `slurm/run_rct_gpu_variant.slurm`.

## Analysis

```bash
python analyze_rct.py --results-root results --game-slug breakout
```

## Requirements

- Python 3.10+
- PyTorch, Gymnasium, MinAtar ≥ 1.0.13, NumPy, SciPy, wandb
