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
slurm/                  Wits HPC GPU job scripts (biggpu partition)
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

## Cluster setup (Wits HPC — GPU only)

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

### 4. Create log directories

```bash
mkdir -p logs/rct-gpu-test logs/rct-gpu-full results
```

### 5. GPU smoke test (run every variant before the full experiment)

```bash
cd ~/research_project
sbatch --export=RCT_VARIANT=control  slurm/run_rct_gpu_test.slurm
sbatch --export=RCT_VARIANT=ddqn     slurm/run_rct_gpu_test.slurm
sbatch --export=RCT_VARIANT=per      slurm/run_rct_gpu_test.slurm
sbatch --export=RCT_VARIANT=nstep    slurm/run_rct_gpu_test.slurm
sbatch --export=RCT_VARIANT=averaged slurm/run_rct_gpu_test.slurm
```

Monitor:

```bash
squeue -u $USER
tail -f logs/rct-gpu-test/rct-gpu-test_<JOBID>_0.out
```

### 6. Full GPU experiment

Submits 15 array tasks (5 variants × 3 games) on the `biggpu` partition:

```bash
sbatch slurm/run_rct_gpu_full.slurm
```

Each task has a 3-day time limit. If a task is not finished, it saves progress,
exits cleanly, and **automatically resubmits** until all 30 individuals complete.

Results: `results/{variant}/{game}/`  
W&B dashboard: https://wandb.ai/juniormatimu572-wits/DQN%20modifications

## Analysis

```bash
python analyze_rct.py --results-root results --game-slug breakout
```

## Requirements

- Python 3.10+
- PyTorch with CUDA, Gymnasium, MinAtar ≥ 1.0.13, NumPy, SciPy, wandb
