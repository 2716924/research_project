# DQN Randomised Control Trial (Honours Research)

Rigorous benchmarking of DQN modifications using a Randomised Control Trial
(RCT) methodology, as described in the research proposal.

## Repository layout

```
rct/                    Core library (trainer, replay, targets, evaluation)
run_rct.py              Main CLI — run one algorithmic arm
dqn_control_group.py    Backward-compatible control-group entry point
analyze_rct.py          Paired t-tests + Bonferroni (Phase 4)
check_env.py            Cluster environment check
slurm/                  Wits HPC job scripts (test + full, CPU + GPU)
docs/EXPLAINER.md       Non-technical project guide
docs/REFERENCES.md      Trusted implementation sources
results/                JSON outputs (gitignored)
logs/                   SLURM logs (gitignored)
```

## Algorithmic arms

| `--variant` | Description |
|-------------|-------------|
| `control` | Canonical Nature-DQN (baseline) |
| `ddqn` | Double DQN — Van Hasselt et al. 2016 |
| `per` | Prioritised Experience Replay — Schaul et al. 2015 |
| `nstep` | n-step learning — Hernandez-Garcia & Sutton 2019 |
| `averaged` | Averaged DQN — Anschel et al. 2017 |

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python check_env.py
python run_rct.py --variant control --pilot
python run_rct.py --variant ddqn --pilot
```

## Cluster (Wits HPC)

```bash
module load anaconda
~/.conda/envs/mltest/bin/python check_env.py

mkdir -p logs/rct-cpu-test results
sbatch --export=RCT_VARIANT=control slurm/run_rct_cpu_test.slurm
```

See `docs/EXPLAINER.md` for a plain-English walkthrough.

## Analysis

```bash
python analyze_rct.py --results-root results --game-slug breakout
```

## Requirements

- Python 3.10+
- PyTorch, Gymnasium, MinAtar ≥ 1.0.13, NumPy, SciPy
