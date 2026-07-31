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

```

## Algorithmic arms

| `--variant` | Description |
|-------------|-------------|
| `control` | Canonical Nature-DQN (baseline) |
| `ddqn` | Double DQN — Van Hasselt et al. 2016 |
| `per` | Prioritised Experience Replay — Schaul et al. 2015 |
| `nstep` | n-step learning — Hernandez-Garcia & Sutton 2019 |
| `averaged` | Averaged DQN — Anschel et al. 2017 |



### Full CPU experiment (one variant at a time, 3 games in parallel)

```bash
cd ~/research_project
sbatch --export=RCT_VARIANT=control  run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=ddqn     run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=per      run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=nstep    run_rct_bigbatch_variant.slurm
sbatch --export=RCT_VARIANT=averaged run_rct_bigbatch_variant.slurm
```

Each job uses the **bigbatch** partition (CPU, 3-day limit). Array tasks 0–2 run Breakout, Asterix, and Freeway simultaneously for the chosen variant. If a task hits the time limit, it saves progress and auto-resubmits.

Results: `results/{variant}/{game}/`  
### Optional: GPU runs on biggpu

See `slurm/run_rct_gpu_test.slurm` and `slurm/run_rct_gpu_variant.slurm`.
