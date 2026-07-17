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
