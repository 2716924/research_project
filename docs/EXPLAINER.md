# Explaining the project (non-technical guide)

Use this document when presenting the project to supervisors, examiners,
or anyone who does not write code.

## The big picture

We are running a **fair experiment** to find out which small improvements
to a famous AI learning algorithm (DQN) actually help, and by how much.

Think of it like a **clinical drug trial**, but for machine-learning
algorithms instead of medicines.

## The five groups

Every "participant" in our trial is not a person — it is a **settings
profile** (learning speed, memory size, etc.) drawn at random from a
fixed list of options.

Each profile is tested under **five versions** of the same AI:

| Name | Plain-English idea | What changes |
|------|-------------------|--------------|
| **Control** | The original recipe | Nothing — baseline |
| **Double DQN** | Ask two experts instead of one | How the AI estimates future reward |
| **PER** | Replay important mistakes more often | Which past experiences get reviewed |
| **n-step** | Look further ahead before updating | How many steps of reward are combined |
| **Averaged DQN** | Average several past "expert" networks | How the target score is smoothed |

**Key rule:** only **one** thing changes per treatment. Everything else
(network shape, games, evaluation) stays the same. That way, if one
version scores higher, we know *which change* caused it.

## The three games

We use simplified Atari-style games (MinAtar):

- **Breakout** — timing and prediction (dense rewards)
- **Asterix** — fast reactions (dynamic enemies)
- **Freeway** — patience (sparse rewards)

Together they test whether an improvement works across different styles
of challenge, not just one lucky game.

## How we measure success

We do **not** just report the single best score. Instead:

1. During training, we pause every 50,000 steps and test the AI fairly
   (no random guessing).
2. We plot performance over time and compute the **area under the curve
   (AUC)** — higher AUC means the AI learned faster and more reliably.
3. After 30+ settings profiles, we run **paired statistical tests** to
   see if differences are real or just noise.

## What each folder does

```
research_project/
├── rct/              # The experiment "engine" (shared code)
├── run_rct.py        # Start one algorithmic arm
├── analyze_rct.py    # Statistics after results are collected
├── check_env.py      # One-time cluster health check
├── slurm/            # Cluster job scripts (test + full, CPU + GPU)
├── docs/             # Guides like this one
└── results/          # Output data (created when jobs run)
```

## Cluster workflow (simple steps)

1. **Check setup** — make sure Python, PyTorch, and games work:
   `~/.conda/envs/mltest/bin/python check_env.py`

2. **Practice run** — short test on CPU:
   `sbatch --export=RCT_VARIANT=control slurm/run_rct_cpu_test.slurm`

3. **Full experiment** — all variants and games:
   `sbatch slurm/run_rct_cpu_full.slurm`
   (or `run_rct_gpu_full.slurm` if using GPUs)

4. **Analyse** — compare treatment vs control:
   `python analyze_rct.py --results-root results --game-slug breakout`

## Test vs full jobs

| Script | Purpose | Scale |
|--------|---------|-------|
| `run_rct_cpu_test.slurm` | Practice / debugging | 3 profiles, ~15k frames |
| `run_rct_cpu_full.slurm` | Real data collection | 30 profiles, 5M frames |
| `run_rct_gpu_test.slurm` | GPU smoke test | Same as CPU test, faster |
| `run_rct_gpu_full.slurm` | GPU full run | Same as CPU full, faster |

## Resuming interrupted jobs

If the cluster stops a job halfway, you can resubmit the same script.
The code skips individuals that already finished and continues from
where it left off.
