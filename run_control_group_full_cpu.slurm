#!/bin/bash
# ===========================================================
# Control-group (canonical Nature-DQN) RCT trainer -- FULL CPU RUN
#
# GPU abandoned: MinAtar's observations are tiny (10x10xC), so this
# workload doesn't need one, and this cluster's CPUs are strong enough.
# Runs on the plain `batch` partition at full spec scale
# (30 individuals x 5,000,000 frames per game).
#
# Run run_control_group_cpu_test.slurm FIRST if you haven't already --
# it does the same environment/registration checks at pilot scale, and
# its timing tells you whether --time below is set sensibly for the
# full run (see note near --time).
# ===========================================================
#SBATCH --job-name=dqn-control-cpu
#SBATCH --array=0-2
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dqn_control_%A_%a.out
#SBATCH --error=logs/dqn_control_%A_%a.err
#SBATCH --requeue

set -euo pipefail
mkdir -p logs
mkdir -p results

# -------------------------
# Load software environment
# -------------------------
module purge
module load anaconda/anaconda-2025-12-2

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate your environment -- replace "dqn-rct" with your actual env name.
conda activate dqn-rct

# -------------------------
# Select MinAtar game
# -------------------------
GAMES=(
    "MinAtar/Breakout-v1"
    "MinAtar/Asterix-v1"
    "MinAtar/Freeway-v1"
)
GAME="${GAMES[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Game: $GAME"
echo "Running on: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# -------------------------
# Run training (CPU only -- dqn_control_group.py already falls back to
# CPU automatically since torch.cuda.is_available() will be False on
# the batch partition; nothing to change in the Python code)
# -------------------------
python -u dqn_control_group.py \
    --game "$GAME" \
    --num-individuals 30 \
    --total-frames 5000000 \
    --eval-frequency 50000 \
    --num-eval-episodes 10 \
    --output-dir results

echo "Task $SLURM_ARRAY_TASK_ID ($GAME) finished."
