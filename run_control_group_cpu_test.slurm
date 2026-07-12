#!/bin/bash
# ===========================================================
# Control-group (canonical Nature-DQN) RCT trainer -- CPU SMOKE TEST
#
# Runs on the plain `batch` partition (no GPU request) at PILOT scale
# (--pilot: 3 individuals, 15k frames) just to confirm the job launches,
# the conda env + MinAtar/Gymnasium registration works, and results get
# written correctly -- before you spend GPU-hours on the real run.
#
# This is NOT the real experiment. Once this finishes cleanly and you've
# checked results/, move on to run_control_group_gpu.slurm.
# ===========================================================
#SBATCH --job-name=dqn-control-cputest
#SBATCH --array=0-2
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dqn_cputest_%A_%a.out
#SBATCH --error=logs/dqn_cputest_%A_%a.err

set -euo pipefail
mkdir -p logs
mkdir -p results_cpu_test

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
echo "CPU SMOKE TEST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Game: $GAME"
echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-Not Set (expected -- this is CPU-only)}"
echo "=========================================="

# First, confirm the environment itself is sound (imports, MinAtar
# registration, etc.) before even trying to train anything.
python check_env.py

# -------------------------
# Run a tiny pilot-scale job
# -------------------------
python -u dqn_control_group.py \
    --pilot \
    --game "$GAME" \
    --output-dir results_cpu_test

echo "Task $SLURM_ARRAY_TASK_ID ($GAME) CPU smoke test finished."
