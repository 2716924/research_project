#!/bin/bash
#
# ============================================================
# DQN CPU Test - Wits HPC
# ============================================================

#SBATCH --job-name=dqn_cputest
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dqn_cputest_%A_%a.out
#SBATCH --error=logs/dqn_cputest_%A_%a.err

set -euo pipefail

echo "=========================================="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Array Task ID : ${SLURM_ARRAY_TASK_ID}"
echo "Host          : $(hostname)"
echo "Working Dir   : $(pwd)"
echo "Start Time    : $(date)"
echo "=========================================="

# ------------------------------------------------------------
# Load Anaconda module (required for the cluster environment)
# ------------------------------------------------------------
module purge
module load anaconda

# ------------------------------------------------------------
# Go to project directory
# ------------------------------------------------------------
cd "$HOME/research_project"

mkdir -p logs
mkdir -p results

# ------------------------------------------------------------
# Python interpreter from your Conda environment
# ------------------------------------------------------------
PYTHON="$HOME/.conda/envs/mltest/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found:"
    echo "$PYTHON"
    exit 1
fi

echo ""
echo "Using Python:"
echo "$PYTHON"

"$PYTHON" --version

echo ""
echo "Checking environment..."

"$PYTHON" - <<EOF
import sys
import torch
import gymnasium
import minatar
import numpy
import pandas
import matplotlib

print("Environment OK")
print("Python:", sys.version)
print("Executable:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# ------------------------------------------------------------
# Game selection
# ------------------------------------------------------------
GAMES=(
    "MinAtar/Breakout-v1"
    "MinAtar/Asterix-v1"
    "MinAtar/Freeway-v1"
)

GAME="${GAMES[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Running game: ${GAME}"
echo ""

# ------------------------------------------------------------
# Run the DQN
# ------------------------------------------------------------
"$PYTHON" -u dqn_control_group.py \
    --game "${GAME}" \
    --num-individuals 2 \
    --total-frames 10000 \
    --eval-frequency 5000 \
    --num-eval-episodes 1 \
    --output-dir results

echo ""
echo "Finished at $(date)"