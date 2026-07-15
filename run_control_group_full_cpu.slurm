#!/bin/bash
# ===========================================================
# Control-group (canonical Nature-DQN) RCT trainer -- FULL CPU RUN
#
# Runs on the Wits HPC cluster using the mltest Conda environment.
# ===========================================================

#SBATCH --job-name=dqn-control-cpu
#SBATCH --array=0-2
#SBATCH --partition=stampede
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dqn_control_%A_%a.out
#SBATCH --error=logs/dqn_control_%A_%a.err
#SBATCH --requeue

set -euo pipefail

# ------------------------------------------------------------
# Move to the directory from which sbatch was submitted
# ------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p results

sleep 20

echo "========================================"
echo "Date: $(date)"
echo "User: $USER"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Working Directory: $PWD"
echo "Submit Directory: $SLURM_SUBMIT_DIR"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================"

# ------------------------------------------------------------
# Load Anaconda module
# ------------------------------------------------------------
module purge
module load anaconda

# ------------------------------------------------------------
# This is a CPU-only job (no --gres=gpu requested), but the
# compute node may still have GPUs physically installed and
# visible. Explicitly hide them from torch so it doesn't try
# to use an unsupported/incompatible GPU (e.g. old GTX 1060s
# whose compute capability is no longer built into newer
# torch+cuda wheels).
# ------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""

# ------------------------------------------------------------
# Some compute nodes fail NNPACK's runtime hardware probe even
# though the fallback conv backend works fine. torch.backends.
# nnpack.enabled=False (set in dqn_control_group.py) doesn't
# reliably suppress the resulting warning on every torch build,
# because the warning comes from c10's logging layer, not the
# nnpack-selection flag. Silence c10 WARNING-level logs directly
# so it can't spam the .err log across 5M frames.
# ------------------------------------------------------------
export TORCH_CPP_LOG_LEVEL=ERROR
echo "TORCH_CPP_LOG_LEVEL is set to: '${TORCH_CPP_LOG_LEVEL}'"

# ------------------------------------------------------------
# Use Python directly from your environment
# ------------------------------------------------------------
PYTHON="$HOME/.conda/envs/mltest/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Cannot find Python interpreter:"
    echo "$PYTHON"
    exit 1
fi

echo ""
echo "Using Python:"
echo "$PYTHON"

"$PYTHON" --version

echo ""
echo "Checking Python environment..."

"$PYTHON" - <<EOF
import sys
import torch
import gymnasium
import minatar
import numpy
import pandas
import matplotlib

print("Environment check passed.")
print("Python:", sys.version)
print("Executable:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# ------------------------------------------------------------
# Select MinAtar game
# ------------------------------------------------------------
GAMES=(
    "MinAtar/Breakout-v1"
    "MinAtar/Asterix-v1"
    "MinAtar/Freeway-v1"
)

GAME="${GAMES[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "========================================"
echo "Running game: $GAME"
echo "========================================"

# ------------------------------------------------------------
# Run the experiment
# ------------------------------------------------------------
"$PYTHON" -u dqn_control_group.py \
    --game "$GAME" \
    --num-individuals 30 \
    --total-frames 5000000 \
    --eval-frequency 50000 \
    --num-eval-episodes 10 \
    --output-dir results

echo ""
echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID ($GAME) completed."
echo "Finished at $(date)"
echo "========================================"