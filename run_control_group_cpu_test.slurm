#!/bin/bash
#
# ============================================================
# DQN CPU Test - Wits HPC
# ============================================================
#
# IMPORTANT -- run this ONCE, from the same directory you will
# `sbatch` this script from, BEFORE submitting:
#
#   mkdir -p logs/dqn_cputest results
#
# Why: SLURM opens the --output/--error files at job *dispatch*
# time (before this script's own commands, including `mkdir -p`,
# ever run). If the "logs/dqn_cputest" directory doesn't already
# exist at submission time, SLURM silently fails to create the
# log files -- the job shows as having run, but you get no .out
# or .err output anywhere. The `mkdir -p logs` line later in this
# script is too late to fix that; it only helps the results/ dir,
# which is created at *runtime* by this script, not by SLURM.
# ============================================================

#SBATCH --job-name=dqn_cputest
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dqn_cputest/%x_%A_%a.out
#SBATCH --error=logs/dqn_cputest/%x_%A_%a.err

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
# This is a CPU-only job (no --gres=gpu requested), but the
# compute node may still have GPUs physically installed and
# visible. Explicitly hide them from torch so it doesn't try
# to use an unsupported/incompatible GPU (e.g. old GTX 1060s
# whose compute capability is no longer built into newer
# torch+cuda wheels).
# ------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""

# Silence c10 WARNING-level logs (e.g. the repeated NNPACK
# "Unsupported hardware" warning) at the source -- see full_cpu
# script for details.
export TORCH_CPP_LOG_LEVEL=ERROR

# ------------------------------------------------------------
# Go to project directory
# ------------------------------------------------------------
cd "$HOME/research_project"

# Safety net for local (non-SLURM-managed) paths only -- does
# NOT retroactively fix SLURM's own --output/--error files if
# their parent dir was missing at submission time (see notice
# at the top of this file).
mkdir -p logs/dqn_cputest

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
# Game selection -- each array task also gets its own results
# subfolder so runs never collide and are easy to find.
# ------------------------------------------------------------
GAMES=(
    "MinAtar/Breakout-v1"
    "MinAtar/Asterix-v1"
    "MinAtar/Freeway-v1"
)
GAME_SLUGS=(
    "breakout"
    "asterix"
    "freeway"
)

GAME="${GAMES[$SLURM_ARRAY_TASK_ID]}"
GAME_SLUG="${GAME_SLUGS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="results/cputest/${GAME_SLUG}"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running game: ${GAME}"
echo "Output dir  : ${OUTPUT_DIR}"
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
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Finished at $(date)"
