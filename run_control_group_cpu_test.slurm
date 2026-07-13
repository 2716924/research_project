#!/bin/bash
# ===========================================================
# Control-group (canonical Nature-DQN) RCT trainer -- CPU SMOKE TEST
#
# Runs on the `stampede` partition (matching the course's working
# examples) at PILOT scale (--pilot: 3 individuals, 15k frames) just to
# confirm the job launches, the conda env + MinAtar/Gymnasium
# registration works, and results get written correctly -- before
# committing to the real, long-running full-scale job.
#
# This is NOT the real experiment. Once this finishes cleanly and you've
# checked results_cpu_test/, move on to run_control_group_full_cpu.slurm.
# ===========================================================
#SBATCH --job-name=dqn-control-cputest
#SBATCH --array=0-2
#SBATCH --partition=stampede
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dqn_cputest_%A_%a.out
#SBATCH --error=logs/dqn_cputest_%A_%a.err

set -euo pipefail

# Move to the directory this job was submitted from, so relative paths
# behave the same regardless of what directory sbatch itself ran in.
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p results_cpu_test

# We pause execution briefly, so you get a chance to see your job in the
# queue. If you don't see your job in the queue at all, execution has not
# even started, and there is likely something wrong with your #SBATCH
# lines above.
sleep 20

# We print out some useful debugging information.
echo "=== Useful Debugging Information ==="
echo "Date: "$(date)
echo "Username: "$USER
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Home Directory: "$HOME
echo "Node: "$HOSTNAME
echo "Current Working Directory: "$PWD
echo "Job Submitted From Machine: "$SLURM_SUBMIT_HOST
echo "Job Submitted From Directory: "$SLURM_SUBMIT_DIR
echo "Array Task: "$SLURM_ARRAY_TASK_ID
echo "======"

# -------------------------
# Load software environment
# -------------------------
source /etc/profile.d/lmod.sh
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
echo "Game: $GAME"

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
