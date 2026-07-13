#!/bin/bash
# ===========================================================
# Control-group (canonical Nature-DQN) RCT trainer -- FULL CPU RUN
#
# GPU abandoned: MinAtar's observations are tiny (10x10xC), so this
# workload doesn't need one, and this cluster's CPUs are strong enough.
# Runs on the `stampede` partition (matching the course's working
# examples) at full spec scale (30 individuals x 5,000,000 frames
# per game).
#
# Run run_control_group_cpu_test.slurm FIRST if you haven't already --
# it does the same environment/registration checks at pilot scale, and
# its timing tells you whether --time below is set sensibly for the
# full run (see note near --time).
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

# Move to the directory this job was submitted from, so relative paths
# (mkdir results, python dqn_control_group.py, etc.) behave the same
# regardless of what directory sbatch itself happened to run in.
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p results

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
echo "CPUs allocated: "$SLURM_CPUS_PER_TASK
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

# -------------------------
# Run training (CPU only -- dqn_control_group.py already falls back to
# CPU automatically since torch.cuda.is_available() will be False here;
# nothing to change in the Python code)
# -------------------------
python -u dqn_control_group.py \
    --game "$GAME" \
    --num-individuals 30 \
    --total-frames 5000000 \
    --eval-frequency 50000 \
    --num-eval-episodes 10 \
    --output-dir results

echo "Task $SLURM_ARRAY_TASK_ID ($GAME) finished."
