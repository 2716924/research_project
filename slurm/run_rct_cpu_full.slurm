#!/bin/bash
# =============================================================================
# RCT CPU FULL — all 5 variants × 3 games, proposal-scale training
# =============================================================================
#
# WHAT THIS DOES (plain English):
#   Runs the full experiment on CPU: 30 hyperparameter individuals,
#   5 million training frames each, for every algorithm variant and game.
#   This is the main data-collection job for the research project.
#
# BEFORE SUBMITTING:
#   mkdir -p logs/rct-cpu-full results
#
# SUBMIT:
#   sbatch slurm/run_rct_cpu_full.slurm
#
# Array tasks 0-14 map to:
#   control/ddqn/per/nstep/averaged × breakout/asterix/freeway
#
# =============================================================================

#SBATCH --job-name=rct-cpu-full
#SBATCH --partition=stampede
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=logs/rct-cpu-full/%x_%A_%a.out
#SBATCH --error=logs/rct-cpu-full/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail

# SLURM copies the script to /var/lib/slurm/... on compute nodes, so $0
# cannot locate common.sh — use SLURM_SUBMIT_DIR (where sbatch was run).
if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "${SLURM_SUBMIT_DIR}/common.sh"
fi
rct_cd_submit_dir
mkdir -p logs/rct-cpu-full

sleep 20
rct_log_header
rct_setup_cpu
rct_check_python
rct_map_array_task

OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       full (30 individuals × 5M frames)"
echo ""

"$PYTHON" -u run_rct.py \
    --variant "$RCT_VARIANT" \
    --game "$RCT_GAME" \
    --num-individuals 30 \
    --total-frames 5000000 \
    --eval-frequency 50000 \
    --num-eval-episodes 10 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} ($RCT_VARIANT / $RCT_GAME) finished at $(date)"
