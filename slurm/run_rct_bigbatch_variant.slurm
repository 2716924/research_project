#!/bin/bash
# =============================================================================
# RCT BIGBATCH — one variant at a time, three games in parallel (CPU)
# =============================================================================
#
# Partition: bigbatch (CPU-only, 3-day limit)
#
# BEFORE SUBMITTING (once, from project root):
#   mkdir -p logs/rct-bigbatch-variant results
#
# SUBMIT one variant at a time (launches 3 array tasks: breakout/asterix/freeway):
#   cd ~/research_project
#   sbatch --export=RCT_VARIANT=control  slurm/run_rct_bigbatch_variant.slurm
#   sbatch --export=RCT_VARIANT=ddqn     slurm/run_rct_bigbatch_variant.slurm
#   sbatch --export=RCT_VARIANT=per      slurm/run_rct_bigbatch_variant.slurm
#   sbatch --export=RCT_VARIANT=nstep    slurm/run_rct_bigbatch_variant.slurm
#   sbatch --export=RCT_VARIANT=averaged slurm/run_rct_bigbatch_variant.slurm
#
# =============================================================================

#SBATCH --job-name=rct-bigbatch
#SBATCH --partition=bigbatch
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/rct-bigbatch-variant/%x_%A_%a.out
#SBATCH --error=logs/rct-bigbatch-variant/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
elif [[ -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
else
    echo "ERROR: slurm/common.sh not found"
    exit 1
fi

rct_cd_project_root
mkdir -p logs/rct-bigbatch-variant results 2>/dev/null

if [[ -z "${RCT_VARIANT:-}" ]]; then
    echo "ERROR: RCT_VARIANT must be set, e.g. control, ddqn, per, nstep, averaged"
    exit 1
fi

export RCT_AUTO_RESUBMIT="${RCT_AUTO_RESUBMIT:-1}"

sleep 20
rct_log_header
rct_export_time_budget
rct_setup_cpu
rct_install_python_deps
rct_check_python
rct_map_game_only

export RCT_NUM_INDIVIDUALS=30
OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}"
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR" 2>/dev/null

echo ""
echo "Variant:       $RCT_VARIANT"
echo "Game:          $RCT_GAME"
echo "Output dir:    $OUTPUT_DIR"
echo "Mode:          CPU full run (bigbatch)"
echo "Auto-resubmit: $RCT_AUTO_RESUBMIT"
echo ""

set +e
"$PYTHON" -u run_rct.py \
    --variant "$RCT_VARIANT" \
    --game "$RCT_GAME" \
    --num-individuals 30 \
    --total-frames 5000000 \
    --eval-frequency 50000 \
    --num-eval-episodes 10 \
    --output-dir "$OUTPUT_DIR"
RUN_EXIT=$?
set -e

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} ($RCT_VARIANT / $RCT_GAME) finished at $(date)"
echo "Exit code: ${RUN_EXIT}"

if [[ "${RUN_EXIT}" -eq 0 || "${RUN_EXIT}" -eq 2 ]]; then
    rct_resubmit_if_incomplete "slurm/run_rct_bigbatch_variant.slurm"
fi

exit "${RUN_EXIT}"
