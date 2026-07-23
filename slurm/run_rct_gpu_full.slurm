#!/bin/bash
# =============================================================================
# RCT GPU FULL — all 5 variants × 3 games on biggpu (3-day limit + auto-resubmit)
# =============================================================================
#
# BEFORE SUBMITTING (from project root):
#   export WANDB_API_KEY=...   # set once in ~/.bashrc (never commit the key)
#   mkdir -p logs/rct-gpu-full results
#
# SUBMIT:
#   sbatch slurm/run_rct_gpu_full.slurm
#
# Each array task runs until all 30 individuals finish. If the 3-day SLURM
# limit is reached, the job exits cleanly (saving progress) and resubmits
# itself until the task is complete.
#
# =============================================================================

#SBATCH --job-name=rct-gpu-full
#SBATCH --partition=biggpu
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/rct-gpu-full/%x_%A_%a.out
#SBATCH --error=logs/rct-gpu-full/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "${SLURM_SUBMIT_DIR}/common.sh"
fi

rct_cd_submit_dir
mkdir -p logs/rct-gpu-full

sleep 20
rct_log_header
rct_export_time_budget
rct_setup_gpu
rct_setup_wandb
rct_check_python
rct_map_array_task

export RCT_NUM_INDIVIDUALS=30
OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}"
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       GPU full run (auto-resubmit on 3-day limit)"
echo "W&B entity: ${WANDB_ENTITY}"
echo "W&B project:${WANDB_PROJECT}"
echo ""

set +e
"$PYTHON" -u run_rct.py \
    --variant "$RCT_VARIANT" \
    --game "$RCT_GAME" \
    --num-individuals 30 \
    --total-frames 5000000 \
    --eval-frequency 50000 \
    --num-eval-episodes 10 \
    --output-dir "$OUTPUT_DIR" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT"
RUN_EXIT=$?
set -e

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} ($RCT_VARIANT / $RCT_GAME) finished at $(date)"
echo "Exit code: ${RUN_EXIT}"

# Exit 2 = stopped early for SLURM time limit; 0 = finished normally.
if [[ "${RUN_EXIT}" -eq 0 || "${RUN_EXIT}" -eq 2 ]]; then
    rct_resubmit_if_incomplete "slurm/run_rct_gpu_full.slurm"
fi

exit "${RUN_EXIT}"
