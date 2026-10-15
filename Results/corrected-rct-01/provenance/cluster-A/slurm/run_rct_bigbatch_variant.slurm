#!/bin/bash
# All five variants x five games; see CLUSTER_RUN_GUIDE.md.
# Create logs/rct-bigbatch-test and logs/rct-bigbatch-variant before sbatch.
# Task ID = variant index * 5 + game index (python -m rct.tasks ID).
# Cluster A: every incomplete control, ddqn and per cell (tasks 4-14), from the start of the array.
# Seaquest tasks (4, 9, 14) are included on request. Greedy evaluation can stay at the
# surface forever; if it does, the 100,000-step guard fails that task (exit 1, no resubmit).

#SBATCH --job-name=rct-five-games-A
#SBATCH --partition=bigbatch
#SBATCH --array=4-14%11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/rct-bigbatch-variant/%x_%A_%a.out
#SBATCH --error=logs/rct-bigbatch-variant/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail

# Resume the existing experiment even if RCT_EXPERIMENT_ID was not exported.
export RCT_EXPERIMENT_ID="${RCT_EXPERIMENT_ID:-corrected-rct-01}"

if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
elif [[ -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
else
    echo "ERROR: slurm/common.sh not found"
    exit 1
fi

rct_cd_project_root
rct_setup_cpu
rct_map_task
rct_setup_experiment full
rct_log_header
rct_export_time_budget
rct_check_python
rct_lock_task

export RCT_NUM_INDIVIDUALS=30
export RCT_AUTO_RESUBMIT="${RCT_AUTO_RESUBMIT:-1}"

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
    --output-dir "$OUTPUT_DIR" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --tracking-dir "$RCT_TRACKING_DIR" \
    --experiment-id "${RCT_EXPERIMENT_ID}/full"
RUN_EXIT=$?
set -e

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} ($RCT_VARIANT / $RCT_GAME) finished at $(date)"
echo "Exit code: ${RUN_EXIT}"

if [[ "${RUN_EXIT}" -eq 0 || "${RUN_EXIT}" -eq 2 ]]; then
    rct_resubmit_if_incomplete "slurm/run_rct_bigbatch_variant.slurm"
fi

rct_log_footer "$RUN_EXIT"
exit "${RUN_EXIT}"
