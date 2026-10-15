#!/bin/bash
# All five variants x five games; see CLUSTER_RUN_GUIDE.md.
# Create logs/rct-bigbatch-test and logs/rct-bigbatch-variant before sbatch.
# Task ID = variant index * 5 + game index (python -m rct.tasks ID).

#SBATCH --job-name=rct-five-games-test
#SBATCH --partition=bigbatch
#SBATCH --array=0-24%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rct-bigbatch-test/%x_%A_%a.out
#SBATCH --error=logs/rct-bigbatch-test/%x_%A_%a.err

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
rct_setup_cpu
rct_map_task
rct_setup_experiment pilot
rct_log_header
rct_export_time_budget
rct_check_python
rct_lock_task

export RCT_NUM_INDIVIDUALS=3

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       CPU pilot (bigbatch)"
echo ""

set +e
"$PYTHON" -u run_rct.py \
    --variant "$RCT_VARIANT" \
    --game "$RCT_GAME" \
    --pilot \
    --output-dir "$OUTPUT_DIR" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --tracking-dir "$RCT_TRACKING_DIR" \
    --experiment-id "${RCT_EXPERIMENT_ID}/pilot"
RUN_EXIT=$?
set -e

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date) with exit code ${RUN_EXIT}"
rct_log_footer "$RUN_EXIT"
exit "${RUN_EXIT}"
