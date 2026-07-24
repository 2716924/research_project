#!/bin/bash
# =============================================================================
# RCT BIGBATCH TEST — one variant, three games, short CPU pilot
# =============================================================================
#
# BEFORE SUBMITTING (once, from project root):
#   mkdir -p logs/rct-bigbatch-test results
#
# SUBMIT:
#   cd ~/research_project
#   sbatch --export=RCT_VARIANT=control slurm/run_rct_bigbatch_test.slurm
#
# =============================================================================

#SBATCH --job-name=rct-bigbatch-test
#SBATCH --partition=bigbatch
#SBATCH --array=0-2
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
mkdir -p logs/rct-bigbatch-test results

if [[ -z "${RCT_VARIANT:-}" ]]; then
    echo "ERROR: RCT_VARIANT must be set, e.g. control, ddqn, per, nstep, averaged"
    exit 1
fi

sleep 5
rct_log_header
rct_export_time_budget
rct_setup_cpu
rct_install_python_deps
rct_setup_wandb
rct_check_python
rct_map_game_only

export RCT_NUM_INDIVIDUALS=3
OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}-pilot"
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

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
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT"
RUN_EXIT=$?
set -e

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date) with exit code ${RUN_EXIT}"
exit "${RUN_EXIT}"
