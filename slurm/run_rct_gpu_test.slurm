#!/bin/bash
# =============================================================================
# RCT GPU TEST — one variant, three games, short pilot on biggpu
# =============================================================================
#
# BEFORE SUBMITTING (from project root):
#   export WANDB_API_KEY=...   # set once in ~/.bashrc (never commit the key)
#   mkdir -p logs/rct-gpu-test results
#
# SUBMIT (test each arm):
#   sbatch --export=RCT_VARIANT=control  slurm/run_rct_gpu_test.slurm
#   sbatch --export=RCT_VARIANT=ddqn     slurm/run_rct_gpu_test.slurm
#   sbatch --export=RCT_VARIANT=per      slurm/run_rct_gpu_test.slurm
#   sbatch --export=RCT_VARIANT=nstep    slurm/run_rct_gpu_test.slurm
#   sbatch --export=RCT_VARIANT=averaged slurm/run_rct_gpu_test.slurm
#
# =============================================================================

#SBATCH --job-name=rct-gpu-test
#SBATCH --partition=biggpu
#SBATCH --array=0-2
# GPU resource requests are cluster-specific. Comment this out or replace it with
# whatever your cluster expects if it does not define a generic resource named "gpu".
# SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rct-gpu-test/%x_%A_%a.out
#SBATCH --error=logs/rct-gpu-test/%x_%A_%a.err

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "${SLURM_SUBMIT_DIR}/common.sh"
fi

rct_cd_project_root
mkdir -p logs/rct-gpu-test

sleep 5
rct_log_header
rct_export_time_budget
rct_setup_gpu
rct_install_python_deps
rct_setup_wandb
rct_check_python
rct_map_game_only

export RCT_NUM_INDIVIDUALS=3
OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}-gpu-pilot"
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       GPU pilot"
echo "W&B entity: ${WANDB_ENTITY}"
echo "W&B project:${WANDB_PROJECT}"
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
