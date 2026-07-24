#!/bin/bash
# =============================================================================
# RCT GPU VARIANT — one variant at a time, three games, full run on biggpu
# =============================================================================
#
# Submit one variant at a time to keep the job count lower and avoid QOS limits.
# Auto-resubmission is enabled by default for this launcher so interrupted runs
# can continue without manual re-submission.
#
# EXAMPLE:
#   sbatch --export=RCT_VARIANT=control slurm/run_rct_gpu_variant.slurm
#   sbatch --export=RCT_VARIANT=ddqn    slurm/run_rct_gpu_variant.slurm
#   sbatch --export=RCT_VARIANT=per     slurm/run_rct_gpu_variant.slurm
#   sbatch --export=RCT_VARIANT=nstep   slurm/run_rct_gpu_variant.slurm
#   sbatch --export=RCT_VARIANT=averaged slurm/run_rct_gpu_variant.slurm
#
# =============================================================================

#SBATCH --job-name=rct-gpu-variant
# Use the GPU-capable partition for these runs.
#SBATCH --partition=biggpu
# Some clusters reject explicit GRES requests; leave this commented and rely on
# the GPU-capable partition selection instead.
# SBATCH --gres=gpu:1
# SBATCH --constraint=gpu
# Run only 3 game tasks for one variant, with limited concurrency.
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/rct-gpu-variant/%x_%A_%a.out
#SBATCH --error=logs/rct-gpu-variant/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "${SLURM_SUBMIT_DIR}/common.sh"
fi

rct_cd_submit_dir
PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$PWD}"/.. && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs/rct-gpu-variant

if [[ -z "${RCT_VARIANT:-}" ]]; then
    echo "ERROR: RCT_VARIANT must be set, e.g. control, ddqn, per, nstep, averaged"
    exit 1
fi

sleep 20
rct_log_header
rct_export_time_budget
rct_setup_gpu
rct_setup_wandb
rct_check_python
rct_map_game_only

export RCT_NUM_INDIVIDUALS=30
OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}"
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       GPU full run (one variant at a time)"
echo "Auto-resubmit: 1"
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

if [[ "${RUN_EXIT}" -eq 0 || "${RUN_EXIT}" -eq 2 ]]; then
    rct_resubmit_if_incomplete "slurm/run_rct_gpu_variant.slurm"
fi

exit "${RUN_EXIT}"
