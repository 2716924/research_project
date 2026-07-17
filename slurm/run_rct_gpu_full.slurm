#!/bin/bash
# =============================================================================
# RCT GPU FULL — all 5 variants × 3 games on GPU
# =============================================================================
#
# WHAT THIS DOES (plain English):
#   Full proposal-scale experiment accelerated with GPUs.
#   Uses the cluster's PyTorch module (see hpc-examples/07-default-pytorch).
#
# BEFORE SUBMITTING:
#   mkdir -p logs/rct-gpu-full results
#
# SUBMIT:
#   sbatch slurm/run_rct_gpu_full.slurm
#
# =============================================================================

#SBATCH --job-name=rct-gpu-full
#SBATCH --partition=bigbatch
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rct-gpu-full/%x_%A_%a.out
#SBATCH --error=logs/rct-gpu-full/%x_%A_%a.err
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
mkdir -p logs/rct-gpu-full

sleep 20
rct_log_header
rct_setup_gpu
rct_check_python
rct_map_array_task

OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       GPU full run"
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
