#!/bin/bash
# =============================================================================
# RCT GPU TEST — one variant, three games, short pilot on GPU
# =============================================================================
#
# WHAT THIS DOES (plain English):
#   Same as the CPU test, but uses a GPU to train faster.
#   Good for checking that PyTorch can see the GPU before long runs.
#
# BEFORE SUBMITTING:
#   mkdir -p logs/rct-gpu-test results
#
# SUBMIT:
#   sbatch --export=RCT_VARIANT=control slurm/run_rct_gpu_test.slurm
#
# =============================================================================

#SBATCH --job-name=rct-gpu-test
#SBATCH --partition=bigbatch
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rct-gpu-test/%x_%A_%a.out
#SBATCH --error=logs/rct-gpu-test/%x_%A_%a.err

set -euo pipefail

# SLURM copies the script to /var/lib/slurm/... on compute nodes, so $0
# cannot locate common.sh — use SLURM_SUBMIT_DIR (where sbatch was run).
if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "${SLURM_SUBMIT_DIR}/common.sh"
fi
rct_cd_submit_dir
mkdir -p logs/rct-gpu-test

sleep 5
rct_log_header
rct_setup_gpu
rct_check_python
rct_map_game_only

OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}-gpu-pilot"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       GPU pilot"
echo ""

"$PYTHON" -u run_rct.py \
    --variant "$RCT_VARIANT" \
    --game "$RCT_GAME" \
    --pilot \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
