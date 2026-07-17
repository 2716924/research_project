#!/bin/bash
# =============================================================================
# RCT CPU TEST — one variant, three games, short pilot run
# =============================================================================
#
# WHAT THIS DOES (plain English):
#   Runs a quick practice version of the experiment on the cluster CPU.
#   It trains the agent for a short time on Breakout, Asterix, and Freeway
#   to check that everything works before the full 5-million-frame runs.
#
# BEFORE SUBMITTING (run once on the login node):
#   mkdir -p logs/rct-cpu-test results
#
# SUBMIT:
#   sbatch --export=RCT_VARIANT=control slurm/run_rct_cpu_test.slurm
#   sbatch --export=RCT_VARIANT=ddqn    slurm/run_rct_cpu_test.slurm
#
# =============================================================================

#SBATCH --job-name=rct-cpu-test
#SBATCH --partition=bigbatch
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rct-cpu-test/%x_%A_%a.out
#SBATCH --error=logs/rct-cpu-test/%x_%A_%a.err

set -euo pipefail

# SLURM copies the script to /var/lib/slurm/... on compute nodes, so $0
# cannot locate common.sh — use SLURM_SUBMIT_DIR (where sbatch was run).
if [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "${SLURM_SUBMIT_DIR}/common.sh"
fi
rct_cd_submit_dir
mkdir -p logs/rct-cpu-test

sleep 5
rct_log_header
rct_setup_cpu
rct_check_python
rct_map_game_only

OUTPUT_DIR="results/${RCT_VARIANT}/${RCT_GAME_SLUG}-pilot"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Variant:    $RCT_VARIANT"
echo "Game:       $RCT_GAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       pilot (short smoke test)"
echo ""

"$PYTHON" -u run_rct.py \
    --variant "$RCT_VARIANT" \
    --game "$RCT_GAME" \
    --pilot \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
