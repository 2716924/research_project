#!/bin/bash
# =============================================================================
# Shared setup helpers for Wits HPC SLURM GPU jobs (biggpu partition)
# =============================================================================

rct_log_header() {
    echo "========================================"
    echo "Date:              $(date)"
    echo "User:              $USER"
    echo "Job ID:            ${SLURM_JOB_ID:-local}"
    echo "Job Name:          ${SLURM_JOB_NAME:-local}"
    echo "Array Task ID:     ${SLURM_ARRAY_TASK_ID:-0}"
    echo "Node:              $(hostname)"
    echo "Working Directory: $PWD"
    echo "Submit Directory:  ${SLURM_SUBMIT_DIR:-$PWD}"
    echo "CPUs per task:     ${SLURM_CPUS_PER_TASK:-1}"
    echo "End time:          ${SLURM_JOB_END_TIME:-n/a}"
    echo "========================================"
}

rct_cd_submit_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        cd "$SLURM_SUBMIT_DIR"
    fi
}

rct_setup_gpu() {
    source /etc/profile.d/lmod.sh
    module purge
    module load pytorch/2026-02-24 2>/dev/null || module load anaconda

    export TORCH_CPP_LOG_LEVEL=ERROR

    if command -v python &>/dev/null && python -c "import torch" &>/dev/null; then
        PYTHON="$(command -v python)"
    else
        PYTHON="$HOME/.conda/envs/mltest/bin/python"
    fi

    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: No usable Python with PyTorch found."
        exit 1
    fi
}

rct_setup_wandb() {
    export WANDB_ENTITY="${WANDB_ENTITY:-juniormatimu572-wits}"
    export WANDB_PROJECT="${WANDB_PROJECT:-DQN modifications}"

    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "WARNING: WANDB_API_KEY is not set — wandb will run in offline mode."
        export WANDB_MODE=offline
    fi
}

rct_export_time_budget() {
    export RCT_TIME_LIMIT_BUFFER="${RCT_TIME_LIMIT_BUFFER:-900}"

    local end_time="${SLURM_JOB_END_TIME:-}"
    if [[ -z "$end_time" || "$end_time" == "Unknown" || "$end_time" == "N/A" ]]; then
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            end_time=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | awk -F= '/EndTime=/{print $2}' | awk '{print $1}')
        fi
    fi

    if [[ -n "$end_time" && "$end_time" != "Unknown" && "$end_time" != "N/A" ]]; then
        export RCT_JOB_END_EPOCH=$(date -d "${end_time}" +%s 2>/dev/null || true)
        echo "Job end time: ${end_time} (stop ${RCT_TIME_LIMIT_BUFFER}s before)"
    fi
}

rct_check_python() {
    echo ""
    echo "Using Python: $PYTHON"
    "$PYTHON" --version
    echo ""
    "$PYTHON" - <<'EOF'
import sys
import torch
import gymnasium
import minatar

print("Environment check passed.")
print("Python:     ", sys.version.split()[0])
print("Executable: ", sys.executable)
print("Torch:      ", torch.__version__)
print("CUDA:       ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:        ", torch.cuda.get_device_name(0))
EOF
}

# Map array task ID (0-14) to variant + game
#   tasks 0-2   -> control  × 3 games
#   tasks 3-5   -> ddqn     × 3 games
#   tasks 6-8   -> per      × 3 games
#   tasks 9-11  -> nstep    × 3 games
#   tasks 12-14 -> averaged × 3 games
rct_map_array_task() {
    local VARIANTS=(control ddqn per nstep averaged)
    local GAMES=("MinAtar/Breakout-v1" "MinAtar/Asterix-v1" "MinAtar/Freeway-v1")
    local GAME_SLUGS=(breakout asterix freeway)

    local TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    VARIANT_IDX=$(( TASK_ID / 3 ))
    GAME_IDX=$(( TASK_ID % 3 ))

    export RCT_VARIANT="${VARIANTS[$VARIANT_IDX]}"
    export RCT_GAME="${GAMES[$GAME_IDX]}"
    export RCT_GAME_SLUG="${GAME_SLUGS[$GAME_IDX]}"
}

# Map array task ID (0-2) to game only (single variant passed via RCT_VARIANT)
rct_map_game_only() {
    local GAMES=("MinAtar/Breakout-v1" "MinAtar/Asterix-v1" "MinAtar/Freeway-v1")
    local GAME_SLUGS=(breakout asterix freeway)

    local TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    export RCT_GAME="${GAMES[$TASK_ID]}"
    export RCT_GAME_SLUG="${GAME_SLUGS[$TASK_ID]}"
    export RCT_VARIANT="${RCT_VARIANT:-control}"
}

# Return 0 when this array task still has unfinished individuals.
rct_task_incomplete() {
    "$PYTHON" - <<'EOF'
import os
import sys

from rct.results import load_existing_results, result_path

output_dir = os.environ["OUTPUT_DIR"]
variant = os.environ["RCT_VARIANT"]
game_id = os.environ["RCT_GAME"]
num_individuals = int(os.environ.get("RCT_NUM_INDIVIDUALS", "30"))

path = result_path(output_dir, variant, game_id)
_, completed = load_existing_results(path)
sys.exit(0 if len(completed) < num_individuals else 1)
EOF
}

# Resubmit the same array task when work remains (used after 3-day limit).
rct_resubmit_if_incomplete() {
    local SLURM_SCRIPT="$1"

    if rct_task_incomplete; then
        echo ""
        echo "Work remains for task ${SLURM_ARRAY_TASK_ID} — resubmitting..."
        sbatch \
            --dependency=afterany:"${SLURM_JOB_ID}" \
            --array="${SLURM_ARRAY_TASK_ID}" \
            --export=ALL \
            "${SLURM_SUBMIT_DIR}/${SLURM_SCRIPT}"
    else
        echo ""
        echo "All individuals complete for task ${SLURM_ARRAY_TASK_ID}."
    fi
}
