#!/bin/bash
# =============================================================================
# Shared setup helpers for Wits HPC SLURM jobs
# =============================================================================
# These functions are sourced by the run_*.slurm scripts in this folder.
# They follow the patterns from the cluster's hpc-examples:
#   - CPU jobs: Anaconda + conda run -n mltest (no "conda activate" needed)
#   - GPU jobs: PyTorch module or CUDA-enabled conda env
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
    echo "========================================"
}

rct_cd_submit_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        cd "$SLURM_SUBMIT_DIR"
    fi
}

# CPU path — matches hpc-examples/06-default-anaconda-with-local-env
rct_setup_cpu() {
    source /etc/profile.d/lmod.sh
    module purge
    module load anaconda

    export CUDA_VISIBLE_DEVICES=""
    export TORCH_CPP_LOG_LEVEL=ERROR

    PYTHON="$HOME/.conda/envs/mltest/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: mltest environment not found at $PYTHON"
        echo "Create it with: conda create -n mltest python=3.11 -y"
        exit 1
    fi
}

# GPU path — matches hpc-examples/07-default-pytorch
rct_setup_gpu() {
    source /etc/profile.d/lmod.sh
    module purge
    module load pytorch/2026-02-24 2>/dev/null || module load anaconda

    export TORCH_CPP_LOG_LEVEL=ERROR

    # Prefer cluster PyTorch module; fall back to mltest conda env
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
