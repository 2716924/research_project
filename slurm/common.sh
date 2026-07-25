#!/bin/bash
# =============================================================================
# Shared setup helpers for Wits HPC SLURM jobs (CPU-only, bigbatch partition)
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

# Find the repo root whether sbatch was run from ~/research_project or ~/research_project/slurm.
rct_resolve_project_root() {
    local submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"

    if [[ -f "${submit_dir}/run_rct.py" ]]; then
        printf '%s\n' "${submit_dir}"
        return 0
    fi

    if [[ -f "${submit_dir}/../run_rct.py" ]]; then
        cd "${submit_dir}/.." && pwd
        return 0
    fi

    echo "ERROR: cannot locate run_rct.py from submit dir: ${submit_dir}" >&2
    exit 1
}

rct_cd_project_root() {
    export PROJECT_ROOT="$(rct_resolve_project_root)"
    cd "$PROJECT_ROOT"
}

rct_setup_cpu() {
    source /etc/profile.d/lmod.sh 2>/dev/null || true
    module purge 2>/dev/null || true
    module load anaconda 2>/dev/null || true

    export RCT_DEVICE=cpu
    export CUDA_VISIBLE_DEVICES=""
    export TORCH_CUDA_ARCH_LIST=""
    export TORCH_CPP_LOG_LEVEL=ERROR

    PYTHON="$HOME/.conda/envs/mltest/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: mltest environment not found at $PYTHON"
        echo "Create it with: conda create -n mltest python=3.11 -y"
        exit 1
    fi
}

rct_install_python_deps() {
    if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        echo "Installing Python requirements from requirements.txt"
        "$PYTHON" -m pip install --user --disable-pip-version-check \
            -r "${PROJECT_ROOT}/requirements.txt" >/dev/null 2>&1 || \
            echo "WARNING: failed to install all requirements; continuing anyway"
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
import os
import sys
import torch
import gymnasium
import minatar

print("Environment check passed.")
print("Python:     ", sys.version.split()[0])
print("Executable: ", sys.executable)
print("Torch:      ", torch.__version__)
print("Device:      CPU (GPU disabled)")
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
EOF
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

rct_resubmit_if_incomplete() {
    local SLURM_SCRIPT="$1"

    if [[ "${RCT_AUTO_RESUBMIT:-0}" != "1" ]]; then
        echo ""
        echo "Auto-resubmit disabled (RCT_AUTO_RESUBMIT=${RCT_AUTO_RESUBMIT:-0})."
        return 0
    fi

    if rct_task_incomplete; then
        echo ""
        echo "Work remains for task ${SLURM_ARRAY_TASK_ID} — resubmitting..."
        sbatch \
            --dependency=afterany:"${SLURM_JOB_ID}" \
            --array="${SLURM_ARRAY_TASK_ID}" \
            --export=ALL \
            "${PROJECT_ROOT}/${SLURM_SCRIPT}"
    else
        echo ""
        echo "All individuals complete for task ${SLURM_ARRAY_TASK_ID}."
    fi
}
