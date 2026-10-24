#!/bin/bash
# =============================================================================
# Shared setup helpers for Wits HPC SLURM jobs (CPU-only, bigbatch partition)
# =============================================================================

# Timestamp used consistently across all log functions
_RCT_START_EPOCH=$(date +%s)

rct_log_header() {
    local variant="${RCT_VARIANT:-unknown}"
    local game="${RCT_GAME_SLUG:-unknown}"
    echo ""
    echo "############################################################"
    echo "#  RCT JOB — ${variant} / ${game}"
    echo "############################################################"
    echo "#  Started:       $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "#  User:          ${USER:-unknown}"
    echo "#  Job ID:        ${SLURM_JOB_ID:-local}"
    echo "#  Job name:      ${SLURM_JOB_NAME:-local}"
    echo "#  Array task:    ${SLURM_ARRAY_TASK_ID:-0}  (variant-major, five games per variant)"
    echo "#  Experiment:    ${RCT_EXPERIMENT_ID:-local}"
    echo "#  W&B:           ${WANDB_MODE:-disabled} / ${WANDB_ENTITY:-}/${WANDB_PROJECT:-}"
    echo "#  Node:          $(hostname)"
    echo "#  CPUs/task:     ${SLURM_CPUS_PER_TASK:-1}"
    echo "#  Mem limit:     ${SLURM_MEM_PER_NODE:-n/a} MB"
    echo "#  Wall limit:    ${SLURM_JOB_END_TIME:-n/a}"
    echo "#  Working dir:   ${PWD}"
    echo "############################################################"
    echo ""
}

rct_log_footer() {
    local exit_code="${1:-0}"
    local end_epoch
    end_epoch=$(date +%s)
    local elapsed=$(( end_epoch - _RCT_START_EPOCH ))
    local hh=$(( elapsed / 3600 ))
    local mm=$(( (elapsed % 3600) / 60 ))
    local ss=$(( elapsed % 60 ))
    local status_label
    if   [[ "$exit_code" -eq 0 ]]; then status_label="SUCCESS"
    elif [[ "$exit_code" -eq 2 ]]; then status_label="PARTIAL (time budget)"
    else                                  status_label="FAILED"
    fi
    echo ""
    echo "############################################################"
    echo "#  JOB COMPLETE — ${RCT_VARIANT:-unknown} / ${RCT_GAME_SLUG:-unknown}"
    echo "#  Status:        ${status_label}  (exit ${exit_code})"
    echo "#  Finished:      $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "#  Wall elapsed:  ${hh}h ${mm}m ${ss}s"
    echo "############################################################"
    echo ""
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
    if [[ -r /etc/profile.d/lmod.sh ]]; then source /etc/profile.d/lmod.sh; fi
    module purge 2>/dev/null || true
    module load anaconda 2>/dev/null || true

    export RCT_DEVICE=cpu
    export CUDA_VISIBLE_DEVICES=""
    export TORCH_CUDA_ARCH_LIST=""
    export TORCH_CPP_LOG_LEVEL=ERROR

    PYTHON="${RCT_PYTHON:-$HOME/.conda/envs/mltest/bin/python}"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: mltest environment not found at $PYTHON"
        echo "Create it with: conda create -n mltest python=3.11 -y"
        exit 1
    fi
}

rct_export_time_budget() {
    export RCT_TIME_LIMIT_BUFFER="${RCT_TIME_LIMIT_BUFFER:-900}"
    # --export=ALL carries the previous allocation's derived deadline on retry.
    unset RCT_JOB_END_EPOCH

    local end_time="${SLURM_JOB_END_TIME:-}"
    if [[ -z "$end_time" || "$end_time" == "Unknown" || "$end_time" == "N/A" ]]; then
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            end_time=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i ~ /^EndTime=/) {sub(/^EndTime=/,"",$i); print $i}}' || true)
        fi
    fi

    if [[ -n "$end_time" && "$end_time" != "Unknown" && "$end_time" != "N/A" ]]; then
        if [[ "$end_time" =~ ^[0-9]+$ ]]; then
            export RCT_JOB_END_EPOCH="$end_time"
        else
            export RCT_JOB_END_EPOCH=$(date -d "${end_time}" +%s 2>/dev/null || true)
        fi
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
import scipy

if os.environ.get("WANDB_MODE", "disabled") != "disabled":
    import wandb
    print("W&B:        ", wandb.__version__)

print("Environment check passed.")
print("Python:     ", sys.version.split()[0])
print("Executable: ", sys.executable)
print("Torch:      ", torch.__version__)
print("Device:      CPU (GPU disabled)")
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
EOF
}

# Map all 25 cells using the same constants as the training CLI.
rct_map_task() {
    local assignment
    assignment=$("$PYTHON" -m rct.tasks "${SLURM_ARRAY_TASK_ID:-}") || return 1
    read -r RCT_VARIANT RCT_GAME RCT_GAME_SLUG <<< "$assignment"
    export RCT_VARIANT RCT_GAME RCT_GAME_SLUG
}

rct_setup_experiment() {
    local phase="$1"
    export RCT_EXPERIMENT_ID="${RCT_EXPERIMENT_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
    if [[ ! "$RCT_EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "ERROR: experiment ID must use only letters, numbers, dots, underscores or hyphens" >&2
        return 1
    fi
    # Absolute shared storage survives allocation changes. Phase separates pilots.
    export RCT_EXPERIMENT_ROOT="${RCT_EXPERIMENT_ROOT:-${PROJECT_ROOT}/experiments/${RCT_EXPERIMENT_ID}}"
    if [[ "$RCT_EXPERIMENT_ROOT" != /* ]]; then
        echo "ERROR: RCT_EXPERIMENT_ROOT must be an absolute shared-filesystem path" >&2
        return 1
    fi
    export OUTPUT_DIR="${RCT_EXPERIMENT_ROOT}/${phase}/results/${RCT_VARIANT}/${RCT_GAME_SLUG}"
    export RCT_TRACKING_DIR="${RCT_EXPERIMENT_ROOT}/${phase}/tracking/${RCT_VARIANT}/${RCT_GAME_SLUG}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_ENTITY="${WANDB_ENTITY:-2716924-wits}"
    export WANDB_PROJECT="${WANDB_PROJECT:-dqn-rct}"
    export WANDB_CONSOLE=off
    export WANDB_DISABLE_CODE=true
    # Offline jobs never need a key. Do not propagate one to the SDK.
    if [[ "$WANDB_MODE" == offline || "$WANDB_MODE" == disabled ]]; then
        unset WANDB_API_KEY
    fi
    mkdir -p "$OUTPUT_DIR" "$RCT_TRACKING_DIR"
}

rct_lock_task() {
    # Keep the descriptor open through training to prevent simultaneous writers
    # if the same experiment/task is accidentally submitted twice.
    exec 9>"${OUTPUT_DIR}/.task.lock"
    if ! flock -n 9; then
        echo "ERROR: this experiment's variant/game is already running" >&2
        return 1
    fi
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
