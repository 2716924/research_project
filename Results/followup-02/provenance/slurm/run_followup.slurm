#!/bin/bash
# Follow-up runs listed by followup.py; see FOLLOWUP_README.md.
# Each array task trains RCT_RUNS_PER_TASK manifest rows side by side on one node.
# Before the first sbatch: mkdir -p logs/rct-followup and run followup.py prepare.
# Submit one group at a time with the ranges printed by: python followup.py tasks
# Override the partition with sbatch -p.

#SBATCH --job-name=rct-followup
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/rct-followup/%x_%A_%a.out
#SBATCH --error=logs/rct-followup/%x_%A_%a.err
#SBATCH --requeue

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

export RCT_EXPERIMENT_ID="${RCT_EXPERIMENT_ID:-followup-01}"
if [[ ! "$RCT_EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: experiment ID must use only letters, numbers, dots, underscores or hyphens" >&2
    exit 1
fi
export RCT_EXPERIMENT_ROOT="${RCT_EXPERIMENT_ROOT:-${PROJECT_ROOT}/experiments/${RCT_EXPERIMENT_ID}}"
export RCT_RUNS_PER_TASK="${RCT_RUNS_PER_TASK:-4}"
export RCT_AUTO_RESUBMIT="${RCT_AUTO_RESUBMIT:-1}"
export RCT_RESUBMIT_MAX="${RCT_RESUBMIT_MAX:-3}"
export RCT_RESUBMIT_COUNT="${RCT_RESUBMIT_COUNT:-0}"
export PYTHONDONTWRITEBYTECODE=1
LOG_DIR="logs/rct-followup/rows"
ROWS_STARTED=0
CANCELLED=0
PYTHON="${RCT_PYTHON:-$HOME/.conda/envs/mltest/bin/python}"

# Auto-resubmit (as in the main RCT script): while this task has unfinished rows, queue
# the same array index after this job ends, with the same partition and time limit.
# Completed rows are skipped there. This also covers a task that dies during start-up,
# and RCT_RESUBMIT_MAX caps the chain so a repeating problem cannot loop forever.
resubmit_if_unfinished() {
    if [[ "$RCT_AUTO_RESUBMIT" != "1" ]]; then
        echo "Auto-resubmit disabled (RCT_AUTO_RESUBMIT=${RCT_AUTO_RESUBMIT})."
        return 0
    fi
    if (( ROWS_STARTED == 1 )) && ! "$PYTHON" followup.py pending "${SLURM_ARRAY_TASK_ID}" \
            --root "$RCT_EXPERIMENT_ROOT" --per-task "$RCT_RUNS_PER_TASK"; then
        echo "All rows of task ${SLURM_ARRAY_TASK_ID} are complete."
        return 0
    fi
    if (( RCT_RESUBMIT_COUNT >= RCT_RESUBMIT_MAX )); then
        echo "Not resubmitting: already resubmitted ${RCT_RESUBMIT_COUNT} times (RCT_RESUBMIT_MAX)." \
             "Check the logs, then resubmit task ${SLURM_ARRAY_TASK_ID} by hand."
        return 0
    fi
    local args=(--dependency=afterany:"${SLURM_JOB_ID}" --array="${SLURM_ARRAY_TASK_ID}" --export=ALL)
    if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then args+=(--partition="${SLURM_JOB_PARTITION}"); fi
    local time_limit
    time_limit=$(squeue -h -j "${SLURM_JOB_ID}" -o %l 2>/dev/null || true)
    if [[ -n "$time_limit" ]]; then args+=(--time="$time_limit"); fi
    echo "Resubmitting task ${SLURM_ARRAY_TASK_ID} (resubmission $(( RCT_RESUBMIT_COUNT + 1 )) of at most ${RCT_RESUBMIT_MAX})..."
    local attempt
    for attempt in 1 2 3; do
        if RCT_RESUBMIT_COUNT=$(( RCT_RESUBMIT_COUNT + 1 )) sbatch "${args[@]}" \
                "${PROJECT_ROOT}/slurm/run_followup.slurm"; then
            return 0
        fi
        sleep 10
    done
    echo "ERROR: sbatch failed three times; resubmit task ${SLURM_ARRAY_TASK_ID} by hand." >&2
}

on_exit() {
    local rc=$?
    trap - EXIT
    set +e
    if (( CANCELLED == 1 )); then
        echo "Task ${SLURM_ARRAY_TASK_ID} was cancelled or killed by a signal; not resubmitting."
        exit "$rc"
    fi
    if (( ROWS_STARTED == 0 )); then
        echo "Task ${SLURM_ARRAY_TASK_ID} stopped during start-up (exit ${rc}) before any run began."
    fi
    resubmit_if_unfinished
    exit "$rc"
}
# scancel (or SLURM killing the job) sends SIGTERM: never resubmit a cancelled task.
on_signal() {
    CANCELLED=1
    exit 143
}
trap on_exit EXIT
trap on_signal TERM INT

# Folders are created up front by `followup.py prepare`; here they are only checked,
# with retries, because simultaneous creation on shared storage can fail
# ("mkdir: Already exists").
ensure_dir() {
    local attempt
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if [[ -d "$1" ]]; then return 0; fi
        mkdir -p "$1" 2>/dev/null || true
        sleep $(( (RANDOM % 5) + 1 ))
    done
    if [[ -d "$1" ]]; then return 0; fi
    echo "ERROR: cannot create $1" >&2
    return 1
}

# Spread out tasks that start at the same moment.
sleep $(( RANDOM % ${RCT_START_JITTER:-30} + 1 ))

rct_setup_cpu
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-2716924-wits}"
export WANDB_PROJECT="${WANDB_PROJECT:-dqn-rct}"
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
if [[ "$WANDB_MODE" == offline || "$WANDB_MODE" == disabled ]]; then
    unset WANDB_API_KEY
fi

# Share the node's cores between the runs of this task.
CORES=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
export RCT_TORCH_THREADS=$(( CORES / RCT_RUNS_PER_TASK ))
(( RCT_TORCH_THREADS >= 1 )) || export RCT_TORCH_THREADS=1

rct_export_time_budget
rct_check_python

ensure_dir "$LOG_DIR"
ensure_dir "$RCT_EXPERIMENT_ROOT"
ROWS=$("$PYTHON" followup.py rows "${SLURM_ARRAY_TASK_ID}" --per-task "$RCT_RUNS_PER_TASK")

echo ""
echo "############################################################"
echo "#  RCT FOLLOW-UP — array task ${SLURM_ARRAY_TASK_ID}"
echo "#  Started:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "#  Job ID:      ${SLURM_JOB_ID:-local}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-n/a}"
echo "#  Experiment:  ${RCT_EXPERIMENT_ROOT}"
echo "#  Rows:        ${ROWS}  (${RCT_TORCH_THREADS} torch threads each, ${CORES} cores)"
echo "#  Resubmits:   ${RCT_RESUBMIT_COUNT} so far (max ${RCT_RESUBMIT_MAX})"
echo "############################################################"

ROW_IDS=()
PIDS=()
ROWS_STARTED=1
for row in $ROWS; do
    "$PYTHON" -u followup.py run "$row" --root "$RCT_EXPERIMENT_ROOT" \
        > "${LOG_DIR}/row${row}_${SLURM_JOB_ID:-local}.log" 2>&1 &
    ROW_IDS+=("$row")
    PIDS+=($!)
    sleep 2
done

FINAL=0
for i in "${!PIDS[@]}"; do
    row="${ROW_IDS[$i]}"
    set +e
    wait "${PIDS[$i]}"
    rc=$?
    set -e
    echo "Row ${row}: exit ${rc} (log ${LOG_DIR}/row${row}_${SLURM_JOB_ID:-local}.log)"
    # 2 = stopped for the time budget; the resubmitted task restarts that run.
    if (( rc != 0 )); then
        if (( rc == 2 && FINAL == 0 )); then FINAL=2; elif (( rc != 2 )); then FINAL=1; fi
    fi
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date) with exit ${FINAL}"
exit "$FINAL"
