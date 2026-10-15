"""SLURM helpers for time-limit safety and resume checks."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import Optional


def _parse_slurm_datetime(value: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _end_time_from_scontrol(job_id: str) -> Optional[datetime]:
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    for token in result.stdout.split():
        if token.startswith("EndTime="):
            end_time = token.split("=", 1)[1]
            if end_time in {"Unknown", "N/A"}:
                return None
            return _parse_slurm_datetime(end_time)
    return None


def slurm_time_remaining_seconds() -> Optional[float]:
    """
    Seconds until SLURM ends this job, or None when not under SLURM.

    Uses SLURM_JOB_END_TIME, RCT_JOB_END_EPOCH (set in slurm/common.sh),
    or falls back to `scontrol show job`.
    """
    end_epoch = os.environ.get("RCT_JOB_END_EPOCH")
    if end_epoch:
        try:
            return float(end_epoch) - datetime.now().timestamp()
        except ValueError:
            pass

    end_time = os.environ.get("SLURM_JOB_END_TIME")
    if end_time and end_time not in {"Unknown", "N/A"}:
        if end_time.isdigit():
            return float(end_time) - datetime.now().timestamp()
        end_dt = _parse_slurm_datetime(end_time)
        if end_dt is not None:
            return (end_dt - datetime.now(tz=end_dt.tzinfo)).total_seconds()

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        end_dt = _end_time_from_scontrol(job_id)
        if end_dt is not None:
            return (end_dt - datetime.now(tz=end_dt.tzinfo)).total_seconds()

    return None


def should_stop_for_time_limit(buffer_seconds=None) -> bool:
    """
    Return True when the job should exit cleanly before SLURM kills it.

    Default buffer is 15 minutes so results can be flushed and the job
    can be resubmitted with resume support.
    """
    if buffer_seconds is None:
        buffer_seconds = int(os.environ.get("RCT_TIME_LIMIT_BUFFER", "900"))

    remaining = slurm_time_remaining_seconds()
    if remaining is None:
        return False
    return remaining <= buffer_seconds


def count_completed_individuals(output_dir, variant, game_id) -> tuple[int, int | None]:
    """Return (completed_count, total_expected_or_none)."""
    from rct.results import load_existing_results, result_path

    expected = os.environ.get("RCT_NUM_INDIVIDUALS")
    total = int(expected) if expected else None
    out_path = result_path(output_dir, variant, game_id)
    _, completed_ids = load_existing_results(out_path)
    return len(completed_ids), total


def task_is_incomplete(output_dir, variant, game_id) -> bool:
    completed, total = count_completed_individuals(output_dir, variant, game_id)
    if total is None:
        return False
    return completed < total
