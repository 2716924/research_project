"""Optional W&B tracking with a local, line-buffered record of every attempt.

Training/result JSON remains authoritative. Offline runs need no credentials.
Each retry gets a new ID because an unfinished individual restarts at frame zero.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch


@contextmanager
def preserve_rng():
    """Prevent logging code from consuming the CPU experiment's RNG streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@dataclass(frozen=True)
class TrackingSettings:
    mode: str = "disabled"
    project: str = "dqn-rct"
    entity: str = "2716924-wits"
    directory: str = "./tracking"
    experiment_id: str = "local"

    def validate(self):
        if self.mode not in ("disabled", "offline", "online"):
            raise ValueError("W&B mode must be disabled, offline or online")
        if self.mode != "disabled":
            # Fail before spending compute if configured tracking is unavailable.
            with preserve_rng():
                import wandb  # noqa: F401


class AttemptTracker:
    def __init__(self, settings, metadata, individual, pilot=False):
        self.settings = settings
        self.attempt_id = uuid.uuid4().hex
        self.run = None
        self.stream = None
        self.finished = False
        if settings.mode == "disabled":
            return
        directory = Path(settings.directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / f"attempt-{self.attempt_id}.jsonl").open("x", buffering=1)
        variant, game = metadata["variant"], metadata["game_id"]
        config = dict(metadata, **individual, experiment_id=settings.experiment_id,
                      pilot=pilot, attempt_id=self.attempt_id,
                      pair_id=f"{game}:{individual['individual_id']}:{individual['seed']}",
                      wandb_version=version("wandb"),
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"))
        self._write({"event": "start", "config": config})
        try:
            with preserve_rng():
                import wandb
                self.run = wandb.init(
                    mode=settings.mode, project=settings.project, entity=settings.entity,
                    dir=str(directory), id=self.attempt_id, config=config,
                    name=f"{variant}-{game.split('/')[1]}-i{individual['individual_id']:02d}-{self.attempt_id[:8]}",
                    group=f"{settings.experiment_id}/{game}", job_type=variant,
                    tags=["pilot" if pilot else "full", variant, game],
                    settings=wandb.Settings(console="off", save_code=False, disable_git=True,
                                            init_timeout=60),
                )
                self.run.define_metric("frames")
                self.run.define_metric("train/*", step_metric="frames")
                self.run.define_metric("eval/*", step_metric="frames")
                self.run.summary.update({"status": "running", "completed": False})
        except BaseException:
            self._write({"event": "finish", "status": "initialization_failed", "completed": False})
            self.stream.close()
            if self.run is not None:
                with preserve_rng():
                    self.run.finish(exit_code=1)
            raise

    def _write(self, record):
        if self.stream is not None:
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")

    def log(self, frames, metrics):
        if self.run is None:
            return
        values = dict(metrics, frames=frames)
        self._write({"event": "metrics", **values})
        with preserve_rng():
            self.run.log(values)

    def finish(self, status, summary=None):
        if self.finished:
            return
        self.finished = True
        values = dict(summary or {}, status=status, completed=(status == "completed"))
        finalized = False
        try:
            if self.run is not None:
                with preserve_rng():
                    self.run.summary.update(values)
                    self.run.finish(exit_code=0 if status == "completed" else 1)
            finalized = True
        finally:
            if self.stream is not None:
                try:
                    self._write({"event": "finish", **values, "wandb_finalized": finalized})
                finally:
                    self.stream.close()
