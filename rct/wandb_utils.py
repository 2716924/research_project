"""Weights & Biases helpers for RCT experiment logging."""

import os
import sys
from typing import Any, Optional


WANDB_ENTITY = "juniormatimu572-wits"
WANDB_PROJECT = "DQN modifications"


def wandb_enabled() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False
    return True


def init_run(
    *,
    variant: str,
    game_id: str,
    individual_id: int,
    seed: int,
    config: dict[str, Any],
    pilot: bool = False,
    entity: Optional[str] = None,
    project: Optional[str] = None,
):
    """
    Start a wandb run for one RCT individual.

    Requires WANDB_API_KEY in the environment (never commit the key to git).
    """
    if not wandb_enabled():
        return None

    try:
        import wandb
    except ModuleNotFoundError:
        print("WARNING: wandb is not installed; skipping W&B logging.", file=sys.stderr)
        return None

    game_slug = game_id.split("/")[-1].replace("-v1", "").lower()
    run_name = f"{variant}-{game_slug}-ind{individual_id:02d}"
    if pilot:
        run_name = f"pilot-{run_name}"

    tags = [variant, game_slug]
    if pilot:
        tags.append("pilot")

    group = f"{variant}-{game_slug}"
    if pilot:
        group = f"pilot-{group}"

    return wandb.init(
        entity=entity or os.environ.get("WANDB_ENTITY", WANDB_ENTITY),
        project=project or os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
        name=run_name,
        group=group,
        job_type="pilot" if pilot else "train",
        tags=tags,
        config={
            "variant": variant,
            "game_id": game_id,
            "individual_id": individual_id,
            "seed": seed,
            "pilot": pilot,
            **config,
        },
        reinit=True,
    )


def log_training_metrics(run, *, frame, loss, epsilon, variant):
    if run is None:
        return
    run.log(
        {
            "train/loss": float(loss),
            "train/epsilon": float(epsilon),
            "train/frame": frame,
            "variant": variant,
        },
        step=frame,
    )


def log_eval_metrics(run, *, frame, mean_return, variant):
    if run is None:
        return
    run.log(
        {
            "eval/mean_return": float(mean_return),
            "eval/frame": frame,
            "variant": variant,
        },
        step=frame,
    )


def finish_run(run, *, exit_code=0):
    if run is None:
        return
    run.finish(exit_code=exit_code)


def log_individual_summary(run, *, auc, num_eval_points):
    if run is None:
        return
    run.log(
        {
            "summary/auc": float(auc),
            "summary/num_eval_points": int(num_eval_points),
        }
    )
    finish_run(run, exit_code=0)
