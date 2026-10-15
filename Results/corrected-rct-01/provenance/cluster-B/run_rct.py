#!/usr/bin/env python3
"""
Run one arm of the DQN Randomised Control Trial (RCT).

This is the main entry point for cluster jobs. Each variant (control
or treatment) uses the same hyperparameter individuals and games, so
results can be compared fairly later.

Examples:
    # Quick CPU smoke test
    python run_rct.py --variant control --pilot

    # Full treatment arm on one game
    python run_rct.py --variant ddqn --game MinAtar/Breakout-v1
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

from rct.constants import GAMES, VARIANTS, VARIANT_LABELS
from rct.trial import execute_rct_trial
from rct.tracking import TrackingSettings


def main():
    parser = argparse.ArgumentParser(
        description="Run one RCT algorithmic arm (control or treatment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=VARIANTS,
        help="Which algorithmic arm to train",
    )
    parser.add_argument("--num-individuals", type=int, default=30)
    parser.add_argument("--total-frames", type=int, default=5_000_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--game", type=str, default=None, choices=GAMES)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Short smoke-test run instead of full proposal scale",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log training loss every N frames",
    )
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"],
                        default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "dqn-rct"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "2716924-wits"))
    parser.add_argument("--tracking-dir", default=None,
                        help="W&B data and attempt JSONL directory; defaults beside results")
    parser.add_argument("--experiment-id", default=os.environ.get("RCT_EXPERIMENT_ID", "local"))
    args = parser.parse_args()
    tracking = TrackingSettings(
        mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
        directory=args.tracking_dir or str(Path(args.output_dir).resolve().parent / "tracking"),
        experiment_id=args.experiment_id,
    )
    tracking.validate()

    # A child process keeps the checks' RNG/thread settings out of the actual
    # experiment. A failed algorithm/integrity check prevents cluster training.
    print("Running algorithm and result-integrity preflight checks...", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).parent / "verify_arms.py")], check=True)

    games = [args.game] if args.game is not None else None

    if args.pilot:
        print(f"Pilot mode — {VARIANT_LABELS[args.variant]}")
        stopped_early = execute_rct_trial(
            variant=args.variant,
            num_individuals=3,
            total_frames=15_000,
            eval_frequency=3_000,
            num_eval_episodes=5,
            output_dir=args.output_dir,
            games=games,
            pilot=True,
            log_every=args.log_every,
            tracking=tracking,
        )
    else:
        stopped_early = execute_rct_trial(
            variant=args.variant,
            num_individuals=args.num_individuals,
            total_frames=args.total_frames,
            eval_frequency=args.eval_frequency,
            num_eval_episodes=args.num_eval_episodes,
            output_dir=args.output_dir,
            games=games,
            pilot=False,
            log_every=args.log_every,
            tracking=tracking,
        )

    if stopped_early:
        print("Stopped early for SLURM time-limit safety — resubmit to continue.")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
