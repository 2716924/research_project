#!/usr/bin/env python3
"""
Run one arm of the DQN Randomised Control Trial (RCT).

This is the main entry point for cluster jobs. Each variant (control
or treatment) uses the same hyperparameter individuals and games, so
results can be compared fairly later.

Examples:
    # Quick GPU smoke test
    python run_rct.py --variant control --pilot

    # Full treatment arm on one game
    python run_rct.py --variant ddqn --game MinAtar/Breakout-v1

    # Disable wandb locally
    WANDB_DISABLED=true python run_rct.py --variant control --pilot
"""

import argparse
import sys

from rct.constants import GAMES, VARIANTS, VARIANT_LABELS
from rct.trial import execute_rct_trial


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
        "--wandb",
        dest="use_wandb",
        action="store_true",
        default=True,
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="wandb entity (defaults to WANDB_ENTITY env or project default)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="wandb project (defaults to WANDB_PROJECT env or project default)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log training loss to wandb every N frames",
    )
    args = parser.parse_args()

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
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            log_every=args.log_every,
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
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            log_every=args.log_every,
        )

    if stopped_early:
        print("Stopped early for SLURM time-limit safety — resubmit to continue.")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
