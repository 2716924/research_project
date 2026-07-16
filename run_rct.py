#!/usr/bin/env python3
"""
Run one arm of the DQN Randomised Control Trial (RCT).

This is the main entry point for cluster jobs. Each variant (control
or treatment) uses the same hyperparameter individuals and games, so
results can be compared fairly later.

Examples:
    # Quick smoke test (3 individuals, short training)
    python run_rct.py --variant control --pilot

    # Full control-group run on one game
    python run_rct.py --variant control --game MinAtar/Breakout-v1

    # Full treatment arm
    python run_rct.py --variant ddqn --output-dir results/ddqn
"""

import argparse

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
    args = parser.parse_args()

    games = [args.game] if args.game is not None else None

    if args.pilot:
        print(f"Pilot mode — {VARIANT_LABELS[args.variant]}")
        execute_rct_trial(
            variant=args.variant,
            num_individuals=3,
            total_frames=15_000,
            eval_frequency=3_000,
            num_eval_episodes=5,
            output_dir=args.output_dir,
            games=games,
        )
    else:
        execute_rct_trial(
            variant=args.variant,
            num_individuals=args.num_individuals,
            total_frames=args.total_frames,
            eval_frequency=args.eval_frequency,
            num_eval_episodes=args.num_eval_episodes,
            output_dir=args.output_dir,
            games=games,
        )


if __name__ == "__main__":
    main()
