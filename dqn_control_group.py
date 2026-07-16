#!/usr/bin/env python3
"""
Backward-compatible wrapper for the control-group trainer.

New work should use `run_rct.py --variant control` instead.
This file remains so older SLURM scripts and notes still work.
"""

import argparse

from rct.constants import GAMES
from rct.trial import execute_rct_trial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control-group (canonical Nature-DQN) RCT trainer")
    parser.add_argument("--num-individuals", type=int, default=30)
    parser.add_argument("--total-frames", type=int, default=5_000_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--game", type=str, default=None, choices=GAMES)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    games = [args.game] if args.game is not None else None

    if args.pilot:
        execute_rct_trial(
            variant="control",
            num_individuals=3,
            total_frames=15_000,
            eval_frequency=3_000,
            num_eval_episodes=5,
            output_dir=args.output_dir,
            games=games,
        )
    else:
        execute_rct_trial(
            variant="control",
            num_individuals=args.num_individuals,
            total_frames=args.total_frames,
            eval_frequency=args.eval_frequency,
            num_eval_episodes=args.num_eval_episodes,
            output_dir=args.output_dir,
            games=games,
        )
