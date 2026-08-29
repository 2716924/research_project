#!/usr/bin/env python3
"""
Statistical analysis for the DQN RCT (Phase 4 of the research proposal).

Reads saved JSON results, computes paired AUC differences, runs Shapiro-Wilk
normality checks, paired t-tests, and Bonferroni correction.

Usage:
    python analyze_rct.py --results-dir results/control/breakout
    python analyze_rct.py --results-root results --game-slug breakout
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

from rct.constants import GAMES, VARIANTS, VARIANT_LABELS

BONFERRONI_ALPHA = 0.05 / 4  # four treatment arms vs one control


def load_auc_by_individual(result_file):
    with open(result_file) as f:
        data = json.load(f)
    return {row["individual_id"]: row["auc"] for row in data["results"]}


def find_result_file(results_root, variant, game_slug):
    game_id = None
    for g in GAMES:
        if g.endswith(game_slug.title()) or game_slug.lower() in g.lower():
            game_id = g
            break
    if game_id is None:
        game_id = f"MinAtar/{game_slug.capitalize()}-v1"

    safe_game = game_id.replace("/", "_")
    candidates = [
        results_root / variant / game_slug / f"{variant}_results_{safe_game}.json",
        results_root / variant / f"{variant}_results_{safe_game}.json",
        results_root / f"{variant}_results_{safe_game}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def paired_analysis(control_aucs, treatment_aucs):
    ids = sorted(set(control_aucs) & set(treatment_aucs))
    control = np.array([control_aucs[i] for i in ids])
    treatment = np.array([treatment_aucs[i] for i in ids])
    differences = treatment - control

    shapiro_stat, shapiro_p = stats.shapiro(differences)
    t_stat, t_p = stats.ttest_rel(treatment, control)

    return {
        "n_pairs": len(ids),
        "mean_control_auc": float(control.mean()),
        "mean_treatment_auc": float(treatment.mean()),
        "mean_difference": float(differences.mean()),
        "shapiro_statistic": float(shapiro_stat),
        "shapiro_p_value": float(shapiro_p),
        "t_statistic": float(t_stat),
        "p_value": float(t_p),
        "significant_at_bonferroni": bool(t_p <= BONFERRONI_ALPHA),
        "bonferroni_alpha": BONFERRONI_ALPHA,
    }


def main():
    parser = argparse.ArgumentParser(description="RCT statistical analysis")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--game-slug", type=str, default="seaquest",
                        help="Game folder name: seaquest or spaceinvaders")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON summary to this path")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    control_file = find_result_file(results_root, "control", args.game_slug)
    if control_file is None:
        raise SystemExit(f"Control results not found under {results_root}")

    control_aucs = load_auc_by_individual(control_file)
    summary = {
        "game": args.game_slug,
        "control_file": str(control_file),
        "comparisons": {},
    }

    print(f"Game: {args.game_slug}")
    print(f"Control individuals: {len(control_aucs)}")
    print(f"Bonferroni alpha: {BONFERRONI_ALPHA:.4f}")
    print()

    for variant in VARIANTS:
        if variant == "control":
            continue

        treatment_file = find_result_file(results_root, variant, args.game_slug)
        if treatment_file is None:
            print(f"[skip] {VARIANT_LABELS[variant]} — no results file")
            continue

        treatment_aucs = load_auc_by_individual(treatment_file)
        result = paired_analysis(control_aucs, treatment_aucs)
        summary["comparisons"][variant] = result

        sig = "YES" if result["significant_at_bonferroni"] else "no"
        print(f"{VARIANT_LABELS[variant]}")
        print(f"  pairs:           {result['n_pairs']}")
        print(f"  mean AUC diff:   {result['mean_difference']:+.2f}")
        print(f"  Shapiro p:       {result['shapiro_p_value']:.4f}")
        print(f"  paired t p:      {result['p_value']:.4f}")
        print(f"  significant:     {sig}")
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
