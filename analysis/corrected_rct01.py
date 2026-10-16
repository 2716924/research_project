"""Endpoint analysis for the corrected-rct-01 cohort.

Builds the analysis dataset, then runs the proposal's paired comparisons against
Standard DQN (two-sided paired t-tests, Bonferroni alpha = 0.05/4 per game).

Dataset assembly (see Results/corrected-rct-01/README.md):
  * ddqn, per, nstep, averaged: Results/corrected-rct-01/results
  * control: Results/original-run-2026-09/results. The corrected cohort reran
    ddqn, nstep and averaged and reproduced every trajectory bit-for-bit; the
    only learning-relevant code change between runs is the PER priority fix,
    so the original control runs are the corrected cohort's control runs.
  * nstep/freeway individuals 18-29: original run, for the same reason
    (individuals 0-17 overlap and are verified identical).

    .venv/bin/python -m analysis.corrected_rct01
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CORRECTED = ROOT / "Results/corrected-rct-01/results"
ORIGINAL = ROOT / "Results/original-run-2026-09/results"
OUT = ROOT / "analysis_output/corrected-rct-01"

VARIANTS = ["control", "ddqn", "per", "nstep", "averaged"]
LABELS = dict(zip(VARIANTS, ["DQN", "Double DQN", "PER", "n-step DQN", "Averaged DQN"]))
COLORS = dict(zip(VARIANTS, ["#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
STYLES = dict(zip(VARIANTS, ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]))
MARKERS = dict(zip(VARIANTS, ["o", "s", "^", "D", "v"]))
GAMES = ["breakout", "asterix", "freeway", "spaceinvaders"]
NAMES = dict(zip(GAMES, ["Breakout", "Asterix", "Freeway", "Space Invaders"]))
ALPHA = 0.05 / 4
METRICS = {"auc": "AUC", "final_return": "Final return", "last_10_return": "Mean of last 10 evaluations"}


def read(root, arm, game):
    path = next((root / arm / game).glob("*_results_*.json"))
    return {r["individual_id"]: r for r in json.loads(path.read_text())["results"]}


def load():
    data, source = {}, {}
    for game in GAMES:
        data["control", game] = read(ORIGINAL, "control", game)
        source["control", game] = "original run"
        for arm in VARIANTS[1:]:
            runs = read(CORRECTED, arm, game)
            source[arm, game] = "corrected"
            if len(runs) < 30:
                original = read(ORIGINAL, arm, game)
                for i, r in runs.items():
                    assert r["eval_trajectory"] == original[i]["eval_trajectory"], (arm, game, i)
                added = sorted(set(original) - set(runs))
                runs.update({i: original[i] for i in added})
                source[arm, game] = f"corrected 0-{min(added) - 1}, original {min(added)}-{max(added)}"
            data[arm, game] = runs
        ids = data["control", game].keys()
        for arm in VARIANTS:
            runs = data[arm, game]
            assert runs.keys() == ids and len(ids) == 30, (arm, game)
            for i in ids:
                c = data["control", game][i]
                assert (runs[i]["seed"], runs[i]["config"]) == (c["seed"], c["config"]), (arm, game, i)
    return data, source


def endpoint(run, metric):
    t = run["eval_trajectory"]
    return {"auc": run["auc"], "final_return": t[-1], "last_10_return": float(np.mean(t[-10:]))}[metric]


def values(data, arm, game, metric):
    runs = data[arm, game]
    return np.array([endpoint(runs[i], metric) for i in sorted(runs)])


def paired(c, t):
    d = t - c
    n, sd = len(d), d.std(ddof=1)
    radius = stats.t.ppf(1 - ALPHA / 2, n - 1) * sd / np.sqrt(n)
    test = stats.ttest_rel(t, c)
    return dict(n=n, mean_control=c.mean(), mean_treatment=t.mean(), mean_diff=d.mean(),
                pct_change=100 * d.mean() / c.mean(), ci98_75_low=d.mean() - radius,
                ci98_75_high=d.mean() + radius, dz=d.mean() / sd, t=test.statistic, p=test.pvalue,
                significant=bool(test.pvalue < ALPHA), wilcoxon_p=stats.wilcoxon(t, c).pvalue,
                shapiro_p=stats.shapiro(d).pvalue, win_fraction=(d > 0).mean())


def tables(data, source):
    desc = []
    for game in GAMES:
        for arm in VARIANTS:
            auc, fin, last = (values(data, arm, game, m) for m in METRICS)
            desc.append(dict(game=game, arm=arm, source=source[arm, game], n=len(auc),
                             auc_mean=auc.mean(), auc_sd=auc.std(ddof=1), auc_median=np.median(auc),
                             final_mean=fin.mean(), final_sd=fin.std(ddof=1), last10_mean=last.mean(),
                             last10_median=np.median(last), last10_best=last.max()))
    pd.DataFrame(desc).to_csv(OUT / "tables/descriptives.csv", index=False)

    tests = {}
    for metric in METRICS:
        rows = []
        for game in GAMES:
            c = values(data, "control", game, metric)
            for arm in VARIANTS[1:]:
                rows.append(dict(game=game, arm=arm, **paired(c, values(data, arm, game, metric))))
        tests[metric] = pd.DataFrame(rows)
        tests[metric].to_csv(OUT / f"tables/paired_{metric}.csv", index=False)

    fix = []
    for game in GAMES:
        old = np.array([r["auc"] for _, r in sorted(read(ORIGINAL, "per", game).items())])
        new = values(data, "per", game, "auc")
        fix.append(dict(game=game, per_auc_before_fix=old.mean(), per_auc_after_fix=new.mean(),
                        pct_change=100 * (new.mean() - old.mean()) / old.mean(),
                        paired_p=stats.ttest_rel(new, old).pvalue))
    pd.DataFrame(fix).to_csv(OUT / "tables/per_fix_effect.csv", index=False)

    c = values(data, "control", "freeway", "auc")[:18]
    t = values(data, "nstep", "freeway", "auc")[:18]
    pd.DataFrame([dict(game="freeway", arm="nstep", subset="corrected individuals 0-17 only", **paired(c, t))]
                 ).to_csv(OUT / "tables/sensitivity_nstep_freeway.csv", index=False)
    return pd.DataFrame(desc), tests


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def learning_curves(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    rng = np.random.default_rng(20261015)
    for ax, game in zip(axes.flat, GAMES):
        for arm in VARIANTS:
            curves = np.array([r["eval_trajectory"] for _, r in sorted(data[arm, game].items())])
            frames = np.arange(1, curves.shape[1] + 1) * 50_000 / 1e6
            boot = curves[rng.integers(0, len(curves), (2000, len(curves)))].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5], axis=0)
            ax.fill_between(frames, low, high, color=COLORS[arm], alpha=.15, linewidth=0)
            ax.plot(frames, curves.mean(axis=0), color=COLORS[arm], ls=STYLES[arm], lw=2, label=LABELS[arm])
        ax.set(title=NAMES[game], xlabel="Training frames (millions)", ylabel="Mean greedy evaluation return")
        ax.grid(alpha=.25, lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9)
    savefig(fig, "learning_curves")


def paired_effects(tests):
    df = tests["auc"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained", sharey=True)
    arms = VARIANTS[1:]
    for ax, game in zip(axes, GAMES):
        sub = df[df.game == game].set_index("arm").loc[arms]
        ctrl = sub.mean_control.iloc[0]
        for y, arm in enumerate(arms):
            r = sub.loc[arm]
            lo, hi, mid = (100 * v / ctrl for v in (r.ci98_75_low, r.ci98_75_high, r.mean_diff))
            ax.plot([lo, hi], [y, y], color=COLORS[arm], lw=2)
            ax.plot(mid, y, MARKERS[arm], color=COLORS[arm], ms=8, mfc=COLORS[arm] if r.significant else "white")
        ax.axvline(0, color="#888888", lw=1)
        ax.set(title=NAMES[game], xlabel="AUC difference vs DQN (% of DQN mean)")
        ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.25, lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired AUC effects with 98.75% CIs (filled marker: p < 0.0125)", fontsize=11)
    savefig(fig, "paired_auc_effects")


def auc_distributions(data):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), layout="constrained")
    rng = np.random.default_rng(7)
    for ax, game in zip(axes, GAMES):
        vals = [values(data, arm, game, "auc") / 1e6 for arm in VARIANTS]
        ax.boxplot(vals, showfliers=False, widths=.55, medianprops=dict(color="black"))
        for i, (arm, v) in enumerate(zip(VARIANTS, vals), 1):
            ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.7, color=COLORS[arm], marker=MARKERS[arm])
        ax.set_xticks(range(1, 6), [LABELS[a].replace(" DQN", "\nDQN") for a in VARIANTS], fontsize=8)
        ax.set(title=NAMES[game], ylabel="AUC (10$^6$ return × frames)")
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "auc_distributions")


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    data, source = load()
    desc, tests = tables(data, source)
    learning_curves(data)
    paired_effects(tests)
    auc_distributions(data)
    pd.set_option("display.width", 200)
    print(desc[["game", "arm", "auc_mean", "final_mean", "last10_mean", "last10_best"]].round(2).to_string(index=False))
    for metric, df in tests.items():
        print(f"\n== {metric}")
        print(df[["game", "arm", "pct_change", "p", "significant", "wilcoxon_p", "shapiro_p", "win_fraction"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
