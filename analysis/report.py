"""Figures, LaTeX tables and a numbers file for the IEEE report (report/main.tex).

Primary analysis: two-sided paired t-tests of each treatment against DQN on AUC,
Bonferroni-corrected over all 16 game x treatment comparisons (alpha = 0.05/16),
with matching simultaneous t intervals. The Wilcoxon signed-rank test and two
secondary endpoints (mean of the last ten evaluations, final evaluation) are
sensitivity analyses. Dataset assembly is shared with analysis.corrected_rct01.

    .venv/bin/python -m analysis.report
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.corrected_rct01 import CORRECTED, GAMES, NAMES, ORIGINAL, ROOT, VARIANTS, load, read, values
from rct.constants import POPULATION_SPACE

OUT = ROOT / "analysis_output/corrected-rct-01/report"
ALPHA = 0.05 / 16
TREATMENTS = VARIANTS[1:]
LABELS = dict(zip(VARIANTS, ["DQN", "Double DQN", "PER", "$n$-step DQN", "Averaged DQN"]))
SHORT = dict(zip(VARIANTS, ["DQN", "DDQN", "PER", "$n$-step", "Avg."]))
# Okabe-Ito hues with Tol's purple for Averaged DQN (passes the CVD/contrast checks);
# DQN is the reference arm, so it stays neutral ink.
COLORS = dict(zip(VARIANTS, ["#333333", "#0072B2", "#D55E00", "#009E73", "#AA4499"]))
STYLES = dict(zip(VARIANTS, ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]))
MARKERS = dict(zip(VARIANTS, ["o", "s", "^", "D", "v"]))
ENDPOINTS = ["auc", "last_10_return", "final_return"]
COL, TEXT = 252 / 72.27, 516 / 72.27  # IEEEtran conference column and text widths (in)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6, "pdf.fonttype": 42,
})


def paired(c, t):
    d = t - c
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    radius = stats.t.ppf(1 - ALPHA / 2, n - 1) * sd / np.sqrt(n)
    return dict(n=n, mean_control=c.mean(), mean_treatment=t.mean(), mean_diff=m,
                pct=100 * m / c.mean(), ci_low_pct=100 * (m - radius) / c.mean(),
                ci_high_pct=100 * (m + radius) / c.mean(), dz=m / sd, p_t=stats.ttest_rel(t, c).pvalue,
                p_w=stats.wilcoxon(d).pvalue, p_shapiro=stats.shapiro(d).pvalue, wins=int((d > 0).sum()))


def analyse(data):
    tests = {e: {g: {a: paired(values(data, "control", g, e), values(data, a, g, e)) for a in TREATMENTS}
                 for g in GAMES} for e in ENDPOINTS}
    desc = {g: {a: dict(mean=values(data, a, g, "auc").mean() / 1e6, sd=values(data, a, g, "auc").std(ddof=1) / 1e6)
                for a in VARIANTS} for g in GAMES}
    per_fix = {}
    for g in GAMES:
        old = np.array([r["auc"] for _, r in sorted(read(ORIGINAL, "per", g).items())])
        new = values(data, "per", g, "auc")
        per_fix[g] = dict(before=old.mean() / 1e6, after=new.mean() / 1e6,
                          pct=100 * (new.mean() - old.mean()) / old.mean(), p=stats.ttest_rel(new, old).pvalue)
    identical = sum(r["eval_trajectory"] == read(ORIGINAL, a, g)[i]["eval_trajectory"]
                    for a in ("ddqn", "nstep", "averaged") for g in GAMES
                    for i, r in read(CORRECTED, a, g).items())
    nstep_fw = paired(values(data, "control", "freeway", "auc")[:18], values(data, "nstep", "freeway", "auc")[:18])
    configs = [data["control", GAMES[0]][i]["config"] for i in range(30)]
    population = {k: {str(v): sum(c[k] == v for c in configs) for v in vals}
                  for k, vals in POPULATION_SPACE.items() if len(vals) > 1}
    lrs = POPULATION_SPACE["lr"]
    by_lr = {g: {a: [float(np.mean([r["auc"] for r in data[a, g].values() if r["config"]["lr"] == lr]) / 1e6)
                     for lr in lrs] for a in VARIANTS} for g in GAMES}
    return dict(alpha=ALPHA, tests=tests, desc=desc, per_fix=per_fix, identical_reruns=int(identical),
                nstep_freeway_0_17=nstep_fw, population=population, lr_levels=lrs, auc_by_lr=by_lr)


# ---------------------------------------------------------------- LaTeX tables

def fmt_p(p, bold=False):
    if p >= 1e-3:
        s = f"{p:#.2g}"
    else:
        mant, exp = f"{p:.1e}".split("e")
        s = rf"{mant}\times10^{{{int(exp)}}}"
    return rf"$\mathbf{{{s}}}$" if bold else f"${s}$"


def fmt_pct(x):
    return f"${x:+.1f}$" if abs(x) >= .05 else "$0.0$"


def table_main(res):
    rows = []
    for g in GAMES:
        d = res["desc"][g]
        rows.append(rf"{NAMES[g]} & DQN & {d['control']['mean']:.1f} ({d['control']['sd']:.1f}) "
                    r"& & & & & & \\")
        for a in TREATMENTS:
            r = res["tests"]["auc"][g][a]
            rows.append(rf" & {LABELS[a]} & {d[a]['mean']:.1f} ({d[a]['sd']:.1f}) & {fmt_pct(r['pct'])} "
                        rf"& $[{r['ci_low_pct']:+.1f}, {r['ci_high_pct']:+.1f}]$ & ${r['dz']:+.2f}$ "
                        rf"& {fmt_p(r['p_t'], r['p_t'] <= ALPHA)} & {fmt_p(r['p_w'], r['p_w'] <= ALPHA)} "
                        rf"& {r['wins']}/30 \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    head = (r"\begin{tabular}{llrrrrrrr}" "\n" r"\toprule" "\n"
            r"Game & Arm & AUC & $\Delta$ (\%) & 99.69\% CI (\%) & $d_z$ & $p_t$ & $p_W$ & Better \\" "\n"
            r"\midrule")
    return "\n".join([head, *rows, r"\end{tabular}"]) + "\n"


def table_robustness(res):
    rows = []
    for g in GAMES:
        for k, a in enumerate(TREATMENTS):
            cells = []
            for e in ENDPOINTS[1:]:
                r = res["tests"][e][g][a]
                cells += [fmt_pct(r["pct"]), fmt_p(r["p_t"], r["p_t"] <= ALPHA)]
            rows.append(rf"{NAMES[g] if k == 0 else ''} & {SHORT[a]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    head = (r"\begin{tabular}{llrrrr}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{2}{c}{Last ten evaluations} & \multicolumn{2}{c}{Final evaluation} \\" "\n"
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
            r"Game & Arm & $\Delta$ (\%) & $p_t$ & $\Delta$ (\%) & $p_t$ \\" "\n" r"\midrule")
    return "\n".join([head, *rows, r"\end{tabular}"]) + "\n"


def table_population(res):
    names = {"lr": "Learning rate", "buffer_capacity": "Replay buffer capacity", "batch_size": "Mini-batch size",
             "target_update_freq": "Target update period (steps)",
             "epsilon_decay_steps": r"$\varepsilon$-decay duration (steps)", "num_layers": "Hidden layers"}
    sci = {0.0001: r"$10^{-4}$", 0.0005: r"$5{\times}10^{-4}$", 0.001: r"$10^{-3}$", 10000: r"$10^4$",
           50000: r"$5{\times}10^4$", 100000: r"$10^5$", 200000: r"$2{\times}10^5$"}
    rows = []
    for k, label in names.items():
        cells = ", ".join(f"{sci.get(v, v)} ({res['population'][k][str(v)]})" for v in POPULATION_SPACE[k])
        rows.append(rf"{label} & {cells} \\")
    head = (r"\begin{tabular}{ll}" "\n" r"\toprule" "\n" r"Hyperparameter & Values (individuals in sample) \\"
            "\n" r"\midrule")
    return "\n".join([head, *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


# --------------------------------------------------------------------- figures

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.3, lw=.4)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def fig_learning_curves(data):
    fig, axes = plt.subplots(2, 2, figsize=(TEXT, 3.9), layout="constrained")
    rng = np.random.default_rng(20261015)
    for ax, game in zip(axes.flat, GAMES):
        for arm in VARIANTS:
            curves = np.array([r["eval_trajectory"] for _, r in sorted(data[arm, game].items())])
            frames = np.arange(1, curves.shape[1] + 1) * 50_000 / 1e6
            boot = curves[rng.integers(0, len(curves), (2000, len(curves)))].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5], axis=0)
            ax.fill_between(frames, low, high, color=COLORS[arm], alpha=.13, linewidth=0)
            ax.plot(frames, curves.mean(axis=0), color=COLORS[arm], ls=STYLES[arm], lw=1, label=LABELS[arm])
        ax.set_title(NAMES[game], pad=3)
        ax.set_xlim(0, 5.05)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Training frames (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean evaluation return")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False, handlelength=2.6)
    save(fig, "learning_curves")


def fig_effects(res):
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.75), layout="constrained", sharey=True)
    for ax, game in zip(axes.flat, GAMES):
        for y, arm in enumerate(TREATMENTS):
            r = res["tests"]["auc"][game][arm]
            sig = r["p_t"] <= ALPHA
            ax.plot([r["ci_low_pct"], r["ci_high_pct"]], [y, y], color=COLORS[arm], lw=1.2, solid_capstyle="round")
            ax.plot(r["pct"], y, MARKERS[arm], color=COLORS[arm], ms=4.5, mew=.9,
                    mfc=COLORS[arm] if sig else "white")
        ax.axvline(0, color="#777777", lw=.6, zorder=0)
        ax.set_title(NAMES[game], pad=3)
        ax.set_yticks(range(len(TREATMENTS)), [SHORT[a] for a in TREATMENTS])
        ax.set_ylim(len(TREATMENTS) - .5, -.5)
        style(ax)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("AUC difference from DQN (% of DQN mean)", fontsize=8)
    save(fig, "paired_effects")


def fig_auc_distributions(data):
    fig, axes = plt.subplots(2, 2, figsize=(COL, 3.0), layout="constrained")
    rng = np.random.default_rng(7)
    for ax, game in zip(axes.flat, GAMES):
        vals = [values(data, arm, game, "auc") / 1e6 for arm in VARIANTS]
        ax.boxplot(vals, showfliers=False, widths=.55, medianprops=dict(color="black", lw=.8),
                   boxprops=dict(lw=.6), whiskerprops=dict(lw=.6), capprops=dict(lw=.6))
        for i, (arm, v) in enumerate(zip(VARIANTS, vals), 1):
            ax.scatter(i + rng.uniform(-.17, .17, len(v)), v, s=5, alpha=.75, color=COLORS[arm],
                       marker=MARKERS[arm], linewidths=0, zorder=3)
        ax.set_xticks(range(1, 6), [SHORT[a] for a in VARIANTS], fontsize=6.5)
        ax.set_title(NAMES[game], pad=3)
        ax.set_ylim(bottom=0)
        style(ax)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"AUC ($10^6$)")
    save(fig, "auc_distributions")


def fig_learning_rate(res):
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.95), layout="constrained")
    x = range(len(res["lr_levels"]))
    for ax, game in zip(axes.flat, GAMES):
        for arm in VARIANTS:
            ax.plot(x, res["auc_by_lr"][game][arm], color=COLORS[arm], ls=STYLES[arm], lw=1, marker=MARKERS[arm],
                    ms=3.5, label=LABELS[arm])
        ax.set_xticks(list(x), [r"$10^{-4}$", r"$5{\times}10^{-4}$", r"$10^{-3}$"])
        ax.set_xlim(-.25, 2.25)
        ax.set_ylim(bottom=0)
        ax.set_title(NAMES[game], pad=3)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Learning rate")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Mean AUC ($10^6$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, handlelength=2.4,
               columnspacing=1.0)
    save(fig, "learning_rate")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load()
    res = analyse(data)
    (OUT / "numbers.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "tab_main.tex").write_text(table_main(res))
    (OUT / "tab_robustness.tex").write_text(table_robustness(res))
    (OUT / "tab_population.tex").write_text(table_population(res))
    fig_learning_curves(data)
    fig_effects(res)
    fig_auc_distributions(data)
    fig_learning_rate(res)
    for g in GAMES:
        for a in TREATMENTS:
            r = res["tests"]["auc"][g][a]
            print(f"{g:14s} {a:9s} {r['pct']:+6.1f}% [{r['ci_low_pct']:+6.1f}, {r['ci_high_pct']:+6.1f}] "
                  f"p_t={r['p_t']:.2g} p_w={r['p_w']:.2g} sig={r['p_t'] <= ALPHA}")


if __name__ == "__main__":
    main()
