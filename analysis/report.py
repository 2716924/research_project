"""Figures, LaTeX tables and a numbers file for the IEEE report (report/main.tex).

Primary analysis: two-sided paired t-tests of each treatment against DQN on AUC,
Bonferroni-corrected over all 16 game x treatment comparisons (alpha = 0.05/16),
with matching simultaneous t intervals. The Wilcoxon signed-rank test and two
secondary endpoints (mean of the last ten evaluations, final evaluation) are
sensitivity analyses. Exploratory analyses, added after the results were known:
per-individual log AUC ratios (geometric-mean ratio), Holm's procedure, the paired
probability of improvement with Wilson intervals, subsampling of individuals,
best-of-30 configurations, a regression of log AUC on the six sampled factors and
Averaged DQN by target-update period. Dataset assembly is
shared with analysis.corrected_rct01.

    .venv/bin/python -m analysis.report
"""
import itertools
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


def log_ratio(c, t):
    """Geometric-mean AUC ratio of treatment to DQN, with the family-wise t-interval on the log scale."""
    r = np.log(t / c)
    n, m, sd = len(r), r.mean(), r.std(ddof=1)
    radius = stats.t.ppf(1 - ALPHA / 2, n - 1) * sd / np.sqrt(n)
    return dict(ratio=np.exp(m), low=np.exp(m - radius), high=np.exp(m + radius),
                p_r=stats.ttest_1samp(r, 0).pvalue, p_w=stats.wilcoxon(r).pvalue, p_shapiro=stats.shapiro(r).pvalue,
                lowest=sorted(np.exp(r))[:2])


def wilson(k, n, z=stats.norm.ppf(.975)):
    p, den = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def holm(pvalues):
    """Number of hypotheses Holm's step-down procedure rejects at family-wise level 0.05."""
    for k, p in enumerate(sorted(pvalues)):
        if p > .05 / (len(pvalues) - k):
            return k
    return len(pvalues)


def subsample(d, k, rng, draws=100_000):
    """Share of k-individual subsamples whose mean difference has the full-sample sign, and whose
    paired t-test (alpha = 0.05, uncorrected) is significant in that or the opposite direction.
    Exhaustive for k <= 5, Monte Carlo otherwise."""
    if k <= 5:
        idx = np.array(list(itertools.combinations(range(len(d)), k)))
    else:
        idx = np.argsort(rng.random((draws, len(d))), axis=1)[:, :k]
    sub, sign = d[idx], np.sign(d.mean())
    m = sub.mean(axis=1)
    out = dict(same_sign=float((np.sign(m) == sign).mean()))
    if k > 1:
        p = 2 * stats.t.sf(np.abs(m / (sub.std(axis=1, ddof=1) / np.sqrt(k))), k - 1)
        out.update(sig_same=float(((p <= .05) & (np.sign(m) == sign)).mean()),
                   sig_opposite=float(((p <= .05) & (np.sign(m) != sign)).mean()))
    return out


FACTORS = [k for k, v in POPULATION_SPACE.items() if len(v) > 1]


def design(configs, drop=None):
    """Treatment-coded design matrix of the six sampled factors (first level is the reference)."""
    cols, names = [np.ones(len(configs))], ["const"]
    for k in FACTORS:
        if k != drop:
            for v in POPULATION_SPACE[k][1:]:
                cols.append(np.array([c[k] == v for c in configs], float))
                names.append(f"{k}={v}")
    return np.column_stack(cols), names


def regression(configs, y):
    """OLS of log AUC on the six factors: adjusted learning-rate ratio (1e-3 vs 1e-4) with a 95%
    interval, partial R^2 of the learning rate and the overall R^2."""
    X, names = design(configs)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    rss, df = resid @ resid, len(y) - X.shape[1]
    se = np.sqrt(np.diag(rss / df * np.linalg.inv(X.T @ X)))
    Xr = design(configs, drop="lr")[0]
    rr = y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    j, q = names.index("lr=0.001"), stats.t.ppf(.975, df)
    return dict(lr_ratio=np.exp(beta[j]), lr_low=np.exp(beta[j] - q * se[j]), lr_high=np.exp(beta[j] + q * se[j]),
                partial_r2_lr=(rr @ rr - rss) / (rr @ rr), r2=1 - rss / ((y - y.mean()) ** 2).sum(),
                target_ratio={v: np.exp(beta[names.index(f"target_update_freq={v}")]) for v in (1000, 2000)})


def analyse(data):
    tests = {e: {g: {a: paired(values(data, "control", g, e), values(data, a, g, e)) for a in TREATMENTS}
                 for g in GAMES} for e in ENDPOINTS}
    ratio = {g: {a: log_ratio(values(data, "control", g, "auc"), values(data, a, g, "auc")) for a in TREATMENTS}
             for g in GAMES}
    poi = {g: {a: dict(wins=tests["auc"][g][a]["wins"], p=tests["auc"][g][a]["wins"] / 30,
                       low=wilson(tests["auc"][g][a]["wins"], 30)[0], high=wilson(tests["auc"][g][a]["wins"], 30)[1])
               for a in TREATMENTS} for g in GAMES}
    p_all = [tests["auc"][g][a]["p_t"] for g in GAMES for a in TREATMENTS]
    sig = [(g, a) for g in GAMES for a in TREATMENTS if tests["auc"][g][a]["p_t"] <= ALPHA]
    rng = np.random.default_rng(20261019)
    subsampling = {f"{g}/{a}": {k: subsample(values(data, a, g, "auc") - values(data, "control", g, "auc"), k, rng)
                                for k in (1, 3, 5, 10)} for g, a in sig}
    best = {}
    for g in GAMES:
        c = values(data, "control", g, "auc")
        best[g] = {a: dict(pct=100 * (values(data, a, g, "auc").max() / c.max() - 1),
                           individual=int(values(data, a, g, "auc").argmax()), dqn_individual=int(c.argmax()))
                   for a in TREATMENTS}
    configs = {g: [data["control", g][i]["config"] for i in range(30)] for g in GAMES}
    reg = {g: {a: regression(configs[g], np.log(values(data, a, g, "auc"))) for a in VARIANTS} for g in GAMES}
    periods = POPULATION_SPACE["target_update_freq"]
    by_period = {}
    for g in GAMES:
        c, t = values(data, "control", g, "auc"), values(data, "averaged", g, "auc")
        period = np.array([cfg["target_update_freq"] for cfg in configs[g]])
        by_period[g] = {str(v): dict(n=int((period == v).sum()), ratio=float(np.exp(np.log(t / c)[period == v].mean())),
                                     wins=int((t > c)[period == v].sum())) for v in periods}
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
    lr_groups = {g: {a: [np.array([r["auc"] for r in data[a, g].values() if r["config"]["lr"] == lr]) / 1e6
                         for lr in lrs] for a in VARIANTS} for g in GAMES}
    by_lr = {g: {a: [float(v.mean()) for v in lr_groups[g][a]] for a in VARIANTS} for g in GAMES}
    se_by_lr = {g: {a: [float(v.std(ddof=1) / np.sqrt(len(v))) for v in lr_groups[g][a]] for a in VARIANTS}
                for g in GAMES}
    lr_n = [int(len(v)) for v in lr_groups[GAMES[0]]["control"]]
    # Like-for-like comparison on the log scale: adjusted DQN learning-rate effect vs geometric-mean treatment effect.
    lr_exceeds = [f"{g}/{a}" for g in GAMES for a in TREATMENTS
                  if abs(np.log(reg[g]["control"]["lr_ratio"])) > abs(np.log(ratio[g][a]["ratio"]))]
    return dict(alpha=ALPHA, tests=tests, desc=desc, per_fix=per_fix, identical_reruns=int(identical),
                nstep_freeway_0_17=nstep_fw, population=population, lr_levels=lrs, auc_by_lr=by_lr,
                auc_se_by_lr=se_by_lr, lr_n=lr_n, ratio=ratio, poi=poi, holm_rejections=holm(p_all),
                p_sorted=sorted(p_all), significant=[f"{g}/{a}" for g, a in sig], subsampling=subsampling,
                best_of_30=best, regression=reg, lr_effect_exceeds_treatment=lr_exceeds,
                averaged_by_target_period=by_period)


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
            r, q = res["tests"]["auc"][g][a], res["ratio"][g][a]
            rows.append(rf" & {LABELS[a]} & {d[a]['mean']:.1f} ({d[a]['sd']:.1f}) & {fmt_pct(r['pct'])} "
                        rf"& ${q['ratio']:.2f}$ $[{q['low']:.2f}, {q['high']:.2f}]$ & ${r['dz']:+.2f}$ "
                        rf"& {fmt_p(r['p_t'], r['p_t'] <= ALPHA)} & {fmt_p(r['p_w'], r['p_w'] <= ALPHA)} "
                        rf"& {fmt_p(q['p_r'], q['p_r'] <= ALPHA)} \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    head = (r"\begin{tabular}{llrrrrrrr}" "\n" r"\toprule" "\n"
            r"Game & Arm & AUC & $\Delta$ (\%) & Ratio (99.69\% CI) & $d_z$ & $p_t$ & $p_W$ & $p_r$ \\" "\n"
            r"\midrule")
    return "\n".join([head, *rows, r"\end{tabular}"]) + "\n"


def table_reliability(res):
    """RQ2: paired probability of improvement, best-of-30 comparison and subsampled trials."""
    rows = []
    for g in GAMES:
        rows.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{NAMES[g]}}}}} \\")
        for a in TREATMENTS:
            p, b = res["poi"][g][a], res["best_of_30"][g][a]
            sub = res["subsampling"].get(f"{g}/{a}")
            cells = [f"{100 * sub[n]['sig_same']:.0f}" for n in (3, 5, 10)] if sub else ["--"] * 3
            rows.append(rf"\quad {SHORT[a]} & ${p['p']:.2f}$ $[{p['low']:.2f}, {p['high']:.2f}]$ "
                        rf"& {fmt_pct(b['pct'])} & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{@{}lrrrrr@{}}" "\n" r"\toprule" "\n"
            r" & & Best of 30 & \multicolumn{3}{c@{}}{Detected (\%)} \\" "\n"
            r"\cmidrule(l){4-6}" "\n"
            r"Arm & PoI (95\% CI) & $\Delta$ (\%) & $k=3$ & 5 & 10 \\" "\n" r"\midrule")
    return "\n".join([head, *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_regression(res):
    """Appendix: adjusted learning-rate effect from the six-factor regression of log AUC."""
    rows = []
    for g in GAMES:
        for k, a in enumerate(VARIANTS):
            r = res["regression"][g][a]
            rows.append(rf"{NAMES[g] if k == 0 else ''} & {SHORT[a]} & ${r['lr_ratio']:.2f}$ "
                        rf"$[{r['lr_low']:.2f}, {r['lr_high']:.2f}]$ & ${r['partial_r2_lr']:.2f}$ & ${r['r2']:.2f}$ \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    head = (r"\begin{tabular}{llrrr}" "\n" r"\toprule" "\n"
            r"Game & Arm & LR ratio (95\% CI) & Partial $R^2$ & $R^2$ \\" "\n" r"\midrule")
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
    fig, axes = plt.subplots(2, 2, figsize=(TEXT, 3.1), layout="constrained")
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
    """Geometric-mean AUC ratio to DQN on a log axis; filled markers follow the primary t-test."""
    ticks = {"breakout": [.9, 1, 1.1, 1.2], "asterix": [.25, .5, 1, 2], "freeway": [.25, .5, 1],
             "spaceinvaders": [.5, 1]}
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.55), layout="constrained", sharey=True)
    for ax, game in zip(axes.flat, GAMES):
        for y, arm in enumerate(TREATMENTS):
            q = res["ratio"][game][arm]
            sig = res["tests"]["auc"][game][arm]["p_t"] <= ALPHA
            ax.plot([q["low"], q["high"]], [y, y], color=COLORS[arm], lw=1.2, solid_capstyle="round")
            ax.plot(q["ratio"], y, MARKERS[arm], color=COLORS[arm], ms=4.5, mew=.9,
                    mfc=COLORS[arm] if sig else "white")
        ax.axvline(1, color="#777777", lw=.6, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(ticks[game], [f"{t:g}" for t in ticks[game]])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(NAMES[game], pad=3)
        ax.set_yticks(range(len(TREATMENTS)), [SHORT[a] for a in TREATMENTS])
        ax.set_ylim(len(TREATMENTS) - .5, -.5)
        style(ax)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("AUC ratio to DQN (geometric mean, log scale)", fontsize=8)
    save(fig, "paired_effects")


def fig_auc_density(data):
    """Ridgeline of the AUC over individuals, after Fig. 3 of Pretorius et al. (2020): one overlapping axis
    per arm with a seaborn Gaussian kernel density (Scott's bandwidth, clipped at zero because the AUC is
    positive), each ridge scaled to its own peak; a dashed line marks the arm's median."""
    import seaborn as sns
    fig = plt.figure(figsize=(COL, 3.4))
    style = {"axes.facecolor": (0, 0, 0, 0), **{k: plt.rcParams[k] for k in ("font.family", "font.serif")}}
    with sns.axes_style("white", rc=style):
        for k, (sub, game) in enumerate(zip(fig.subfigures(2, 2, hspace=.02, wspace=.03).flat, GAMES)):
            vals = {arm: values(data, arm, game, "auc") / 1e6 for arm in VARIANTS}
            xmax = 1.04 * max(v.max() for v in vals.values())
            grid = sub.add_axes([.22, .2, .75, .7], zorder=-1)  # one continuous grid behind all ridges
            axes = sub.subplots(len(VARIANTS), 1, sharex=True)
            sub.subplots_adjust(left=.22, right=.97, top=.9, bottom=.2, hspace=-.4)
            for ax, arm in zip(axes, VARIANTS):
                kw = dict(bw_method="scott", clip=(0, None), ax=ax)
                sns.kdeplot(vals[arm], fill=True, color=COLORS[arm], alpha=.4, linewidth=0, **kw)
                sns.kdeplot(vals[arm], color=COLORS[arm], linewidth=.9, **kw)
                x, y = ax.lines[-1].get_xydata().T
                med = float(np.median(vals[arm]))
                ax.plot([med, med], [0, np.interp(med, x, y)], color=COLORS[arm], lw=.8, ls=(0, (2.5, 1.5)))
                ax.axhline(0, color=COLORS[arm], lw=.9, clip_on=False)
                ax.set(xlim=(0, xmax), ylabel="")
                ax.set_ylim(bottom=0)
                ax.set_yticks([0], [SHORT[arm]])
                ax.tick_params(axis="y", length=2.5, width=.6, pad=1.5, labelsize=7)
                sns.despine(ax=ax, left=True, bottom=ax is not axes[-1])
                if ax is not axes[-1]:
                    ax.tick_params(axis="x", length=0)
            grid.set(xlim=(0, xmax), ylim=(0, 1))
            grid.axis("off")
            for t in axes[-1].get_xticks():
                if 0 < t < xmax:
                    grid.axvline(t, color="#e3e3e3", lw=.6)
            axes[-1].tick_params(axis="x", labelsize=7, length=2.5, width=.6, pad=1.5)
            axes[-1].spines["bottom"].set_linewidth(.6)
            axes[-1].set_xlabel(r"AUC ($10^6$)" if k >= 2 else "", labelpad=1.5)
            sub.suptitle(NAMES[game], fontsize=8, y=.99)
    save(fig, "auc_density")


def fig_learning_rate(res):
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.9), layout="constrained")
    x = np.arange(len(res["lr_levels"]))
    for ax, game in zip(axes.flat, GAMES):
        for k, arm in enumerate(VARIANTS):
            # Small horizontal offsets keep the +/- 1 standard error bars of the five arms apart.
            ax.errorbar(x + (k - 2) * .07, res["auc_by_lr"][game][arm], yerr=res["auc_se_by_lr"][game][arm],
                        color=COLORS[arm], ls=STYLES[arm], lw=1, marker=MARKERS[arm], ms=3.2, elinewidth=.6,
                        capsize=0, label=LABELS[arm])
        ax.set_xticks(list(x), [rf"$10^{{-4}}$" "\n" rf"($n={res['lr_n'][0]}$)",
                                rf"$5{{\times}}10^{{-4}}$" "\n" rf"($n={res['lr_n'][1]}$)",
                                rf"$10^{{-3}}$" "\n" rf"($n={res['lr_n'][2]}$)"])
        ax.set_xlim(-.3, 2.3)
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
    (OUT / "tab_reliability.tex").write_text(table_reliability(res))
    (OUT / "tab_regression.tex").write_text(table_regression(res))
    fig_learning_curves(data)
    fig_effects(res)
    fig_auc_density(data)
    fig_learning_rate(res)
    for g in GAMES:
        for a in TREATMENTS:
            r = res["tests"]["auc"][g][a]
            print(f"{g:14s} {a:9s} {r['pct']:+6.1f}% [{r['ci_low_pct']:+6.1f}, {r['ci_high_pct']:+6.1f}] "
                  f"p_t={r['p_t']:.2g} p_w={r['p_w']:.2g} sig={r['p_t'] <= ALPHA}")


if __name__ == "__main__":
    main()
