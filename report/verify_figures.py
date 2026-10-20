"""Independent check of the report figures (run: .venv/bin/python report/verify_figures.py).

Expected values are recomputed from the raw result JSON with numpy/scipy only. The plotted values are read
from the matplotlib artists that analysis/report.py builds (its save() is intercepted), so this tests what
is actually drawn, not what the analysis code believes it drew.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GAMES = ["breakout", "asterix", "freeway", "spaceinvaders"]
NAMES = ["Breakout", "Asterix", "Freeway", "Space Invaders"]
ARMS = ["control", "ddqn", "per", "nstep", "averaged"]
SHORT = ["DQN", "DDQN", "PER", "$n$-step", "Avg."]
fails = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        fails.append(what)


def runs(exp, arm, game):
    (p,) = glob.glob(str(ROOT / "Results" / exp / "results" / arm / game / "*.json"))
    return {r["individual_id"]: r for r in json.load(open(p))["results"]}


# ---------------------------------------------------------------- raw data, assembled independently
raw = {}
for g in GAMES:
    raw["control", g] = runs("original-run-2026-09", "control", g)
    for a in ARMS[1:]:
        new = runs("corrected-rct-01", a, g)
        raw[a, g] = {**runs("original-run-2026-09", a, g), **new} if len(new) < 30 else new
assert all(sorted(v) == list(range(30)) for v in raw.values())


def trap(y):
    y = np.asarray(y, float)
    return float(np.sum((y[1:] + y[:-1]) / 2) * 50_000)


curves = {k: np.array([v[i]["eval_trajectory"] for i in range(30)], float) for k, v in raw.items()}
aucs = {k: np.array([trap(c) for c in curves[k]]) / 1e6 for k in raw}
lr = {g: np.array([raw["control", g][i]["config"]["lr"] for i in range(30)]) for g in GAMES}

# ---------------------------------------------------------------- capture the drawn figures
import matplotlib.pyplot as plt
from analysis import report as R

captured = {}
R.save = lambda fig, name: captured.__setitem__(name, fig)
data, _ = R.load()
res = R.analyse(data)
R.fig_learning_curves(data)
R.fig_effects(res)
R.fig_auc_density(data)
R.fig_learning_rate(res)
check(set(captured) == {"learning_curves", "paired_effects", "auc_density", "learning_rate"}, "all four data figures built")

# ---------------------------------------------------------------- Fig. learning curves
fig = captured["learning_curves"]
axes = [ax for ax in fig.axes]
check([ax.get_title() for ax in axes] == NAMES, "learning curves: panel titles in game order")
x_exp = np.arange(1, 101) * 0.05
for ax, g in zip(axes, GAMES):
    lines = [l for l in ax.lines if len(l.get_xdata()) == 100]
    check(len(lines) == 5, f"learning curves {g}: five arm lines")
    for line, a in zip(lines, ARMS):
        mean = curves[a, g].mean(axis=0)
        check(np.allclose(line.get_xdata(), x_exp) and np.allclose(line.get_ydata(), mean),
              f"learning curves {g}/{a}: line = mean of 30 raw curves at 0.05..5.0 M frames")
    rng = np.random.default_rng(1)
    for coll, a in zip(ax.collections, ARMS):
        verts = coll.get_paths()[0].vertices
        lo_hi = {}
        for xv, yv in verts:
            k = int(round(xv / 0.05)) - 1
            if 0 <= k < 100 and abs(xv - x_exp[k]) < 1e-9:
                lo_hi.setdefault(k, []).append(yv)
        lo = np.array([min(lo_hi[k]) for k in range(100)]); hi = np.array([max(lo_hi[k]) for k in range(100)])
        mean = curves[a, g].mean(axis=0)
        boot = curves[a, g][rng.integers(0, 30, (4000, 30))].mean(axis=1)
        elo, ehi = np.percentile(boot, [2.5, 97.5], axis=0)
        width = ehi - elo
        ok = (np.all(lo <= mean + 1e-9) and np.all(hi >= mean - 1e-9)
              and np.max(np.abs(lo - elo) / width) < .15 and np.max(np.abs(hi - ehi) / width) < .15)
        check(ok, f"learning curves {g}/{a}: band brackets the mean and matches an independent 95% bootstrap "
                  f"(max endpoint gap {max(np.max(np.abs(lo - elo) / width), np.max(np.abs(hi - ehi) / width)):.1%} of width)")
    check(ax.get_xlim()[0] <= 0.05 and ax.get_xlim()[1] >= 5.0, f"learning curves {g}: x-range shows all 100 evaluations")

# ---------------------------------------------------------------- Fig. paired effects
fig = captured["paired_effects"]
tcrit = stats.t.ppf(1 - 0.05 / 32, 29)
check([ax.get_title() for ax in fig.axes] == NAMES, "paired effects: panel titles in game order")
for ax, g in zip(fig.axes, GAMES):
    left = fig.axes[(GAMES.index(g) // 2) * 2]  # right-hand panels share the left panel's y-axis
    check(ax.get_shared_y_axes().joined(ax, left) and [t.get_text() for t in left.get_yticklabels()] == SHORT[1:]
          and list(ax.get_yticks()) == [0, 1, 2, 3], f"paired effects {g}: rows labelled DDQN, PER, n-step, Avg.")
    check(ax.get_xscale() == "log", f"paired effects {g}: log x-axis")
    segs = [l for l in ax.lines if len(l.get_xdata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1]]
    marks = [l for l in ax.lines if len(l.get_xdata()) == 1]
    vline = [l for l in ax.lines if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]]
    check(len(vline) == 1 and vline[0].get_xdata()[0] == 1, f"paired effects {g}: reference line at ratio 1")
    for i, a in enumerate(ARMS[1:]):
        d = np.log(aucs[a, g] / aucs["control", g])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(30)
        lo, hi, gm = np.exp(m - tcrit * se), np.exp(m + tcrit * se), np.exp(m)
        seg = [s for s in segs if s.get_ydata()[0] == i][0]
        mk = [s for s in marks if s.get_ydata()[0] == i][0]
        p_t = stats.ttest_rel(aucs[a, g], aucs["control", g]).pvalue
        filled = matplotlib.colors.to_hex(mk.get_markerfacecolor()) != "#ffffff"
        check(np.allclose(seg.get_xdata(), [lo, hi]) and np.isclose(mk.get_xdata()[0], gm),
              f"paired effects {g}/{a}: ratio {gm:.3f} and 99.69% interval [{lo:.3f}, {hi:.3f}]")
        check(filled == (p_t <= 0.05 / 16), f"paired effects {g}/{a}: marker {'filled' if filled else 'open'} "
                                             f"matches p_t={p_t:.2g} vs 0.05/16")
        x0, x1 = ax.get_xlim()
        check(x0 <= lo and x1 >= hi, f"paired effects {g}/{a}: interval fully inside the axis")

# ---------------------------------------------------------------- Fig. AUC density
fig = captured["auc_density"]
fig.canvas.draw()
subs = sorted(fig.subfigs, key=lambda s: (-s.bbox.y0, s.bbox.x0))  # reading order: top-left first
check([s._suptitle.get_text() for s in subs] == NAMES, "density: panels in reading order Breakout, Asterix, Freeway, Space Invaders")
for sub in subs:
    g = GAMES[NAMES.index(sub._suptitle.get_text())]  # data are checked against the game named in the title
    grid, *ridges = sub.axes
    check(len(ridges) == 5, f"density {g}: five ridges")
    for ax, a in zip(ridges, ARMS):
        v = aucs[a, g]
        check([t.get_text() for t in ax.get_yticklabels()] == [SHORT[ARMS.index(a)]], f"density {g}/{a}: ridge labelled {SHORT[ARMS.index(a)]}")
        kde_line, med_line = ax.lines[0], ax.lines[1]
        x, y = kde_line.get_xydata().T
        kde = stats.gaussian_kde(v, bw_method="scott")
        check(np.allclose(y, kde(x), rtol=1e-6, atol=1e-12), f"density {g}/{a}: curve = Gaussian KDE (Scott) of the 30 raw AUCs")
        sd_bw = np.sqrt(kde.covariance[0, 0])
        check(np.isclose(x.min(), max(0, v.min() - 3 * sd_bw)) and np.isclose(x.max(), v.max() + 3 * sd_bw),
              f"density {g}/{a}: support clipped at zero, else 3 bandwidths beyond the data")
        check(np.isclose(med_line.get_xdata()[0], np.median(v)) and np.isclose(med_line.get_ydata()[1], np.interp(np.median(v), x, y)),
              f"density {g}/{a}: dashed line at the median {np.median(v):.1f}, up to the curve")
        check(ax.get_xlim()[1] >= v.max() and ax.get_xlim()[0] <= v.min(), f"density {g}/{a}: all 30 AUCs inside the x-range")
        check(ax.get_ylim()[1] >= y.max() * .999, f"density {g}/{a}: peak not cut off")

# ---------------------------------------------------------------- Fig. learning rate
fig = captured["learning_rate"]
axes = [ax for ax in fig.axes]
check([ax.get_title() for ax in axes] == NAMES, "learning rate: panel titles in game order")
levels = sorted(set(lr["breakout"]))
check(levels == [1e-4, 5e-4, 1e-3], "learning rate: three levels 1e-4, 5e-4, 1e-3")
ns = [int((lr["breakout"] == L).sum()) for L in levels]
check(all(np.array_equal(lr[g], lr["breakout"]) for g in GAMES), "learning rate: same configurations in every game")
for ax, g in zip(axes, GAMES):
    labels = [t.get_text() for t in ax.get_xticklabels()]
    check(all(f"n={n}" in lab for n, lab in zip(ns, labels)), f"learning rate {g}: tick labels n = {ns}")
    for cont, a in zip(ax.containers, ARMS):
        line, _, (bars,) = cont.lines[0], cont.lines[1], cont.lines[2]
        mean = np.array([aucs[a, g][lr[g] == L].mean() for L in levels])
        se = np.array([aucs[a, g][lr[g] == L].std(ddof=1) / np.sqrt((lr[g] == L).sum()) for L in levels])
        segs = bars.get_segments()
        lo = np.array([s[:, 1].min() for s in segs]); hi = np.array([s[:, 1].max() for s in segs])
        check(np.allclose(line.get_ydata(), mean) and np.allclose(lo, mean - se) and np.allclose(hi, mean + se),
              f"learning rate {g}/{a}: points = subgroup mean AUC, bars = +/-1 SE")
    check(ax.get_ylim()[1] >= max(max(s[:, 1].max() for s in c.lines[2][0].get_segments()) for c in ax.containers),
          f"learning rate {g}: no error bar cut off at the top")

# ---------------------------------------------------------------- Fig. 1 (design diagram) claims
from rct.constants import POPULATION_SPACE
meta = json.load(open(glob.glob(str(ROOT / "Results/corrected-rct-01/results/per/freeway/*.json"))[0]))
check(sum(len(v) > 1 for v in POPULATION_SPACE.values()) == 6, "design: six sampled hyperparameters")
check(all(raw[a, g][i]["seed"] == 1337 + i for (a, g) in raw for i in range(30)), "design: training seed 1337 + j for every run")
check(meta["total_frames"] == 5_000_000 and meta["eval_frequency"] == 50_000, "design: 5e6 frames, evaluation every 5e4")
check(len(GAMES) * (len(ARMS) - 1) == 16, "design: 16 comparisons")

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
