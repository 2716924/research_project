"""Independent check of every number in the report against the raw result JSON.

Deliberately imports nothing from analysis/ or rct/: it reassembles the dataset from
Results/, recomputes all statistics, compares each cell of the generated tables and
asserts that every number quoted in main.tex appears there. It also scans main.tex for
American spellings, leftover template text, and unresolved references in the LaTeX log.

    .venv/bin/python report/verify_report.py [path/to/main.tex]
"""
import glob
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
GEN = ROOT / "analysis_output/corrected-rct-01/report"
GAMES = ["breakout", "asterix", "freeway", "spaceinvaders"]
ARMS = ["control", "ddqn", "per", "nstep", "averaged"]
ALPHA = 0.05 / 16
failures = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def runs(exp, arm, game):
    (path,) = glob.glob(str(ROOT / "Results" / exp / "results" / arm / game / "*.json"))
    return {r["individual_id"]: r for r in json.load(open(path))["results"]}


def auc(curve):
    y = np.asarray(curve, float)
    return float(np.sum((y[1:] + y[:-1]) / 2) * 50_000)


# ---------------------------------------------------------------- dataset
data = {}
for g in GAMES:
    data["control", g] = runs("original-run-2026-09", "control", g)
    for a in ARMS[1:]:
        new = runs("corrected-rct-01", a, g)
        old = runs("original-run-2026-09", a, g)
        data[a, g] = {**old, **new} if len(new) < 30 else new
check(all(sorted(v) == list(range(30)) for v in data.values()) and len(data) * 30 == 600, "600 runs, 30 per cell")
check(all(abs(auc(r["eval_trajectory"]) - r["auc"]) < 1e-6 for v in data.values() for r in v.values()),
      "every stored AUC recomputes from its curve")
check(all((r["seed"], r["config"]) == (data["control", g][i]["seed"], data["control", g][i]["config"])
          for (a, g), v in data.items() for i, r in v.items()), "seed and configuration identical across arms")
same = sum(r["eval_trajectory"] == runs("original-run-2026-09", a, g)[i]["eval_trajectory"]
           for a in ("ddqn", "nstep", "averaged") for g in GAMES for i, r in runs("corrected-rct-01", a, g).items())
check(same == 348, f"non-PER arms reproduce all overlapping curves exactly ({same} of 348)")


def endpoint(arm, g, kind):
    out = []
    for i in range(30):
        t = data[arm, g][i]["eval_trajectory"]
        out.append({"auc": auc(t), "last10": float(np.mean(t[-10:])), "final": t[-1]}[kind])
    return np.array(out)


def paired(c, t):
    d = t - c
    rad = stats.t.ppf(1 - ALPHA / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(pct=100 * d.mean() / c.mean(), lo=100 * (d.mean() - rad) / c.mean(), hi=100 * (d.mean() + rad) / c.mean(),
                dz=d.mean() / d.std(ddof=1), pt=stats.ttest_rel(t, c).pvalue, pw=stats.wilcoxon(d).pvalue,
                ps=stats.shapiro(d).pvalue, wins=int((d > 0).sum()))


def ratio_scale(c, t):
    """Geometric-mean ratio from the log ratios, its 99.69% interval, one-sample t and Wilcoxon p, Shapiro p."""
    lr = np.log(t) - np.log(c)
    rad = stats.t.ppf(1 - ALPHA / 2, len(lr) - 1) * lr.std(ddof=1) / np.sqrt(len(lr))
    return dict(gm=np.exp(lr.mean()), lo=np.exp(lr.mean() - rad), hi=np.exp(lr.mean() + rad),
                pr=stats.ttest_1samp(lr, 0.0).pvalue, pw=stats.wilcoxon(lr).pvalue, ps=stats.shapiro(lr).pvalue,
                two_lowest=np.sort(t / c)[:2])


T = {k: {(g, a): paired(endpoint("control", g, k), endpoint(a, g, k)) for g in GAMES for a in ARMS[1:]}
     for k in ("auc", "last10", "final")}
R = {(g, a): ratio_scale(endpoint("control", g, "auc"), endpoint(a, g, "auc")) for g in GAMES for a in ARMS[1:]}
sig = {key for key, r in T["auc"].items() if r["pt"] <= ALPHA}
mean = {(a, g): endpoint(a, g, "auc").mean() / 1e6 for a in ARMS for g in GAMES}
sd = {(a, g): endpoint(a, g, "auc").std(ddof=1) / 1e6 for a in ARMS for g in GAMES}
curve = {(a, g): np.mean([data[a, g][i]["eval_trajectory"] for i in range(30)], axis=0) for a in ARMS for g in GAMES}
lrs = [0.0001, 0.0005, 0.001]
by_lr = {(a, g): [np.mean([r["auc"] for r in data[a, g].values() if r["config"]["lr"] == lr]) / 1e6 for lr in lrs]
         for a in ARMS for g in GAMES}

# ------------------------------------------------------- generated tables
def fmt_p(p):
    if p >= 1e-3:
        return f"{p:#.2g}"
    m, e = f"{p:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def strip(s):
    return re.sub(r"\s+", " ", s.replace("$", "").replace("\\mathbf{", "").replace("}", "").replace("{", ""))


tab = strip((GEN / "tab_main.tex").read_text())
names = dict(zip(GAMES, ["Breakout", "Asterix", "Freeway", "Space Invaders"]))
labels = dict(zip(ARMS, ["DQN", "Double DQN", "PER", "n-step DQN", "Averaged DQN"]))
for g in GAMES:
    check(f"DQN & {mean['control', g]:.1f} ({sd['control', g]:.1f})" in tab, f"Table III DQN row, {g}")
    for a in ARMS[1:]:
        r, q = T["auc"][g, a], R[g, a]
        row = (f"{labels[a]} & {mean[a, g]:.1f} ({sd[a, g]:.1f}) & {r['pct']:+.1f} & {q['gm']:.2f} [{q['lo']:.2f}, {q['hi']:.2f}] "
               f"& {r['dz']:+.2f} & {strip(fmt_p(r['pt']))} & {strip(fmt_p(r['pw']))} & {strip(fmt_p(q['pr']))}")
        check(row in tab, f"Table III row {g}/{a}")
rob = strip((GEN / "tab_robustness.tex").read_text())
short = dict(zip(ARMS, ["DQN", "DDQN", "PER", "n-step", "Avg."]))
for g in GAMES:
    for a in ARMS[1:]:
        l, f = T["last10"][g, a], T["final"][g, a]
        cell = f"{short[a]} & {l['pct']:+.1f} & {strip(fmt_p(l['pt']))} & {f['pct']:+.1f} & {strip(fmt_p(f['pt']))}"
        check(cell in rob, f"Table IV row {g}/{a}")
pop = strip((GEN / "tab_population.tex").read_text())
counts = {k: [sum(data["control", "breakout"][i]["config"][k] == v for i in range(30)) for v in vals]
          for k, vals in {"lr": lrs, "buffer_capacity": [10000, 50000, 100000], "batch_size": [32, 64],
                          "target_update_freq": [500, 1000, 2000], "epsilon_decay_steps": [50000, 100000, 200000],
                          "num_layers": [1, 2]}.items()}
for k, c in counts.items():
    check(all(f"({n})" in pop for n in c) and sum(c) == 30, f"Table I counts {k} = {c}")


# ------------------------------------------- reliability of smaller evaluations
def wilson_bounds(k, n, z=stats.norm.ppf(0.975)):
    # Roots of (k/n - p)^2 = z^2 p (1 - p) / n, solved as a quadratic in p.
    a, b, c = 1 + z * z / n, -(2 * k / n + z * z / n), (k / n) ** 2
    disc = np.sqrt(b * b - 4 * a * c)
    return (-b - disc) / (2 * a), (-b + disc) / (2 * a)


def detect(d, k, subsets):
    sub = d[subsets]
    res = stats.ttest_1samp(sub, 0.0, axis=1)
    agree = np.sign(sub.mean(axis=1)) == np.sign(d.mean())
    return dict(sign=agree.mean(), same=((res.pvalue <= .05) & agree).mean(), opposite=((res.pvalue <= .05) & ~agree).mean())


diffs = {key: endpoint(key[1], key[0], "auc") - endpoint("control", key[0], "auc") for key in sig}
rng = np.random.default_rng(7)
mc10 = np.array([rng.permutation(30)[:10] for _ in range(50_000)])
SUB = {key: {3: detect(d, 3, np.array(list(itertools.combinations(range(30), 3)))),
             5: detect(d, 5, np.array(list(itertools.combinations(range(30), 5)))),
             10: detect(d, 10, mc10)} for key, d in diffs.items()}
best = {(g, a): 100 * (endpoint(a, g, "auc").max() / endpoint("control", g, "auc").max() - 1) for g in GAMES for a in ARMS[1:]}
rel = strip((GEN / "tab_reliability.tex").read_text())
rel_blocks = {g: rel[rel.index(f"\\emph{names[g]}"):] for g in GAMES}
for i, g in enumerate(GAMES):
    block = rel_blocks[g][:rel_blocks[g].index(f"\\emph{names[GAMES[i + 1]]}")] if i < 3 else rel_blocks[g]
    for a in ARMS[1:]:
        lo, hi = wilson_bounds(T["auc"][g, a]["wins"], 30)
        head = f"{short[a]} & {T['auc'][g, a]['wins'] / 30:.2f} [{lo:.2f}, {hi:.2f}] & {best[g, a]:+.1f} & "
        if (g, a) in sig:
            s = SUB[g, a]
            m = re.search(re.escape(head) + r"(\d+) & (\d+) & (\d+)", block)
            ok = (m is not None and int(m.group(1)) == round(100 * s[3]["same"]) and int(m.group(2)) == round(100 * s[5]["same"])
                  and abs(int(m.group(3)) - 100 * s[10]["same"]) <= 1.5)
        else:
            ok = head + "-- & -- & --" in block
        check(ok, f"Table IV row {g}/{a}")

# ------------------------------------------------- six-factor regression
LEVELS = {"lr": lrs, "buffer_capacity": [10000, 50000, 100000], "batch_size": [32, 64],
          "target_update_freq": [500, 1000, 2000], "epsilon_decay_steps": [50000, 100000, 200000], "num_layers": [1, 2]}


def dummies(g, drop=None):
    # Last level as the reference (the analysis uses the first), so the contrast is computed differently.
    cfg = [data["control", g][i]["config"] for i in range(30)]
    cols = [np.ones(30)] + [np.array([c[k] == v for c in cfg], float)
                            for k, vals in LEVELS.items() if k != drop for v in vals[:-1]]
    return np.column_stack(cols)


def ols_qr(X, y):
    q, r = np.linalg.qr(X)
    beta = np.linalg.solve(r, q.T @ y)
    e = y - X @ beta
    return beta, e @ e, np.linalg.inv(r.T @ r)


REG = {}
for g in GAMES:
    for a in ARMS:
        y = np.log(endpoint(a, g, "auc"))
        beta, rss, xtx_inv = ols_qr(dummies(g), y)
        df = 30 - len(beta)
        rss_without = ols_qr(dummies(g, drop="lr"), y)[1]
        se = np.sqrt(rss / df * xtx_inv[1, 1])  # column 1 is lr = 1e-4 against the 1e-3 reference
        q = stats.t.ppf(.975, df)
        tu = [np.exp(beta[1 + 2 + 2 + 1 + i]) for i in range(2)]  # target period 500, 1000 vs 2000
        REG[g, a] = dict(ratio=np.exp(-beta[1]), lo=np.exp(-beta[1] - q * se), hi=np.exp(-beta[1] + q * se),
                         partial=(rss_without - rss) / rss_without, r2=1 - rss / ((y - y.mean()) ** 2).sum(),
                         period_2000_vs_500=1 / tu[0], df=df)
reg_tab = strip((GEN / "tab_regression.tex").read_text())
for i, g in enumerate(GAMES):
    start = reg_tab.index(names[g])
    block = reg_tab[start:reg_tab.index(names[GAMES[i + 1]])] if i < 3 else reg_tab[start:]
    for a in ARMS:
        r = REG[g, a]
        check(f"{short[a]} & {r['ratio']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] & {r['partial']:.2f} & {r['r2']:.2f}" in block,
              f"Table VI row {g}/{a}")
check(all(r["df"] == 19 for r in REG.values()), "regression has 19 residual degrees of freedom")

# ------------------------------------------------- numbers quoted in the text
tex = Path(sys.argv[1] if len(sys.argv) > 1 else REPORT / "main.tex").read_text()
text = re.sub(r"\s+", " ", tex.replace("$", "").replace("~", " "))
A = T["auc"]
r1 = lambda x: f"{x:.1f}"
pct = lambda g, a, k="auc": f"{T[k][g, a]['pct']:+.1f}\\%"
lr_ratio = {g: by_lr["control", g][0] / by_lr["control", g][2] for g in GAMES}
# Like-for-like on the log scale: adjusted DQN learning-rate effect against the geometric-mean treatment effect.
exceeds = [(g, a) for g in GAMES for a in ARMS[1:] if abs(np.log(R[g, a]["gm"])) >= abs(np.log(REG[g, "control"]["ratio"]))]
nstep_r = {g: [by_lr["nstep", g][k] / by_lr["control", g][k] for k in range(3)] for g in GAMES}
ratio_sig = [key for key in sig if R[key]["pr"] <= ALPHA]
ratio_lost = sorted(set(sig) - set(ratio_sig))
holm_p = sorted(r["pt"] for r in A.values())
holm_k = next((k for k, p in enumerate(holm_p) if p > .05 / (16 - k)), 16)
reversed_best = [key for key in sig if np.sign(best[key]) != np.sign(T["auc"][key]["pct"])]
lr_col = {g: np.array([data["control", g][i]["config"]["lr"] for i in range(30)]) for g in GAMES}
period = {g: np.array([data["control", g][i]["config"]["target_update_freq"] for i in range(30)]) for g in GAMES}
avg_period = {g: [np.exp(np.mean(np.log(endpoint("averaged", g, "auc") / endpoint("control", g, "auc"))[period[g] == v]))
                  for v in (500, 1000, 2000)] for g in GAMES}
sub_range = {k: [min(SUB[key][k]["same"] for key in sig), max(SUB[key][k]["same"] for key in sig)] for k in (3, 5, 10)}
sign_range = {k: min(SUB[key][k]["sign"] for key in sig) for k in (3, 5)}
weakest = sorted(sig, key=lambda key: SUB[key][10]["same"])[:2]


seaquest_aborted = sorted({p.split("/logs/")[1].split("/")[0] for p in glob.glob(str(ROOT / "Results/corrected-rct-01/logs/*/seaquest/*.err"))
                           if "exceeded 100000 steps" in open(p, errors="ignore").read()})
versions = {(json.load(open(f))["provenance"]["package_versions"]["torch"], json.load(open(f))["provenance"]["package_versions"]["minatar"])
            for f in glob.glob(str(ROOT / "Results/corrected-rct-01/results/*/*/*.json"))}
fw_lo_n = endpoint("nstep", "freeway", "auc")[lr_col["freeway"] == 1e-4] / 1e6
fw_lo_c = endpoint("control", "freeway", "auc")[lr_col["freeway"] == 1e-4] / 1e6
aborted = [p for p in glob.glob(str(ROOT / "Results/corrected-rct-01/logs/*/*/*.err"))
           if "exceeded 100000 steps" in open(p, errors="ignore").read() and "/seaquest/" not in p]
complete_curves = all(len(r["eval_trajectory"]) == 100 for v in data.values() for r in v.values())
per_r = [by_lr["per", g][k] / by_lr["control", g][k] for g in ("freeway", "spaceinvaders") for k in range(3)]
nonsig_become = [(k, key) for k in ("last10", "final") for key, r in T[k].items() if key not in sig and r["pt"] <= ALPHA]
d_sig = sorted(abs(A[k]["dz"]) for k in sig)
cv = {g: 100 * sd["control", g] / mean["control", g] for g in GAMES}
best_lr_exceptions = sorted((a, g) for a in ARMS for g in GAMES if int(np.argmax(by_lr[a, g])) != 0)

quotes = {
    "abstract: Averaged per cent": "by " + ", ".join(f"{A[g, 'averaged']['pct']:.1f}" for g in GAMES[1:3]) + f" and {A['spaceinvaders', 'averaged']['pct']:.1f} per cent",
    "abstract: DDQN per cent": f"by {A['spaceinvaders', 'ddqn']['pct']:.1f} per cent",
    "abstract: up to 80 per cent": "by up to 80 per cent" if round(-min(r["pct"] for r in A.values())) == 80 else None,
    "abstract: seven of nine on ratio scale": "seven of these nine findings remained significant; the gains of Averaged DQN in Asterix and of Double DQN in Space Invaders did not" if len(ratio_sig) == 7 and ratio_lost == [("asterix", "averaged"), ("spaceinvaders", "ddqn")] else None,
    "abstract: best-of-30 reversed two": "comparing the best configuration of each method reversed two of them" if len(reversed_best) == 2 else None,
    "abstract: more than one game": "the only extension that improved on DQN in more than one game" if [a for a in ARMS[1:] if sum((g, a) in sig and A[g, a]["pct"] > 0 for g in GAMES) > 1] == ["averaged"] else None,
    "nine significant": "nine of which are significant" if len(sig) == 9 else None,
    "alpha": "0.05/16=0.003125",
    "Averaged wins": f"beat DQN on {', '.join(str(A[g, 'averaged']['wins']) for g in GAMES[1:3])} and {A['spaceinvaders', 'averaged']['wins']} of the 30",
    "Averaged effects": f"Asterix ({pct('asterix', 'averaged')}), Freeway ({pct('freeway', 'averaged')}) and Space Invaders ({pct('spaceinvaders', 'averaged')})",
    "DDQN SI": f"Invaders ({pct('spaceinvaders', 'ddqn')}, {A['spaceinvaders', 'ddqn']['wins']} of 30 individuals)",
    "DDQN Asterix": f"Asterix ({pct('asterix', 'ddqn')}, p_t={fmt_p(A['asterix', 'ddqn']['pt'])})",
    "DDQN Freeway": f"Freeway ({pct('freeway', 'ddqn')}, p_t={fmt_p(A['freeway', 'ddqn']['pt'])})",
    "PER effects": f"Freeway ({pct('freeway', 'per')}) and Space Invaders ({pct('spaceinvaders', 'per')})",
    "PER wins": f"only {A['freeway', 'per']['wins']} and {A['spaceinvaders', 'per']['wins']} individuals",
    "n-step effects": f"Asterix ({pct('asterix', 'nstep')}), Freeway ({pct('freeway', 'nstep')}) and Space Invaders ({pct('spaceinvaders', 'nstep')})",
    "n-step wins": f"with {A['asterix', 'nstep']['wins']}, {A['freeway', 'nstep']['wins']} and {A['spaceinvaders', 'nstep']['wins']} wins",
    "|dz| range": f"|d_z| between {d_sig[0]:.2f} and {d_sig[-1]:.2f}",
    "Breakout ratios": f"ratios lie between {min(R['breakout', a]['gm'] for a in ARMS[1:]):.2f} and {max(R['breakout', a]['gm'] for a in ARMS[1:]):.2f}, but their intervals extend from {min(R['breakout', a]['lo'] for a in ARMS[1:]):.2f} to {max(R['breakout', a]['hi'] for a in ARMS[1:]):.2f}",
    "Breakout last-ten range": f"between {min(curve[a, 'breakout'][-10:].mean() for a in ARMS):.1f} and {max(curve[a, 'breakout'][-10:].mean() for a in ARMS):.1f}",
    "Breakout at 0.5M": f"{curve['nstep', 'breakout'][9]:.1f} against {curve['control', 'breakout'][9]:.1f} for DQN at 5\\times10^5",
    "Asterix n-step max": f"never exceeded {curve['nstep', 'asterix'].max():.2f}, whereas DQN ended at {curve['control', 'asterix'][-10:].mean():.1f}",
    "Freeway early": f"({curve['averaged', 'freeway'][9]:.1f} against {curve['control', 'freeway'][9]:.1f} for DQN",
    "Freeway end": f"({curve['averaged', 'freeway'][-10:].mean():.1f} against {curve['control', 'freeway'][-10:].mean():.1f})",
    "SI PER start": f"({curve['per', 'spaceinvaders'][0]:.1f} against {curve['control', 'spaceinvaders'][0]:.1f} at the first",
    "SI PER dip": f"fell to {curve['per', 'spaceinvaders'][:20].min():.1f} by 10^6" if int(np.argmin(curve['per', 'spaceinvaders'])) == 19 else None,
    "SI PER end": f"({curve['per', 'spaceinvaders'][-10:].mean():.1f} against {curve['control', 'spaceinvaders'][-10:].mean():.1f})",
    "DQN CV range": f"from {min(cv.values()):.0f}\\% (Breakout) to {max(cv.values()):.0f}\\% (Asterix)" if min(cv, key=cv.get) == "breakout" and max(cv, key=cv.get) == "asterix" else None,
    "lr ratio range": f"{min(lr_ratio.values()):.1f} to {max(lr_ratio.values()):.1f} times",
    "lr means": "(" + ", ".join(f"{names[g]} {by_lr['control', g][0]:.1f} against {by_lr['control', g][2]:.1f}" for g in GAMES[:3]) + f", Space Invaders {by_lr['control', 'spaceinvaders'][0]:.1f} against {by_lr['control', 'spaceinvaders'][2]:.1f})",
    "15 of 16, exception": "exception was n-step DQN in Freeway" if exceeds == [("freeway", "nstep")] else None,
    "best lr exceptions": "except n-step DQN in Asterix and Freeway" if best_lr_exceptions == [("nstep", "asterix"), ("nstep", "freeway")] else None,
    "n-step Freeway lr": f"its mean AUC there was {by_lr['nstep', 'freeway'][0]:.1f}, against {by_lr['control', 'freeway'][0]:.1f} for DQN, whereas at the two larger rates it reached {by_lr['nstep', 'freeway'][1]:.1f} and {by_lr['nstep', 'freeway'][2]:.1f}" if int(np.argmax(by_lr['nstep', 'freeway'])) != 0 and all(int(np.argmax(by_lr[a, 'freeway'])) == 0 for a in ARMS if a != 'nstep') else None,
    "n-step Freeway ratios": f"ratios of means to DQN of {nstep_r['freeway'][1]:.2f} and {nstep_r['freeway'][2]:.2f}",
    "n-step Asterix flat": f"its mean AUC was {by_lr['nstep', 'asterix'][0]:.1f}, {by_lr['nstep', 'asterix'][1]:.1f} and {by_lr['nstep', 'asterix'][2]:.1f} from the lowest rate to the highest, and its ratio of mean AUC to that of DQN rose from {nstep_r['asterix'][0]:.2f} to {nstep_r['asterix'][2]:.2f}" if max(by_lr['nstep', 'asterix']) < 5 else None,
    "PER ratios": f"between {min(per_r):.2f} and {max(per_r):.2f}" if max(per_r) < 1 else None,
    "per-game threshold same nine": "the same nine comparisons are significant" if {k for k, r in A.items() if r["pt"] <= .0125} == sig else None,
    "Wilcoxon confirms eight": "confirms eight of them" if sum(A[k]["pw"] <= ALPHA for k in sig) == 8 and A["spaceinvaders", "ddqn"]["pw"] > ALPHA else None,
    "DDQN SI p_W and CI": f"(p_W={A['spaceinvaders', 'ddqn']['pw']:.2g}), whose interval also only just excludes zero ([{A['spaceinvaders', 'ddqn']['lo']:+.1f}\\%, {A['spaceinvaders', 'ddqn']['hi']:+.1f}\\%])",
    "last ten: five of nine": "five of the nine remain significant" if sum(T["last10"][k]["pt"] <= ALPHA for k in sig) == 5 else None,
    "final: only three": "only three do" if sum(T["final"][k]["pt"] <= ALPHA for k in sig) == 3 else None,
    "directions agree": "agree in direction with all nine" if all(np.sign(T[k][key]["pct"]) == np.sign(A[key]["pct"]) for k in ("last10", "final") for key in sig) else None,
    "none become significant": "No comparison that was not significant on the AUC became significant" if not nonsig_become else None,
    "Shapiro 6 of 16": f"for {sum(r['ps'] < .05 for r in A.values())} of the 16 comparisons",
    "probability range": f"probability of {min(30 - A[g, 'averaged']['wins'] for g in GAMES[1:]) / 30:.2f} to {max(30 - A[g, 'averaged']['wins'] for g in GAMES[1:]) / 30:.2f}",
    "Averaged Wilcoxon all three": "the Wilcoxon test confirmed all three" if all(A[g, "averaged"]["pw"] <= ALPHA for g in GAMES[1:]) else None,
    "at least 24": f"at least {min(A[g, 'averaged']['wins'] for g in GAMES[1:])} of 30",
    "600 runs": "4\\times5\\times30=600",
    # ---- log-ratio analysis, Holm
    "ratio: seven of nine": "seven of the nine findings remain significant" if len(ratio_sig) == 7 else None,
    "ratio: exceptions are gains": "The two exceptions are gains" if all(A[k]["pct"] > 0 for k in ratio_lost) else None,
    "ratio: Averaged Asterix": f"Averaged DQN in Asterix has a geometric-mean ratio of {R['asterix', 'averaged']['gm']:.2f} [{R['asterix', 'averaged']['lo']:.2f}, {R['asterix', 'averaged']['hi']:.2f}] (p_r={R['asterix', 'averaged']['pr']:.2g})",
    "ratio: two collapses": f"beat DQN on {A['asterix', 'averaged']['wins']} individuals, but on two others it reached only {R['asterix', 'averaged']['two_lowest'][0]:.2f} and {R['asterix', 'averaged']['two_lowest'][1]:.2f}",
    "ratio: Wilcoxon on log": f"still rejects (p={R['asterix', 'averaged']['pw']:.2g})" if R['asterix', 'averaged']['pw'] <= ALPHA else None,
    "ratio: DDQN SI": f"Double DQN in Space Invaders has a ratio of {R['spaceinvaders', 'ddqn']['gm']:.2f} [{R['spaceinvaders', 'ddqn']['lo']:.2f}, {R['spaceinvaders', 'ddqn']['hi']:.2f}] (p_r={R['spaceinvaders', 'ddqn']['pr']:.2g})",
    "ratio: Shapiro 9 of 16": f"rejected normality for {sum(r['ps'] < .05 for r in R.values())} of the 16 comparisons",
    "Holm": f"rejects the same nine hypotheses: the tenth-smallest p-value, {holm_p[9]:.2g}, exceeds 0.05/7" if holm_k == 9 and holm_p[9] > .05 / 7 else None,
    # ---- regression and learning rate
    "lr adjusted DQN range": f"multiplied the AUC of DQN by {min(REG[g, 'control']['ratio'] for g in GAMES):.2f} (Asterix) to {max(REG[g, 'control']['ratio'] for g in GAMES):.2f} (Freeway)" if min(GAMES, key=lambda g: REG[g, 'control']['ratio']) == 'asterix' and max(GAMES, key=lambda g: REG[g, 'control']['ratio']) == 'freeway' else None,
    "lr partial R2 range": f"explained {100 * min(REG[g, 'control']['partial'] for g in GAMES):.0f}\\% to {100 * max(REG[g, 'control']['partial'] for g in GAMES):.0f}\\% of the variance",
    "lr exceeds 15 of 16": "larger than the geometric-mean treatment effect in 15 of the 16 comparisons; the exception was n-step DQN in Freeway" if exceeds == [("freeway", "nstep")] else None,
    "n-step Freeway adjusted": f"its AUC at 10^-3 was {REG['freeway', 'nstep']['ratio']:.2f} [{REG['freeway', 'nstep']['lo']:.2f}, {REG['freeway', 'nstep']['hi']:.2f}] times that at 10^-4, whereas the corresponding factor was below one for DQN, Double DQN and Averaged DQN" if all(REG['freeway', a]['ratio'] < 1 for a in ('control', 'ddqn', 'averaged')) else None,
    "PER at 1e-4": f"at 10^-4 the ratios were {by_lr['per', 'freeway'][0] / by_lr['control', 'freeway'][0]:.2f} and {by_lr['per', 'spaceinvaders'][0] / by_lr['control', 'spaceinvaders'][0]:.2f}",
    "PER at 1e-4 percent": f"PER remained {100 * (1 - by_lr['per', 'freeway'][0] / by_lr['control', 'freeway'][0]):.0f}\\% and {100 * (1 - by_lr['per', 'spaceinvaders'][0] / by_lr['control', 'spaceinvaders'][0]):.0f}\\% below DQN at 10^-4",
    "Averaged by period n": "(8, 11 and 11 individuals)" if [int((period['freeway'] == v).sum()) for v in (500, 1000, 2000)] == [8, 11, 11] else None,
    "Averaged by period": "geometric-mean ratio to DQN was " + " in Freeway, ".join([", ".join(f"{x:.2f}" for x in avg_period['freeway'][:2]) + f" and {avg_period['freeway'][2]:.2f}"]) + " in Freeway, " + ", ".join(f"{x:.2f}" for x in avg_period['asterix'][:2]) + f" and {avg_period['asterix'][2]:.2f} in Asterix and " + ", ".join(f"{x:.2f}" for x in avg_period['spaceinvaders'][:2]) + f" and {avg_period['spaceinvaders'][2]:.2f} in Space Invaders",
    "DQN period effect": f"changed the AUC of DQN itself by a factor of {min(REG[g, 'control']['period_2000_vs_500'] for g in GAMES):.2f} to {max(REG[g, 'control']['period_2000_vs_500'] for g in GAMES):.2f}",
    "23 of 30 too high": "23 of the 30 configurations" if int((lr_col['breakout'] > 1e-4).sum()) == 23 else None,
    # ---- reliability of smaller evaluations
    "PoI Averaged": f"Averaged DQN was {A['asterix', 'averaged']['wins'] / 30:.2f}, {A['freeway', 'averaged']['wins'] / 30:.2f} and {A['spaceinvaders', 'averaged']['wins'] / 30:.2f} in Asterix, Freeway and Space Invaders",
    "sign agreement 3/5": f"in {100 * sign_range[3]:.0f}\\% to 100\\% of the subsets of three individuals and {100 * sign_range[5]:.0f}\\% to 100\\% of those of five" if all(max(SUB[key][k]['sign'] for key in sig) == 1 for k in (3, 5)) else None,
    "detection ranges": f"detected the effects in {100 * sub_range[3][0]:.0f}\\% to {100 * sub_range[3][1]:.0f}\\% of the subsets of three, {100 * sub_range[5][0]:.0f}\\% to {100 * sub_range[5][1]:.0f}\\% of those of five and {100 * sub_range[10][0]:.0f}\\% to {100 * sub_range[10][1]:.0f}\\% of those of ten",
    "two weakest": "Double DQN in Space Invaders and Averaged DQN in Asterix, in fewer than half of the subsets of ten" if set(weakest) == {("spaceinvaders", "ddqn"), ("asterix", "averaged")} and all(SUB[k][10]["same"] < .5 for k in weakest) and sorted(SUB[k][10]["same"] for k in sig)[2] >= .5 else None,
    "wrong direction": f"at most {100 * max(SUB[key][k]['opposite'] for key in sig for k in (3, 5, 10)):.1f}\\% of subsets",
    "best-of-30 reversals": f"n-step DQN was {best['spaceinvaders', 'nstep']:.1f}\\% better than DQN in Space Invaders and PER {best['freeway', 'per']:.1f}\\% better in Freeway" if sorted(reversed_best) == [("freeway", "per"), ("spaceinvaders", "nstep")] else None,
    "RQ2: three or five": "detected only three or five" if sorted(sum(T[k][key]["pt"] <= ALPHA for key in sig) for k in ("last10", "final")) == [3, 5] else None,
    # ---- density figure, Seaquest, compute environment
    "n-step Asterix at most 5": f"in Asterix, {int((endpoint('nstep', 'asterix', 'auc') / 1e6 <= 5).sum())} of its 30 individuals had an AUC of at most 5",
    "n-step Freeway lowest rate": f"its seven individuals at the lowest learning rate all had an AUC between {fw_lo_n.min():.1f} and {fw_lo_n.max():.1f}, against {fw_lo_c.min():.1f} to {fw_lo_c.max():.1f} for DQN at that rate" if len(fw_lo_n) == 7 and fw_lo_n.max() < fw_lo_c.min() else None,
    "software versions": "(version 2.13) with the Gymnasium \\citep{towers2024gymnasium} interface to MinAtar (version 1.0.15)" if versions == {("2.13.0", "1.0.15")} else None,
    "Seaquest four arms": "in each of the four arms in which it was run" if seaquest_aborted == ["averaged", "control", "ddqn", "per"] else None,
    "no aborted runs": "No run in the four reported games was aborted" if not aborted and complete_curves else None,
    "3e9 frames": "3\\times10^9 frames" if 600 * 5_000_000 == 3_000_000_000 else None,
}
for what, s in quotes.items():
    check(s is not None and s.replace("10^-4", "10^{-4}").replace("10^-3", "10^{-3}").replace("10^6", "10^{6}").replace("10^5", "10^{5}").replace("10^9", "10^{9}") in text
          or (s is not None and s in text), f"text: {what}" + ("" if s is not None else " (claim not supported by data)"))

# ------------------------------------------------------- language and build
american = r"\b\w*(?:iz(?:e|es|ed|ing|ation|ations)|yz(?:e|ed|es|ing)|ization)\b|\b(?:behavior|color|favor|center|modeling|labeled|gray|artifact)\w*"
body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", tex, flags=re.S)  # drawing code, not prose
body = re.sub(r"\\(?:cite[a-z]*|ref|label|eqref|input|url|bibliography\w*|includegraphics|begin|end)\{[^}]*\}", "", body)
body = re.sub(r"\\[A-Za-z]+", "", body)  # LaTeX command names are not prose
hits = sorted({m.group(0) for m in re.finditer(american, body, re.I)} - {"size", "Size", "sizes"})
check(not hits, f"British spelling (American forms found: {hits})")
check(not re.search(r"TODO|TBD|XXX|Conference Paper Title|Given Name Surname", tex), "no placeholders or template text")
log = (REPORT / "main.log").read_text(errors="ignore") if (REPORT / "main.log").exists() else ""
check(log and not re.search(r"undefined|Citation .* undefined|There were undefined", log), "LaTeX log: no undefined references")
bib = (REPORT / "refs.bib").read_text()
keys = set(re.findall(r"@\w+\{([^,]+),", bib)) - {"IEEEexample:BSTcontrol"}
cited = {k.strip() for c in re.findall(r"\\cite[a-z]*\{([^}]*)\}", tex) for k in c.split(",")}
check(cited <= keys, f"every cited key is in refs.bib (missing: {sorted(cited - keys)})")
check(keys <= cited, f"every refs.bib entry is cited (unused: {sorted(keys - cited)})")

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
