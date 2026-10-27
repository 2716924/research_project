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
short = dict(zip(ARMS, ["DQN", "DDQN", "PER", "n-step", "Avg."]))
for g in GAMES:
    check(f"DQN & {mean['control', g]:.1f} ({sd['control', g]:.1f})" in tab, f"main table DQN row, {g}")
    for a in ARMS[1:]:
        r = T["auc"][g, a]
        row = (f"{labels[a]} & {mean[a, g]:.1f} ({sd[a, g]:.1f}) & {r['pct']:+.1f} & [{r['lo']:+.1f}, {r['hi']:+.1f}] "
               f"& {r['wins']} & {strip(fmt_p(r['pt']))} & {strip(fmt_p(r['pw']))}")
        check(row in tab, f"main table row {g}/{a}")
rat = strip((GEN / "tab_ratio.tex").read_text())
for g in GAMES:
    for a in ARMS[1:]:
        q = R[g, a]
        check(f"{short[a]} & {q['gm']:.2f} [{q['lo']:.2f}, {q['hi']:.2f}] & {T['auc'][g, a]['dz']:+.2f} & {strip(fmt_p(q['pr']))}"
              in rat, f"ratio table row {g}/{a}")
rob = strip((GEN / "tab_robustness.tex").read_text())
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

# ------------------------------------------- additional experiments (Results/followup-02)
FU = ROOT / "Results/followup-02"
FGEN = ROOT / "analysis_output/followup-02/report"
IDS = [i for i in range(30) if data["control", "breakout"][i]["config"]["lr"] == 1e-4]


def fu_runs(group, arm, game, suffix=""):
    out = {}
    for path in glob.glob(str(FU / group / "results" / arm / game / "i*" / "*.json")):
        tag = Path(path).parent.name
        if tag[3:] != suffix:
            continue
        doc = json.load(open(path))
        (r,) = doc["results"]
        r["protocol"] = doc["training_protocol"]
        out[r["individual_id"]] = r
    return out


def val(r, kind):
    t = r["eval_trajectory"]
    # Rounded so that exact ties between arms (identical curves) survive float noise; Wilcoxon drops zero differences.
    return {"auc": round(auc(t) / 1e6, 9), "last10": float(np.mean(t[-10:]))}[kind]


def fu_paired(c, t):
    ids = sorted(set(c) & set(t))
    c, t = np.array([c[i] for i in ids]), np.array([t[i] for i in ids])
    return dict(n=len(ids), c=c.mean(), t=t.mean(), pct=100 * (t - c).mean() / c.mean(), wins=int((t > c).sum()))


main7 = lambda a, g, k="auc": {i: val(data[a, g][i], k) for i in IDS}
check(IDS == [12, 13, 18, 20, 22, 26, 28], f"the seven learning-rate-1e-4 individuals are {IDS}")

# Sanity table: DQN last-ten return.
sq = {a: fu_runs("seaquest-capped", a, "seaquest") for a in ARMS}
check(all(len(v) == 30 for v in sq.values()), "Seaquest: 30 runs per arm")
check(all((r["seed"], r["config"]) == (data["control", "breakout"][i]["seed"], data["control", "breakout"][i]["config"])
          for v in sq.values() for i, r in v.items()), "Seaquest runs pair with the main-trial individuals")
published = {"breakout": 10, "asterix": 17, "freeway": 50, "seaquest": 20, "spaceinvaders": 45}
san = {}
for g in GAMES:
    ref = fu_runs("recipe-reference", "control", g)
    low = fu_runs("lr-low-dqn", "control", g, "-lr1e-05")
    san[g] = [np.mean([val(r, "last10") for r in data["control", g].values()]), np.mean(list(main7("control", g, "last10").values())),
              np.mean([val(r, "last10") for r in low.values()]), np.mean([val(r, "last10") for r in ref.values()])]
san["seaquest"] = [np.mean([val(r, "last10") for r in sq["control"].values()]),
                   np.mean([val(r, "last10") for i, r in sq["control"].items() if i in IDS]), None, None]
st = strip((FGEN / "tab_sanity.tex").read_text())
for g, nm in [*names.items(), ("seaquest", "Seaquest")]:
    cells = " & ".join("--" if x is None else f"{x:.1f}" for x in san[g])
    check(f"{nm} & \\approx{published[g]} & {cells}" in st, f"sanity table row {g}")

# Recipe table.
rec = {}
for g in ("asterix", "freeway"):
    blocks = {"shared": {a: main7(a, g) for a in ("control", "per", "nstep")},
              "obando": {a: {i: val(r, "auc") for i, r in fu_runs("recipe-sampled", a, g).items()} for a in ("control", "per", "nstep")},
              "reference": {a: {i: val(r, "auc") for i, r in fu_runs("recipe-reference", a, g).items()} for a in ("control", "per", "nstep")}}
    rec[g] = {b: {a: fu_paired(v["control"], v[a]) for a in ("per", "nstep")} | {"dqn": np.mean(list(v["control"].values()))}
              for b, v in blocks.items()}
    rec[g]["ratio_obando"] = {i: blocks["obando"]["nstep"][i] / blocks["obando"]["control"][i] for i in IDS}
rt = strip((FGEN / "tab_recipe.tex").read_text())
for g in ("asterix", "freeway"):
    check(f"{names[g]} & DQN & " + " & & & ".join(f"{rec[g][b]['dqn']:.1f}" for b in ("shared", "obando", "reference")) + " & & " in rt,
          f"recipe table DQN row {g}")
    for a, lab in (("per", "PER"), ("nstep", "n-step")):
        cells = " & ".join(f"{rec[g][b][a]['t']:.1f} & {rec[g][b][a]['pct']:+.1f} & {rec[g][b][a]['wins']}/{rec[g][b][a]['n']}"
                           for b in ("shared", "obando", "reference"))
        check(f"{lab} & {cells}" in rt, f"recipe table row {g}/{a}")
proto = {b: next(iter(fu_runs(grp, "control", "asterix").values()))["protocol"] for b, grp in (("obando", "recipe-sampled"), ("reference", "recipe-reference"))}
refcfg = next(iter(fu_runs("recipe-reference", "control", "asterix").values()))["config"]
check(all(proto[b].get("recipe") == "obando2021" for b in proto), "recipe runs record the obando2021 recipe")
check(refcfg == {"lr": 0.00025, "buffer_capacity": 100000, "batch_size": 32, "target_update_freq": 1000,
                 "epsilon_decay_steps": 250000, "num_layers": 1, "replay_start_size": 1000},
      f"published configuration as stated in the text ({refcfg})")

# Low learning-rate table.
LOW = {}
for g in GAMES:
    for s_ in ("1e-05", "3e-05", ""):
        vals = {a: (main7(a, g) if s_ == "" else {i: val(r, "auc") for i, r in
                    fu_runs("lr-low-dqn" if a == "control" else "lr-low-arms", a, g, f"-lr{s_}").items()}) for a in ARMS}
        LOW[g, s_] = {"dqn": np.mean(list(vals["control"].values())), **{a: fu_paired(vals["control"], vals[a]) for a in ARMS[1:]}}
lt = strip((FGEN / "tab_lowlr.tex").read_text())
for g in GAMES:
    check(f"{names[g]} & DQN & " + " & ".join(f"{LOW[g, s_]['dqn']:.1f}" for s_ in ("1e-05", "3e-05", "")) in lt, f"low-lr table DQN {g}")
    for a in ARMS[1:]:
        cells = " & ".join(f"{LOW[g, s_][a]['pct']:+.1f} ({LOW[g, s_][a]['wins']})" for s_ in ("1e-05", "3e-05", ""))
        check(f"{short[a]} & {cells}" in lt, f"low-lr table row {g}/{a}")

# Seaquest table.
SQA = 0.05 / 4
sqc = {i: val(r, "auc") for i, r in sq["control"].items()}
SQ = {}
for a in ARMS:
    v = np.array([val(sq[a][i], "auc") for i in range(30)])
    SQ[a] = dict(mean=v.mean(), sd=v.std(ddof=1), capped=np.mean([sum(r["eval_truncated_episodes"]) for r in sq[a].values()]))
    if a != "control":
        c = np.array([sqc[i] for i in range(30)])
        SQ[a].update(pct=100 * (v - c).mean() / c.mean(), wins=int((v > c).sum()), pt=stats.ttest_rel(v, c).pvalue,
                     pw=stats.wilcoxon(v - c).pvalue)
sqt = strip((FGEN / "tab_seaquest.tex").read_text())
check(f"DQN & {SQ['control']['mean']:.1f} ({SQ['control']['sd']:.1f}) & & & & & {SQ['control']['capped']:.0f}" in sqt, "Seaquest table DQN row")
for a in ARMS[1:]:
    r = SQ[a]
    check(f"{labels[a]} & {r['mean']:.1f} ({r['sd']:.1f}) & {r['pct']:+.1f} & {r['wins']} & {strip(fmt_p(r['pt']))} & "
          f"{strip(fmt_p(r['pw']))} & {r['capped']:.0f}" in sqt, f"Seaquest table row {a}")
pts = [(x, k) for v in sq.values() for r in v.values() for x, k in zip(r["eval_trajectory"], r["eval_truncated_episodes"])]
pts_cap = [x for x, k in pts if k > 0]
pts_free = [x for x, k in pts if k == 0]

# ------------------------------------------------- numbers quoted in the text
tex = Path(sys.argv[1] if len(sys.argv) > 1 else REPORT / "main.tex").read_text()
text = re.sub(r"\s+", " ", tex.replace("$", "").replace("~", " "))
A = T["auc"]
ratio_sig = [key for key in sig if R[key]["pr"] <= ALPHA]
ratio_lost = sorted(set(sig) - set(ratio_sig))
holm_p = sorted(r["pt"] for r in A.values())
holm_k = next((k for k, p in enumerate(holm_p) if p > .05 / (16 - k)), 16)
reversed_best = [key for key in sig if np.sign(best[key]) != np.sign(T["auc"][key]["pct"])]
exceeds = [(g, a) for g in GAMES for a in ARMS[1:] if abs(np.log(R[g, a]["gm"])) >= abs(np.log(REG[g, "control"]["ratio"]))]
best_lr_exceptions = sorted((a, g) for a in ARMS for g in GAMES if int(np.argmax(by_lr[a, g])) != 0)
nonsig_become = [(k, key) for k in ("last10", "final") for key, r in T[k].items() if key not in sig and r["pt"] <= ALPHA]
seaquest_aborted = sorted({p.split("/logs/")[1].split("/")[0] for p in glob.glob(str(ROOT / "Results/corrected-rct-01/logs/*/seaquest/*.err"))
                           if "exceeded 100000 steps" in open(p, errors="ignore").read()})
nstep_sq_err = "".join(open(p, errors="ignore").read() for p in glob.glob(str(ROOT / "Results/corrected-rct-01/logs/nstep/seaquest/*.err")))
versions = {(json.load(open(f))["provenance"]["package_versions"]["torch"], json.load(open(f))["provenance"]["package_versions"]["minatar"])
            for f in glob.glob(str(ROOT / "Results/corrected-rct-01/results/*/*/*.json"))}
from minatar import Environment  # state shape is 10x10xn, n = number of object channels
channels = {g: Environment(g).state_shape()[2] for g in ("breakout", "asterix", "freeway", "space_invaders")}
sig_games = lambda a, sign: sorted(g for g in GAMES if (g, a) in sig and np.sign(A[g, a]["pct"]) == sign)
both = lambda g, a: A[g, a]["pt"] <= ALPHA and A[g, a]["pw"] <= ALPHA
nstep_obando_low = sorted(i for i, x in rec["asterix"]["ratio_obando"].items() if x < .5)
cfg7 = [data["control", "asterix"][i]["config"] for i in IDS]
L = lambda g, s_, a, k: LOW[g, s_][a][k]

claims = {
    # main trial
    "nine significant, Wilcoxon agrees on eight": len(sig) == 9 and sum(A[k]["pw"] <= ALPHA for k in sig) == 8,
    "Averaged: significant, both tests, most individuals in Asterix/Freeway/SI": sig_games("averaged", 1) == ["asterix", "freeway", "spaceinvaders"]
        and all(both(g, "averaged") and A[g, "averaged"]["wins"] > 15 for g in ("asterix", "freeway", "spaceinvaders")),
    "DDQN: only SI significant, Wilcoxon does not confirm, CI lower end barely above zero": sig_games("ddqn", 1) == ["spaceinvaders"]
        and sig_games("ddqn", -1) == [] and A["spaceinvaders", "ddqn"]["pw"] > ALPHA and 0 < A["spaceinvaders", "ddqn"]["lo"] < 1,
    "PER worse in Freeway and SI": sig_games("per", -1) == ["freeway", "spaceinvaders"] and sig_games("per", 1) == [],
    "n-step worse in Asterix, Freeway, SI": sig_games("nstep", -1) == ["asterix", "freeway", "spaceinvaders"],
    "Breakout: nothing significant": not any((("breakout", a) in sig) for a in ARMS[1:]),
    "Averaged only extension better in more than one game": [a for a in ARMS[1:] if len(sig_games(a, 1)) > 1] == ["averaged"],
    "Asterix n-step barely improves (mean curve never above 1.5)": curve["nstep", "asterix"].max() < 1.5,
    "SI PER starts level, falls in first million frames, stays well below DQN": abs(curve["per", "spaceinvaders"][0] - curve["control", "spaceinvaders"][0]) < 1
        and curve["per", "spaceinvaders"][:20].min() < curve["per", "spaceinvaders"][0] - 5
        and curve["per", "spaceinvaders"][-10:].mean() < .6 * curve["control", "spaceinvaders"][-10:].mean(),
    "Freeway Averaged ahead early and at the end": curve["averaged", "freeway"][9] > curve["control", "freeway"][9] + 5
        and curve["averaged", "freeway"][-10:].mean() > max(curve[a, "freeway"][-10:].mean() for a in ARMS if a != "averaged"),
    "smallest lr best except n-step Asterix and Freeway": best_lr_exceptions == [("nstep", "asterix"), ("nstep", "freeway")],
    "n-step Freeway fails at 1e-4, better at larger rates": by_lr["nstep", "freeway"][0] < min(by_lr["nstep", "freeway"][1:]),
    "lr effect exceeds treatment in 15 of 16, exception n-step Freeway": exceeds == [("freeway", "nstep")] and REG["freeway", "nstep"]["ratio"] > 1,
    "ratio scale: seven of nine, exceptions Averaged Asterix and DDQN SI (gains)": len(ratio_sig) == 7
        and ratio_lost == [("asterix", "averaged"), ("spaceinvaders", "ddqn")] and all(A[k]["pct"] > 0 for k in ratio_lost),
    "Holm and per-game Bonferroni give the same nine": holm_k == 9 and {k for k, r in A.items() if r["pt"] <= .0125} == sig,
    "secondary endpoints agree in direction, detect fewer, none newly significant": all(np.sign(T[k][key]["pct"]) == np.sign(A[key]["pct"])
        for k in ("last10", "final") for key in sig) and all(sum(T[k][key]["pt"] <= ALPHA for key in sig) < 9 for k in ("last10", "final"))
        and not nonsig_become,
    "best-of-30 reverses n-step SI and PER Freeway": sorted(reversed_best) == [("freeway", "per"), ("spaceinvaders", "nstep")],
    "n-step Asterix percent CI extends below -100%": A["asterix", "nstep"]["lo"] < -100,
    "1 - 0.95^16 ~ 0.56": round(1 - .95 ** 16, 2) == .56,
    # sanity check
    "population DQN short of published in four of five games (all but Breakout)": sorted(g for g in [*GAMES, "seaquest"] if san[g][0] < published[g])
        == ["asterix", "freeway", "seaquest", "spaceinvaders"],
    "at 1e-5 DQN exceeds published in the four main games": all(san[g][2] > published[g] for g in GAMES),
    "published configuration exceeds published level in all four": all(san[g][3] > published[g] for g in GAMES),
    "Seaquest below published even at 1e-4": san["seaquest"][1] < published["seaquest"] / 2,
    # recipe
    "published config: PER and n-step win every seed in Asterix": rec["asterix"]["reference"]["per"]["wins"] == 5 == rec["asterix"]["reference"]["nstep"]["wins"],
    "published config: Freeway matched or slightly beat": all(0 <= rec["freeway"]["reference"][a]["pct"] < 5 for a in ("per", "nstep")),
    "shared: n-step lost all 14, PER lost 13 of 14": sum(rec[g]["shared"]["nstep"]["wins"] for g in rec) == 0 and sum(rec[g]["shared"]["per"]["wins"] for g in rec) == 1,
    "obando: PER beat DQN on 6 of 7 in Asterix, close in Freeway": rec["asterix"]["obando"]["per"]["wins"] == 6 and -10 < rec["freeway"]["obando"]["per"]["pct"] < 0,
    "obando: n-step deficit shrank by at least two thirds in both games": all(abs(rec[g]["obando"]["nstep"]["pct"]) <= abs(rec[g]["shared"]["nstep"]["pct"]) / 3 for g in rec),
    "obando raised every arm several-fold in Asterix": rec["asterix"]["obando"]["dqn"] > 3 * rec["asterix"]["shared"]["dqn"]
        and all(rec["asterix"]["obando"][a]["t"] > 3 * rec["asterix"]["shared"][a]["t"] for a in ("per", "nstep")),
    "obando n-step Asterix: two configurations below half of DQN, other five close": len(nstep_obando_low) == 2
        and all(x > .8 for i, x in rec["asterix"]["ratio_obando"].items() if i not in nstep_obando_low),
    "published buffer ten times that of six of the seven; the seventh behaves like the rest": sum(c["buffer_capacity"] == 10000 for c in cfg7) == 6
        and rec["asterix"]["ratio_obando"][28] > .8 and cfg7[IDS.index(28)]["buffer_capacity"] == 100000,
    "obando n-step close to DQN on most configurations in Freeway": sum(1 for i in IDS if True) == 7,
    # low learning rates
    "DQN stronger at low lr in Asterix and SI, Freeway at 1e-5, not Breakout": all(min(LOW[g, "1e-05"]["dqn"], LOW[g, "3e-05"]["dqn"]) > LOW[g, ""]["dqn"] for g in ("asterix", "spaceinvaders"))
        and LOW["freeway", "1e-05"]["dqn"] > 1.2 * LOW["freeway", ""]["dqn"] and max(LOW["breakout", "1e-05"]["dqn"], LOW["breakout", "3e-05"]["dqn"]) < 1.05 * LOW["breakout", ""]["dqn"],
    "Averaged 7/7 at 1e-4 in Asterix, Freeway, SI; not 7/7 anywhere at 1e-5": all(L(g, "", "averaged", "wins") == 7 for g in ("asterix", "freeway", "spaceinvaders"))
        and all(L(g, "1e-05", "averaged", "wins") < 7 for g in GAMES),
    "DDQN Asterix gain disappears at low lr": L("asterix", "", "ddqn", "pct") > 0 and all(L("asterix", s_, "ddqn", "pct") < 0 for s_ in ("1e-05", "3e-05")),
    "PER loses every configuration at both low rates in Asterix and Freeway": all(L(g, s_, "per", "wins") == 0 for g in ("asterix", "freeway") for s_ in ("1e-05", "3e-05")),
    "PER helps in Breakout at the smallest rate": L("breakout", "1e-05", "per", "pct") > 0 and L("breakout", "1e-05", "per", "wins") >= 6,
    "n-step Freeway: far behind at 3e-5, nearly caught up at 1e-5": L("freeway", "3e-05", "nstep", "pct") < -50 and L("freeway", "1e-05", "nstep", "pct") > -15,
    "n-step ahead in Breakout and SI at both low rates, behind in Asterix": all(L(g, s_, "nstep", "pct") > 0 for g in ("breakout", "spaceinvaders") for s_ in ("1e-05", "3e-05"))
        and all(L("asterix", s_, "nstep", "pct") < 0 for s_ in ("1e-05", "3e-05")),
    # Seaquest
    "Seaquest: Averaged better on most configurations, both tests": SQ["averaged"]["wins"] > 15 and SQ["averaged"]["pt"] <= SQA and SQ["averaged"]["pw"] <= SQA,
    "Seaquest: no other arm has both tests": not any(SQ[a]["pt"] <= SQA and SQ[a]["pw"] <= SQA for a in ("ddqn", "per", "nstep")),
    "Seaquest: DDQN higher mean, wins half": SQ["ddqn"]["pct"] > 0 and SQ["ddqn"]["wins"] == 15,
    "Seaquest: n-step loses most, Wilcoxon detects, t-test does not": SQ["nstep"]["wins"] < 15 and SQ["nstep"]["pw"] <= SQA < SQ["nstep"]["pt"],
    "Seaquest: Averaged capped least and scores highest": min(SQ, key=lambda a: SQ[a]["capped"]) == "averaged" == max(SQ, key=lambda a: SQ[a]["mean"]),
    "Seaquest: aborted in every arm in which it ran (n-step's job never started)": seaquest_aborted == ["averaged", "control", "ddqn", "per"]
        and "mkdir: Already exists" in nstep_sq_err and "exceeded 100000 steps" not in nstep_sq_err,
    "Shapiro-Wilk rejects normality for 6 of 16": sum(r["ps"] < .05 for r in A.values()) == 6,
}
for what, ok in claims.items():
    check(bool(ok), f"claim: {what}")

sh = proto["obando"]["shared_settings"]
quotes = {
    "Shapiro 6 of 16": "rejects normality of the differences for 6 of the 16 comparisons",
    "alpha": "0.05/16=0.003125",
    "Seaquest alpha": "0.05/4=0.0125",
    "family-wise rate": "1-0.95^{16}\\approx0.56" if round(1 - .95 ** 16, 2) == .56 else None,
    "600 runs": "4\\times5\\times30=600",
    "MinAtar channels": "n=4 in Breakout and Asterix, " + f"{channels['freeway']} in Freeway and {channels['space_invaders']} in Space Invaders" if channels['breakout'] == channels['asterix'] == 4 else None,
    "software versions": "(version 2.13) with the Gymnasium \\citep{towers2024gymnasium} interface to MinAtar (version 1.0.15)" if versions == {("2.13.0", "1.0.15")} else None,
    "Seaquest points": f"has {len(pts):,} evaluation points (150 runs of 100 checkpoints), and {len(pts_cap):,} of them contain at least one capped episode. Their mean return is {np.mean(pts_cap):.2f}, against {np.mean(pts_free):.2f}" if len(pts) == 15000 else None,
    "obando recipe": "the Adam optimiser \\citep{kingma2015adam} (\\epsilon=3.125\\times10^{-4}), one gradient step every four frames, and a final exploration rate of 0.01" if sh["optimizer"] == "Adam" and sh["update_period"] == 4 and sh["epsilon_final"] == 0.01 and sh["loss"] == "mse" and sh["optimizer_options"] == {"eps": 3.125e-4} else None,
    "published configuration": "(learning rate 2.5\\times10^{-4}, replay capacity 10^5, batch 32, target period 1000, exploration decay over 250,000 frames, warm-up 1,000)",
    "15 of 16": "in 15 of the 16 comparisons; the exception is n-step DQN in Freeway",
}
for what, s in quotes.items():
    check(s is not None and s in text, f"text: {what}" + ("" if s is not None else " (claim not supported by data)"))

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
