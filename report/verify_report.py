"""Independent check of every number in the report against the raw result JSON.

Deliberately imports nothing from analysis/ or rct/: it reassembles the dataset from
Results/, recomputes all statistics, compares each cell of the generated tables and
asserts that every number quoted in main.tex appears there. It also scans main.tex for
American spellings, leftover template text, and unresolved references in the LaTeX log.

    .venv/bin/python report/verify_report.py [path/to/main.tex]
"""
import glob
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


T = {k: {(g, a): paired(endpoint("control", g, k), endpoint(a, g, k)) for g in GAMES for a in ARMS[1:]}
     for k in ("auc", "last10", "final")}
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
        r = T["auc"][g, a]
        row = (f"{labels[a]} & {mean[a, g]:.1f} ({sd[a, g]:.1f}) & {r['pct']:+.1f} & [{r['lo']:+.1f}, {r['hi']:+.1f}] "
               f"& {r['dz']:+.2f} & {strip(fmt_p(r['pt']))} & {strip(fmt_p(r['pw']))} & {r['wins']}/30")
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

# ------------------------------------------------- numbers quoted in the text
tex = Path(sys.argv[1] if len(sys.argv) > 1 else REPORT / "main.tex").read_text()
text = re.sub(r"\s+", " ", tex.replace("$", "").replace("~", " "))
A = T["auc"]
r1 = lambda x: f"{x:.1f}"
pct = lambda g, a, k="auc": f"{T[k][g, a]['pct']:+.1f}\\%"
lr_ratio = {g: by_lr["control", g][0] / by_lr["control", g][2] for g in GAMES}
spread = {g: max(by_lr["control", g]) / min(by_lr["control", g]) for g in GAMES}
treat_ratio = {(g, a): max(mean[a, g] / mean["control", g], mean["control", g] / mean[a, g]) for g in GAMES for a in ARMS[1:]}
exceeds = [(g, a) for (g, a), v in treat_ratio.items() if v >= spread[g]]
nstep_r = {g: [by_lr["nstep", g][k] / by_lr["control", g][k] for k in range(3)] for g in GAMES}
per_r = [by_lr["per", g][k] / by_lr["control", g][k] for g in ("freeway", "spaceinvaders") for k in range(3)]
nonsig_become = [(k, key) for k in ("last10", "final") for key, r in T[k].items() if key not in sig and r["pt"] <= ALPHA]
per_fix = {}
for g in GAMES:
    old = np.array([auc(r["eval_trajectory"]) for _, r in sorted(runs("original-run-2026-09", "per", g).items())])
    new = endpoint("per", g, "auc")
    c = endpoint("control", g, "auc")
    per_fix[g] = (100 * (new.mean() - old.mean()) / old.mean(), stats.ttest_rel(new, old).pvalue,
                  100 * (old.mean() - c.mean()) / c.mean(), stats.ttest_rel(old, c).pvalue)
fw17 = paired(endpoint("control", "freeway", "auc")[:18], endpoint("nstep", "freeway", "auc")[:18])
d_sig = sorted(abs(A[k]["dz"]) for k in sig)
cv = {g: 100 * sd["control", g] / mean["control", g] for g in GAMES}
best_lr_exceptions = sorted((a, g) for a in ARMS for g in GAMES if int(np.argmax(by_lr[a, g])) != 0)

quotes = {
    "abstract: 58, 21 and 19 per cent": "by 58, 21 and 19 per cent" if [round(A[g, "averaged"]["pct"]) for g in GAMES[1:]] == [58, 21, 19] else None,
    "abstract: 11 per cent": "by 11 per cent" if round(A["spaceinvaders", "ddqn"]["pct"]) == 11 else None,
    "abstract: up to 80 per cent": "by up to 80 per cent" if round(-min(r["pct"] for r in A.values())) == 80 else None,
    "abstract: fifteen of sixteen": "fifteen of the sixteen" if len(exceeds) == 1 else None,
    "abstract: three of nine (final)": "confirmed three of the nine" if sum(T["final"][k]["pt"] <= ALPHA for k in sig) == 3 else None,
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
    "Breakout point estimates": f"between {min(A['breakout', a]['pct'] for a in ARMS[1:]):+.1f}\\% and {max(A['breakout', a]['pct'] for a in ARMS[1:]):+.1f}\\%",
    "Breakout intervals": f"from {min(A['breakout', a]['lo'] for a in ARMS[1:]):.1f}\\% to {max(A['breakout', a]['hi'] for a in ARMS[1:]):+.1f}\\%",
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
    "n-step Freeway lr": f"at 10^-4 was {by_lr['nstep', 'freeway'][0]:.1f}, against {by_lr['control', 'freeway'][0]:.1f} for DQN, a ratio of {nstep_r['freeway'][0]:.2f}",
    "n-step Freeway ratios": f"ratios of {nstep_r['freeway'][1]:.2f} and {nstep_r['freeway'][2]:.2f}",
    "n-step Asterix ratios": f"({nstep_r['asterix'][0]:.2f}, {nstep_r['asterix'][1]:.2f} and {nstep_r['asterix'][2]:.2f})",
    "PER ratios": f"between {min(per_r):.2f} and {max(per_r):.2f}" if max(per_r) < 1 else None,
    "per-game threshold same nine": "the same nine comparisons are significant" if {k for k, r in A.items() if r["pt"] <= .0125} == sig else None,
    "Wilcoxon confirms eight": "confirms eight of them" if sum(A[k]["pw"] <= ALPHA for k in sig) == 8 and A["spaceinvaders", "ddqn"]["pw"] > ALPHA else None,
    "DDQN SI p_W and CI": f"(p_W={A['spaceinvaders', 'ddqn']['pw']:.2g}), whose interval also only just excludes zero ([{A['spaceinvaders', 'ddqn']['lo']:+.1f}\\%, {A['spaceinvaders', 'ddqn']['hi']:+.1f}\\%])",
    "n-step Freeway 0-17": f"gives {fw17['pct']:.1f}\\% (p_t={fmt_p(fw17['pt'])})",
    "last ten: five of nine": "five of the nine remain significant" if sum(T["last10"][k]["pt"] <= ALPHA for k in sig) == 5 else None,
    "final: only three": "only three do" if sum(T["final"][k]["pt"] <= ALPHA for k in sig) == 3 else None,
    "directions agree": "agree in direction with all nine" if all(np.sign(T[k][key]["pct"]) == np.sign(A[key]["pct"]) for k in ("last10", "final") for key in sig) else None,
    "none become significant": "No comparison that was not significant on the AUC became significant" if not nonsig_become else None,
    "PER fix": ", ".join(f"{per_fix[g][0]:+.1f}\\% in {names[g]} (p_t={per_fix[g][1]:.2f})" for g in GAMES[:2]),
    "PER fix 2": f"{per_fix['freeway'][0]:.1f}\\% in Freeway (p_t={per_fix['freeway'][1]:.2f}) and {per_fix['spaceinvaders'][0]:+.1f}\\% in Space Invaders (p_t={per_fix['spaceinvaders'][1]:.2f})",
    "PER pre-fix deficit": f"already {-per_fix['freeway'][2]:.1f}\\% and {-per_fix['spaceinvaders'][2]:.1f}\\% below DQN" if per_fix["freeway"][3] <= ALPHA and per_fix["spaceinvaders"][3] <= ALPHA and per_fix["breakout"][3] > ALPHA and per_fix["asterix"][3] > ALPHA else None,
    "348 identical": f"all {same} of their overlapping" if same == 348 else None,
    "Shapiro 6 of 16": f"for {sum(r['ps'] < .05 for r in A.values())} of the 16 comparisons",
    "Averaged losses": f"failed to beat DQN on {30 - A['asterix', 'averaged']['wins']}, {30 - A['freeway', 'averaged']['wins']} and {30 - A['spaceinvaders', 'averaged']['wins']} of the 30",
    "probability range": f"probability of {min(30 - A[g, 'averaged']['wins'] for g in GAMES[1:]) / 30:.2f} to {max(30 - A[g, 'averaged']['wins'] for g in GAMES[1:]) / 30:.2f}",
    "Averaged Wilcoxon all three": "the Wilcoxon test confirmed all three" if all(A[g, "averaged"]["pw"] <= ALPHA for g in GAMES[1:]) else None,
    "at least 24": f"at least {min(A[g, 'averaged']['wins'] for g in GAMES[1:])} of 30",
    "600 runs": "4\\times5\\times30=600",
    "3e9 frames": "3\\times10^9 frames" if 600 * 5_000_000 == 3_000_000_000 else None,
}
for what, s in quotes.items():
    check(s is not None and s.replace("10^-4", "10^{-4}").replace("10^6", "10^{6}").replace("10^5", "10^{5}").replace("10^9", "10^{9}") in text
          or (s is not None and s in text), f"text: {what}" + ("" if s is not None else " (claim not supported by data)"))

# ------------------------------------------------------- language and build
american = r"\b\w*(?:iz(?:e|es|ed|ing|ation|ations)|yz(?:e|ed|es|ing)|ization)\b|\b(?:behavior|color|favor|center|modeling|labeled|gray|artifact)\w*"
body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", tex, flags=re.S)  # drawing code, not prose
body = re.sub(r"\\(?:cite[tp]?|ref|label|eqref|input|url|bibliography\w*|includegraphics|begin|end)\{[^}]*\}", "", body)
body = re.sub(r"\\[A-Za-z]+", "", body)  # LaTeX command names are not prose
hits = sorted({m.group(0) for m in re.finditer(american, body, re.I)} - {"size", "Size", "sizes"})
check(not hits, f"British spelling (American forms found: {hits})")
check(not re.search(r"TODO|TBD|XXX|Conference Paper Title|Given Name Surname", tex), "no placeholders or template text")
log = (REPORT / "main.log").read_text(errors="ignore") if (REPORT / "main.log").exists() else ""
check(log and not re.search(r"undefined|Citation .* undefined|There were undefined", log), "LaTeX log: no undefined references")
bib = (REPORT / "refs.bib").read_text()
keys = set(re.findall(r"@\w+\{([^,]+),", bib)) - {"IEEEexample:BSTcontrol"}
cited = {k.strip() for c in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex) for k in c.split(",")}
check(cited <= keys, f"every cited key is in refs.bib (missing: {sorted(cited - keys)})")
check(keys <= cited, f"every refs.bib entry is cited (unused: {sorted(keys - cited)})")

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
