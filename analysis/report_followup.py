"""Tables and a figure for the additional experiments in the report (Results/followup-02).

  tab_sanity.tex    DQN against the published MinAtar DQN (Young and Tian 2019, Fig. 3)
  tab_recipe.tex    DQN, PER and n-step DQN under the shared recipe, the Obando-Ceron and Castro
                    (2021) recipe on the same seven configurations, and their published configuration
  tab_lowlr.tex     every arm at learning rates 1e-4 (main trial), 3e-5 and 1e-5, same seven configurations
  tab_seaquest.tex  Seaquest with greedy evaluation episodes capped at 100,000 steps (all 30 configurations)
  low_lr.pdf        mean AUC against the learning rate for the seven configurations

Pairing with the main trial is asserted by analysis.followup02.check_pairing. The small groups
(n = 5 or 7) are exploratory, so their tables report effects and win counts but no tests;
Seaquest (n = 30) gets the primary analysis as its own family of four comparisons.

    .venv/bin/python -m analysis.report_followup
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.corrected_rct01 import GAMES, NAMES, ROOT, VARIANTS, load
from analysis.followup02 import LR_IDS, check_pairing, endpoint, read
from analysis.report import COL, COLORS, LABELS, MARKERS, SHORT, STYLES, fmt_p, fmt_pct, style

OUT = ROOT / "analysis_output/followup-02/report"
SQ_ALPHA = 0.05 / 4
# Approximate final DQN returns read by eye from Fig. 3 of Young and Tian (2019): the moving average
# of the last 100 training episodes (epsilon-greedy) after 5M frames, at the best step size, 30 seeds.
PUBLISHED = {"breakout": 10, "asterix": 17, "freeway": 50, "seaquest": 20, "spaceinvaders": 45}
SANITY_GAMES = ["breakout", "asterix", "freeway", "seaquest", "spaceinvaders"]
SANITY_NAMES = {**NAMES, "seaquest": "Seaquest"}
LRS = ["1e-05", "3e-05", ""]  # tag suffixes; "" is the main-trial rate 1e-4
LR_LABELS = {"1e-05": r"$10^{-5}$", "3e-05": r"$3{\times}10^{-5}$", "": r"$10^{-4}$"}


def by_id(runs, arm, game, metric="auc", suffix=""):
    return {r["individual_id"]: endpoint(r, metric) for (a, g, tag), r in runs.items()
            if a == arm and g == game and tag[3:] == suffix}


def main_ids(data, arm, game, metric="auc", ids=LR_IDS):
    return {i: endpoint(data[arm, game][i], metric) for i in ids}


def low_lr(data, groups, arm, game, suffix, metric="auc"):
    """Per-individual values for one arm at one learning rate on the seven configurations."""
    if suffix == "":
        return main_ids(data, arm, game, metric)
    return by_id(groups["lr-low-dqn" if arm == "control" else "lr-low-arms"], arm, game, metric, f"-lr{suffix}")


def compare(c, t):
    ids = sorted(c.keys() & t.keys())
    c, t = np.array([c[i] for i in ids]), np.array([t[i] for i in ids])
    d = t - c
    out = dict(n=len(ids), control=c.mean(), treatment=t.mean(), pct=100 * d.mean() / c.mean(), wins=int((d > 0).sum()))
    if len(ids) >= 20:
        out.update(p_t=stats.ttest_rel(t, c).pvalue, p_w=stats.wilcoxon(d).pvalue)
    return out


def analyse(data, groups):
    res = {}
    sq, rr, rs = groups["seaquest-capped"], groups["recipe-reference"], groups["recipe-sampled"]

    res["sanity"] = {}
    for g in SANITY_GAMES:
        if g == "seaquest":
            pop = [endpoint(r, "last_10_return") for (a, _, _), r in sq.items() if a == "control"]
            lr4 = [endpoint(r, "last_10_return") for (a, _, _), r in sq.items()
                   if a == "control" and r["config"]["lr"] == 1e-4]
            row = dict(population=np.mean(pop), lr_1e4=np.mean(lr4), lr_1e5=None, reference=None, n_lr4=len(lr4))
        else:
            row = dict(population=np.mean([endpoint(r, "last_10_return") for r in data["control", g].values()]),
                       lr_1e4=np.mean(list(main_ids(data, "control", g, "last_10_return").values())),
                       lr_1e5=np.mean(list(low_lr(data, groups, "control", g, "1e-05", "last_10_return").values())),
                       reference=np.mean(list(by_id(rr, "control", g, "last_10_return").values())), n_lr4=len(LR_IDS))
        res["sanity"][g] = dict(published=PUBLISHED[g], **row)

    res["recipe"] = {}
    for g in ("asterix", "freeway"):
        blocks = {"shared": {a: main_ids(data, a, g) for a in ("control", "per", "nstep")},
                  "obando": {a: by_id(rs, a, g) for a in ("control", "per", "nstep")},
                  "reference": {a: by_id(rr, a, g) for a in ("control", "per", "nstep")}}
        res["recipe"][g] = {b: {"control_mean": float(np.mean(list(v["control"].values()))),
                                **{a: compare(v["control"], v[a]) for a in ("per", "nstep")}}
                            for b, v in blocks.items()}

    res["lowlr"] = {}
    for g in GAMES:
        res["lowlr"][g] = {}
        for s in LRS:
            vals = {a: low_lr(data, groups, a, g, s) for a in VARIANTS}
            res["lowlr"][g][s or "1e-04"] = {
                "means": {a: float(np.mean(list(v.values()))) for a, v in vals.items()},
                "se": {a: float(np.std(list(v.values()), ddof=1) / np.sqrt(len(v))) for a, v in vals.items()},
                "n": {a: len(v) for a, v in vals.items()},
                **{a: compare(vals["control"], vals[a]) for a in VARIANTS[1:]}}

    res["seaquest"] = {}
    c = by_id(sq, "control", "seaquest")
    for a in VARIANTS:
        runs = [r for (arm, _, _), r in sq.items() if arm == a]
        trunc = [sum(r["eval_truncated_episodes"]) for r in runs]
        res["seaquest"][a] = dict(mean=float(np.mean([endpoint(r, "auc") for r in runs])),
                                  sd=float(np.std([endpoint(r, "auc") for r in runs], ddof=1)),
                                  truncated=float(np.mean(trunc)), runs=len(runs),
                                  **(compare(c, by_id(sq, a, "seaquest")) if a != "control" else {}))
    pts = [(ret, k) for r in sq.values() for ret, k in zip(r["eval_trajectory"], r["eval_truncated_episodes"])]
    res["seaquest_points"] = dict(
        n=len(pts), with_truncation=sum(k > 0 for _, k in pts),
        mean_return_truncated=float(np.mean([x for x, k in pts if k > 0])),
        mean_return_untruncated=float(np.mean([x for x, k in pts if k == 0])))
    return res


# ---------------------------------------------------------------- LaTeX tables

def num(x, digits=1):
    return "--" if x is None else f"{x:.{digits}f}"


def table_sanity(res):
    rows = []
    for g in SANITY_GAMES:
        s = res["sanity"][g]
        rows.append(rf"{SANITY_NAMES[g]} & $\approx${s['published']} & {num(s['population'])} & {num(s['lr_1e4'])} "
                    rf"& {num(s['lr_1e5'])} & {num(s['reference'])} \\")
    head = (r"\begin{tabular}{@{}lrrrrr@{}}" "\n" r"\toprule" "\n"
            r" & Published & \multicolumn{3}{c}{Shared recipe} & Published \\" "\n"
            r"\cmidrule(lr){3-5}" "\n"
            r"Game & DQN & All 30 & $10^{-4}$ (7) & $10^{-5}$ (7) & config (5) \\" "\n" r"\midrule")
    return "\n".join([head, *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_recipe(res):
    rows = []
    for g in ("asterix", "freeway"):
        for k, a in enumerate(("control", "per", "nstep")):
            cells = []
            for b in ("shared", "obando", "reference"):
                blk = res["recipe"][g][b]
                if a == "control":
                    cells += [f"{blk['control_mean']:.1f}", "", ""]
                else:
                    r = blk[a]
                    cells += [f"{r['treatment']:.1f}", fmt_pct(r["pct"]), f"{r['wins']}/{r['n']}"]
            rows.append(rf"{NAMES[g] if k == 0 else ''} & {SHORT[a]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    head = (r"\begin{tabular}{@{}llrrrrrrrrr@{}}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{3}{c}{Shared recipe, 7 configurations} & "
            r"\multicolumn{3}{c}{Obando-Ceron--Castro recipe, same 7} & "
            r"\multicolumn{3}{c@{}}{Published configuration, 5 seeds} \\" "\n"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}" "\n"
            r"Game & Arm & AUC & $\Delta$ (\%) & Wins & AUC & $\Delta$ (\%) & Wins & AUC & $\Delta$ (\%) & Wins \\"
            "\n" r"\midrule")
    return "\n".join([head, *rows, r"\end{tabular}"]) + "\n"


def table_lowlr(res):
    rows = []
    for g in GAMES:
        for k, a in enumerate(VARIANTS):
            cells = []
            for s in LRS:
                blk = res["lowlr"][g][s or "1e-04"]
                cells.append(f"{blk['means']['control']:.1f}" if a == "control"
                             else f"{fmt_pct(blk[a]['pct'])} ({blk[a]['wins']})")
            rows.append(rf"{NAMES[g] if k == 0 else ''} & {SHORT[a]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    head = (r"\begin{tabular}{@{}llrrr@{}}" "\n" r"\toprule" "\n"
            r"Game & Arm & " + " & ".join(LR_LABELS[s] for s in LRS) + r" \\" "\n" r"\midrule")
    return "\n".join([head, *rows, r"\end{tabular}"]) + "\n"


def table_seaquest(res):
    rows = []
    for a in VARIANTS:
        r = res["seaquest"][a]
        if a == "control":
            rows.append(rf"DQN & {r['mean']:.1f} ({r['sd']:.1f}) & & & & & {r['truncated']:.0f} \\")
        else:
            rows.append(rf"{LABELS[a]} & {r['mean']:.1f} ({r['sd']:.1f}) & {fmt_pct(r['pct'])} & {r['wins']} "
                        rf"& {fmt_p(r['p_t'], r['p_t'] <= SQ_ALPHA)} & {fmt_p(r['p_w'], r['p_w'] <= SQ_ALPHA)} "
                        rf"& {r['truncated']:.0f} \\")
    head = (r"\begin{tabular}{@{}lrrrrrr@{}}" "\n" r"\toprule" "\n"
            r"Arm & AUC & $\Delta$ (\%) & Wins & $p_t$ & $p_W$ & Capped \\" "\n" r"\midrule")
    return "\n".join([head, *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


# --------------------------------------------------------------------- figure

def fig_low_lr(res):
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.9), layout="constrained")
    x = np.arange(len(LRS))
    for ax, game in zip(axes.flat, GAMES):
        for k, arm in enumerate(VARIANTS):
            blks = [res["lowlr"][game][s or "1e-04"] for s in LRS]
            ax.errorbar(x + (k - 2) * .07, [b["means"][arm] for b in blks], yerr=[b["se"][arm] for b in blks],
                        color=COLORS[arm], ls=STYLES[arm], lw=1, marker=MARKERS[arm], ms=3.2, elinewidth=.6,
                        capsize=0, label=LABELS[arm])
        ax.set_xticks(list(x), [LR_LABELS[s] for s in LRS])
        ax.set_xlim(-.3, len(LRS) - .7)
        ax.set_ylim(bottom=0)
        ax.set_title(NAMES[game], pad=3)
        style(ax)
    fig.supxlabel("Learning rate (the same seven configurations)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Mean AUC ($10^6$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, handlelength=2.4,
               columnspacing=1.0)
    fig.savefig(OUT / "low_lr.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(OUT / "low_lr.png", dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load()
    groups = {g: read(g) for g in ("recipe-sampled", "recipe-reference", "lr-low-dqn", "seaquest-capped", "lr-low-arms")}
    for name, runs in groups.items():
        if name == "recipe-reference":  # published config, not a sampled one: same seeds only
            assert all(r["seed"] == data["control", "breakout"][r["individual_id"]]["seed"] for r in runs.values())
        else:
            check_pairing(runs, data)
    assert {len(v) for v in groups.values()} == {42, 40, 56, 150, 224}
    res = analyse(data, groups)
    (OUT / "numbers.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "tab_sanity.tex").write_text(table_sanity(res))
    (OUT / "tab_recipe.tex").write_text(table_recipe(res))
    (OUT / "tab_lowlr.tex").write_text(table_lowlr(res))
    (OUT / "tab_seaquest.tex").write_text(table_seaquest(res))
    fig_low_lr(res)
    for name in ("tab_sanity", "tab_recipe", "tab_lowlr", "tab_seaquest"):
        print(f"--- {name}\n" + (OUT / f"{name}.tex").read_text())
    print(json.dumps(res["seaquest_points"], indent=1))


if __name__ == "__main__":
    main()
