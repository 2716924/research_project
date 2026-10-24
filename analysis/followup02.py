"""Paired analysis of the follow-up runs (Results/followup-02).

Every comparison is paired by individual (same seed and sampled config) or, for
recipe-reference, by seed. Small groups (n = 5 or 7) are exploratory: read the
win counts next to the p-values. Groups with no results yet are skipped, and a
partly finished group uses the individuals present in both arms (n is reported).

  recipe-sampled    shared recipe (main trial) vs obando2021 recipe, per arm;
                    PER / n-step vs DQN within each recipe; change in the gap
  recipe-reference  PER / n-step vs DQN, published configuration, paired by seed
  lr-low-dqn        DQN at lr 3e-5 and 1e-5 vs the main-trial DQN at lr 1e-4
  seaquest-capped   each arm vs DQN on 30 individuals (truncated greedy episodes)
  lr-low-arms       each arm vs DQN at the same low lr; change in the gap from lr 1e-4

Main-trial partners come from analysis.corrected_rct01.load(), which asserts its
own provenance and pairing; this script asserts that each follow-up run has
the same seed and sampled config as its partner.

    .venv/bin/python -m analysis.followup02
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.corrected_rct01 import load

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = ROOT / "Results/followup-02"
OUT = ROOT / "analysis_output/followup-02"

VARIANTS = ["control", "ddqn", "per", "nstep", "averaged"]
LR_IDS = [12, 13, 18, 20, 22, 26, 28]
LOW_LRS = ["3e-05", "1e-05"]
METRICS = ["auc", "last_10_return"]


def read(group):
    runs = {}
    for path in sorted((FOLLOWUP / group / "results").glob("*/*/*/*_results_*.json")):
        arm, game, tag = path.parts[-4:-1]
        doc = json.loads(path.read_text())
        (run,) = doc["results"]
        assert doc["variant"] == arm and run["individual_id"] == int(tag[1:3]), path
        run["recipe"] = doc["training_protocol"].get("recipe", "shared")
        runs[arm, game, tag] = run
    return runs


def endpoint(run, metric):
    t = run["eval_trajectory"]
    return {"auc": run["auc"] / 1e6, "last_10_return": float(np.mean(t[-10:]))}[metric]


def compare(c, t):
    """Paired statistics for dicts id -> value, on the ids present in both."""
    ids = sorted(c.keys() & t.keys())
    c, t = np.array([c[i] for i in ids]), np.array([t[i] for i in ids])
    d = t - c
    varies = len(ids) > 1 and d.std() > 0
    return dict(n=len(ids), mean_control=c.mean(), mean_treatment=t.mean(), mean_diff=d.mean(),
                pct_change=100 * d.mean() / c.mean(), wins=int((d > 0).sum()),
                p=stats.ttest_rel(t, c).pvalue if varies else np.nan,
                wilcoxon_p=stats.wilcoxon(t, c).pvalue if varies else np.nan)


def gap_change(c0, t0, c1, t1):
    """Does the treatment-minus-DQN gap move between two conditions (1 vs 0)?"""
    ids = sorted(c0.keys() & t0.keys() & c1.keys() & t1.keys())
    g0 = {i: t0[i] - c0[i] for i in ids}
    g1 = {i: t1[i] - c1[i] for i in ids}
    s = compare(g0, g1)
    return dict(n=s["n"], gap_before=s["mean_control"], gap_after=s["mean_treatment"],
                gap_change=s["mean_diff"], gap_improved=s["wins"], p=s["p"], wilcoxon_p=s["wilcoxon_p"])


def check_pairing(runs, main):
    for (arm, game, tag), r in runs.items():
        partner = main.get((arm, game), {}).get(r["individual_id"])
        if partner is None:
            continue
        config = dict(r["config"], lr=partner["config"]["lr"]) if "-lr" in tag else r["config"]
        assert (r["seed"], config) == (partner["seed"], partner["config"]), (arm, game, tag)


def by_id(runs, arm, game, metric, suffix=""):
    return {r["individual_id"]: endpoint(r, metric) for (a, g, tag), r in runs.items()
            if a == arm and g == game and tag[3:] == suffix}


def main_ids(main, arm, game, metric, ids=LR_IDS):
    return {i: endpoint(main[arm, game][i], metric) for i in ids}


def analyse(main, groups):
    paired, gaps = [], []

    def add(group, game, metric, comparison, s, lr="", rows=paired):
        rows.append(dict(group=group, game=game, lr=lr, metric=metric, comparison=comparison, **s))

    for metric in METRICS:
        rs = groups["recipe-sampled"]
        for game in sorted({g for _, g, _ in rs}):
            shared = {a: main_ids(main, a, game, metric) for a in ("control", "per", "nstep")}
            obando = {a: by_id(rs, a, game, metric) for a in ("control", "per", "nstep")}
            for a in shared:
                add("recipe-sampled", game, metric, f"{a}: shared -> obando2021", compare(shared[a], obando[a]))
            for a in ("per", "nstep"):
                add("recipe-sampled", game, metric, f"{a} vs control (shared, same 7)",
                    compare(shared["control"], shared[a]))
                add("recipe-sampled", game, metric, f"{a} vs control (obando2021)",
                    compare(obando["control"], obando[a]))
                add("recipe-sampled", game, metric, f"{a} gap: shared -> obando2021",
                    gap_change(shared["control"], shared[a], obando["control"], obando[a]), rows=gaps)

        rr = groups["recipe-reference"]
        for game in sorted({g for _, g, _ in rr}):
            c = by_id(rr, "control", game, metric)
            for a in ("per", "nstep"):
                t = by_id(rr, a, game, metric)
                if t:
                    add("recipe-reference", game, metric, f"{a} vs control (by seed)", compare(c, t))

        ld = groups["lr-low-dqn"]
        for game in sorted({g for _, g, _ in ld}):
            for lr in LOW_LRS:
                add("lr-low-dqn", game, metric, "control: lr 1e-4 -> low lr",
                    compare(main_ids(main, "control", game, metric), by_id(ld, "control", game, metric, f"-lr{lr}")),
                    lr=lr)

        sq = groups["seaquest-capped"]
        c = by_id(sq, "control", "seaquest", metric)
        for a in VARIANTS[1:]:
            t = by_id(sq, a, "seaquest", metric)
            if c and t:
                add("seaquest-capped", "seaquest", metric, f"{a} vs control", compare(c, t))

        la = groups["lr-low-arms"]
        for game in sorted({g for _, g, _ in la}):
            for lr in LOW_LRS:
                c = by_id(ld, "control", game, metric, f"-lr{lr}")
                for a in VARIANTS[1:]:
                    t = by_id(la, a, game, metric, f"-lr{lr}")
                    if not (c and t):
                        continue
                    add("lr-low-arms", game, metric, f"{a} vs control (same lr)", compare(c, t), lr=lr)
                    add("lr-low-arms", game, metric, f"{a} gap: lr 1e-4 -> low lr",
                        gap_change(main_ids(main, "control", game, metric), main_ids(main, a, game, metric), c, t),
                        lr=lr, rows=gaps)
    return pd.DataFrame(paired), pd.DataFrame(gaps)


def run_table(groups):
    rows = []
    for group, runs in groups.items():
        for (arm, game, tag), r in sorted(runs.items()):
            t = r["eval_trajectory"]
            truncated = r.get("eval_truncated_episodes")
            rows.append(dict(group=group, arm=arm, game=game, tag=tag, individual_id=r["individual_id"],
                             seed=r["seed"], recipe=r["recipe"], **r["config"], auc=r["auc"],
                             final_return=t[-1], last_10_return=float(np.mean(t[-10:])),
                             truncated_eval_episodes=sum(truncated) if truncated is not None else np.nan,
                             evaluations_with_truncation=(sum(x > 0 for x in truncated)
                                                          if truncated is not None else np.nan)))
    return pd.DataFrame(rows)


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    data, _ = load()
    groups = {g: read(g) for g in ("recipe-sampled", "recipe-reference", "lr-low-dqn",
                                   "seaquest-capped", "lr-low-arms")}
    for group, runs in groups.items():
        if group == "recipe-reference":  # published config, not the sampled one: check seeds only
            assert all(r["seed"] == data["control", "breakout"][r["individual_id"]]["seed"] for r in runs.values())
        else:
            check_pairing(runs, data)
    runs = run_table(groups)
    paired, gaps = analyse(data, groups)
    runs.to_csv(OUT / "tables/runs.csv", index=False)
    paired.to_csv(OUT / "tables/paired.csv", index=False)
    gaps.to_csv(OUT / "tables/gap_change.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print("Runs per group:", {g: len(r) for g, r in groups.items()})
    cols = ["game", "lr", "comparison", "n", "mean_control", "mean_treatment", "pct_change", "wins", "p", "wilcoxon_p"]
    for (group, metric), df in paired.groupby(["group", "metric"], sort=False):
        print(f"\n== {group} / {metric}")
        print(df[cols].round(4).to_string(index=False))
    if len(gaps):
        print("\n== gap change (treatment - DQN), before -> after")
        print(gaps.drop(columns="group").round(4).to_string(index=False))
    sq = runs[runs.group == "seaquest-capped"]
    if len(sq):
        print("\n== seaquest-capped: greedy episodes cut at the step cap (of 1000 per run)")
        print(sq.groupby("arm")[["truncated_eval_episodes", "evaluations_with_truncation"]]
              .agg(["mean", "max"]).round(1).to_string())


if __name__ == "__main__":
    main()
