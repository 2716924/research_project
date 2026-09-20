#!/usr/bin/env python3
"""Reproduce the training-data review without modifying experiment inputs.

Run: .venv/bin/python -m analysis.review
Requires numpy, scipy, pandas and matplotlib. Outputs go to analysis_output/review.
"""
import ast
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rct_review_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rct_review_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_output/review"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
AUDIT = OUT / "audit"
LOG_ROOTS = [ROOT / "slurm/logs", ROOT / "new_logs", ROOT / "logs"]
VARIANTS = ["control", "ddqn", "per", "nstep", "averaged"]
LABELS = ["DQN", "Double DQN", "PER", "n-step", "Averaged DQN"]
COLORS = ["#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
GAMES = ["breakout", "asterix", "freeway", "spaceinvaders"]
NAMES = dict(zip(GAMES, ["Breakout", "Asterix", "Freeway", "Space Invaders"]))


def savefig(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_results():
    data, rows, evaluation_rows = {}, [], []
    for path in sorted((ROOT / "results").rglob("*.json")):
        raw = json.loads(path.read_text())
        game, variant = path.parent.name, raw["variant"]
        rs = sorted(raw["results"], key=lambda r: r["individual_id"])
        ids = [r["individual_id"] for r in rs]
        assert len(ids) == len(set(ids)) == 30, path
        assert ids == list(range(30)), path
        assert raw["total_frames"] == 5_000_000 and raw["eval_frequency"] == 50_000, path
        assert raw["num_eval_episodes"] == 10 and raw["eval_seeds"] == list(range(1000, 1010)), path
        for r in rs:
            y = np.asarray(r["eval_trajectory"])
            assert len(y) == 100 and np.isfinite(y).all(), path
            x = np.arange(1, len(y) + 1) * raw["eval_frequency"]
            auc = float(np.trapezoid(y, x=x) if hasattr(np, "trapezoid") else np.trapz(y, x=x))
            assert np.isclose(auc, r["auc"], rtol=1e-12, atol=1e-6), path
            rows.append(dict(game=game, variant=variant, individual_id=r["individual_id"],
                             seed=r["seed"], auc=auc, auc_per_observed_frame=auc/(x[-1]-x[0]),
                             final_checkpoint=y[-1], last_10_checkpoints_mean=y[-10:].mean(),
                             **r["config"]))
            evaluation_rows.extend(dict(game=game, variant=variant, individual_id=r["individual_id"],
                                        frame=int(frame), mean_return=float(score)) for frame, score in zip(x, y))
        data[game, variant] = rs
    for game in GAMES:
        for v in VARIANTS:
            for c, t in zip(data[game, "control"], data[game, v]):
                assert (c["individual_id"], c["seed"], c["config"]) == (t["individual_id"], t["seed"], t["config"]), (game, v)
    pd.DataFrame(rows).to_csv(TABLES / "individual_metrics.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(TABLES / "evaluation_curves.csv.gz", index=False)
    return data


def result_figures(data):
    rng = np.random.default_rng(20260917)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, game in zip(axes.flat, GAMES):
        for v, label, color in zip(VARIANTS, LABELS, COLORS):
            y = np.array([r["eval_trajectory"] for r in data[game, v]])
            # Resample whole individuals, preserving each longitudinal trajectory.
            bootstrap = y[rng.integers(0, len(y), (3000, len(y)))].mean(axis=1)
            low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
            x = np.arange(1, 101) * .05
            ax.plot(x, y.mean(axis=0), color=color, label=label, lw=1.4)
            ax.fill_between(x, low, high, color=color, alpha=.12)
        ax.set(title=NAMES[game], xlabel="Training environment frames (millions)", ylabel="Greedy evaluation return")
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Learning curves: 30 configuration–seed individuals per arm\nMean and pointwise 95% bootstrap intervals; no temporal smoothing", fontsize=13)
    savefig(fig, "learning_curves")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    comparisons = []
    for ax, game in zip(axes.flat, GAMES):
        values = [np.array([r["auc"] for r in data[game, v]]) / 1e6 for v in VARIANTS]
        ax.boxplot(values, showfliers=False, widths=.5)
        for i, (values_v, color) in enumerate(zip(values, COLORS), 1):
            ax.scatter(i + rng.uniform(-.15, .15, len(values_v)), values_v, color=color, alpha=.65, s=17)
        ax.set_xticks(range(1, 6), LABELS, rotation=15, fontsize=8)
        ax.set(title=NAMES[game], ylabel="AUC (million return × frames)")
        ax.grid(axis="y", alpha=.15)
        for v in VARIANTS[1:]:
            c = np.array([r["auc"] for r in data[game, "control"]])
            t = np.array([r["auc"] for r in data[game, v]])
            d = t-c
            se = stats.sem(d)
            radius = stats.t.ppf(1-.0125/2, len(d)-1) * se
            test = stats.ttest_rel(t, c)
            comparisons.append(dict(game=game, variant=v, n_pairs=len(d), mean_control_auc=c.mean(),
                                    mean_treatment_auc=t.mean(), mean_difference=d.mean(),
                                    ci_98_75_low=d.mean()-radius, ci_98_75_high=d.mean()+radius,
                                    percent_change_of_means=100*d.mean()/c.mean(),
                                    paired_dz=d.mean()/d.std(ddof=1), t_statistic=test.statistic,
                                    p_value=test.pvalue, p_bonferroni_4=min(1,4*test.pvalue),
                                    p_bonferroni_16=min(1,16*test.pvalue),
                                    significant_per_game=bool(test.pvalue<=.0125),
                                    shapiro_p=stats.shapiro(d).pvalue, matched_win_fraction=(d>0).mean()))
    fig.suptitle("AUC distributions: each dot is one individual\nBoxes show median and IQR; all 30 observations are displayed", fontsize=13)
    savefig(fig, "auc_distributions")
    pd.DataFrame(comparisons).to_csv(TABLES / "paired_auc_comparisons.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, game in zip(axes.flat, GAMES):
        for i, row in enumerate(r for r in comparisons if r["game"] == game):
            mean = row["mean_difference"] / 1e6
            error = (row["ci_98_75_high"]-row["mean_difference"]) / 1e6
            ax.errorbar(mean, i, xerr=error, fmt="o", color=COLORS[i+1], capsize=4)
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_yticks(range(4), LABELS[1:])
        ax.invert_yaxis()
        ax.set(title=NAMES[game], xlabel="Treatment − DQN AUC (million return × frames)")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Paired treatment effects: mean difference and 98.75% t intervals\nBonferroni family: four comparisons within each game; inspect normality diagnostics", fontsize=12)
    savefig(fig, "paired_auc_effects")
    return comparisons


def density_figures(data):
    """Adapt Pretorius et al. Fig. 3; KDE is not a normality assumption."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    parameters = []
    for ax, game in zip(axes.flat, GAMES):
        samples = [np.array([r["auc"] for r in data[game, v]]) / 1e6 for v in VARIANTS]
        kernels = [stats.gaussian_kde(values, bw_method="scott") for values in samples]
        # Include the full visible support, including smoothing tails below zero.
        bandwidths = [float(np.sqrt(k.covariance[0, 0])) for k in kernels]
        low = min(values.min()-3*bw for values, bw in zip(samples, bandwidths))
        high = max(values.max()+3*bw for values, bw in zip(samples, bandwidths))
        grid = np.linspace(low, high, 700)
        densities = [k(grid) for k in kernels]
        for i, (v, values, density, bw, color) in enumerate(zip(VARIANTS, samples, densities, bandwidths, COLORS)):
            baseline = 4-i
            # Peak normalization keeps broad densities visible next to narrow ones.
            # This is a shape display, not a comparison of vertical density heights.
            scale = .82 / density.max()
            ax.fill_between(grid, baseline, baseline + density*scale, color=color, alpha=.3)
            ax.plot(grid, baseline + density*scale, color=color, lw=1.5)
            ax.plot(values, np.full(len(values), baseline-.05), "|", color=color, ms=6, alpha=.7)
            parameters.append(dict(game=game, variant=v, n=len(values), mean_auc_millions=values.mean(),
                                   sd_auc_millions=values.std(ddof=1), kde_bandwidth_millions=bw))
        ax.axvline(0, color="grey", lw=.7, ls=":")
        ax.set_yticks(range(5), LABELS[::-1])
        ax.set(title=NAMES[game], xlabel="AUC (million return × frames)", ylim=(-.2,5))
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("AUC density curves — adapted from Pretorius et al. (2020), Fig. 3\nGaussian KDE, Scott bandwidth; ticks show 30 individuals; each row scaled to the same peak height", fontsize=12)
    fig.supxlabel("Kernel smoothing can extend below zero; the fitted curves do not assert Gaussian outcomes.", fontsize=10)
    savefig(fig, "pretorius_auc_densities")
    pd.DataFrame(parameters).to_csv(TABLES / "density_parameters.csv", index=False)

    fig, axes = plt.subplots(4, 4, figsize=(15, 12), layout="constrained")
    for row, game in enumerate(GAMES):
        control = np.array([r["auc"] for r in data[game,"control"]]) / 1e6
        for col, v in enumerate(VARIANTS[1:]):
            ax = axes[row,col]
            values = np.array([r["auc"] for r in data[game,v]]) / 1e6 - control
            mean, sd = values.mean(), values.std(ddof=1)
            low, high = min(values.min(),mean-3*sd), max(values.max(),mean+3*sd)
            grid = np.linspace(low, high, 300)
            ax.hist(values, bins="auto", density=True, color=COLORS[col+1], alpha=.3)
            ax.plot(grid, stats.norm.pdf(grid, mean, sd), color=COLORS[col+1], lw=1.7)
            ax.plot(values, np.zeros_like(values), "|", color=COLORS[col+1], ms=7)
            ax.axvline(0, color="grey", ls="--", lw=.8)
            p = stats.shapiro(values).pvalue
            ax.set_title(f"{NAMES[game]} · {LABELS[col+1]}\nShapiro p={p:.3g}" + (" *" if p<.05 else ""), fontsize=10)
            ax.set_xlabel("Paired AUC difference (millions)",fontsize=9)
            ax.set_ylabel("Density",fontsize=9)
            ax.tick_params(labelsize=8)
    fig.suptitle("Gaussian reference fits to the 30 paired AUC differences\nHistogram: observed data · line: normal density with sample mean and SD · * Shapiro p < 0.05", fontsize=13)
    fig.supxlabel("Normal fits are diagnostics, not evidence that outcomes are Gaussian; this extends the Pretorius-style density display.", fontsize=10)
    savefig(fig, "paired_auc_gaussian_diagnostics")


def parse_logs(data):
    attempts, jobs = [], []
    number = r"[-+\w.]+"
    loss_re = re.compile(r"\[(\w+)\] frame (\d+)/(\d+) \| loss: ("+number+r") \| epsilon: ("+number+r")")
    eval_re = re.compile(r"\[(\w+)\] frame (\d+)/(\d+) \| eval mean return: ("+number+r")")
    file_hashes = {}
    paths = sorted({p for root in LOG_ROOTS for p in root.rglob("*.out")})
    for path in paths:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        text = content.decode(errors="replace")
        game_match = re.search(r"^Game:\s+MinAtar/(\w+)-v1", text, re.M)
        variant_match = re.search(r"^Variant:\s+(\w+)\s*$", text, re.M)
        if not game_match or not variant_match:
            raise ValueError(f"Unrecognised log metadata: {path}")
        game, variant = game_match[1].lower(), variant_match[1]
        err = path.with_suffix(".err")
        err_text = err.read_text(errors="replace") if err.exists() else ""
        job_id_match = re.search(r"^\s*(?:#\s*)?Job ID:\s+(\d+)", text, re.M)
        job_id = job_id_match[1] if job_id_match else ""
        source = str(path.relative_to(ROOT))
        duplicate_of = file_hashes.get(digest, "")
        jobs.append(dict(source=str(path.relative_to(ROOT)), game=game, variant=variant, job_id=job_id,
                         stdout_sha256=digest, duplicate_stdout_of=duplicate_of,
                         completed_markers=len(re.findall(r"^Completed ", text, re.M)),
                         time_limit_cancelled="DUE TO TIME LIMIT" in err_text,
                         cuda_warning="CUDA initialization" in err_text,
                         traceback="Traceback" in err_text,
                         resubmitted=bool(re.search(r"Submitted batch job", text))))
        if duplicate_of:
            continue
        file_hashes[digest] = source
        current = None
        for line_number, line in enumerate(text.splitlines(), 1):
            start = re.match(r"--- Individual (\d+)/(\d+) ---", line)
            if start:
                current = dict(source=str(path.relative_to(ROOT)), job_id=job_id, game=game, variant=variant,
                               individual_id=int(start[1])-1, start_line=line_number,
                               completed=False, loss=[], evaluation=[])
                attempts.append(current)
            if current is None:
                continue
            settings = re.match(r"Seed: (\d+) \| Settings: (.*)", line)
            if settings:
                current.update(seed=int(settings[1]), config=ast.literal_eval(settings[2]))
            loss = loss_re.search(line)
            evaluation = eval_re.search(line)
            if loss:
                assert loss[1] == variant
                current["loss"].append((int(loss[2]), float(loss[4]), float(loss[5])))
            if evaluation:
                assert evaluation[1] == variant
                current["evaluation"].append((int(evaluation[2]), float(evaluation[4])))
            complete = re.match(r"Completed (\w+)\. AUC: (.*)", line)
            if complete:
                assert complete[1] == variant
                current.update(completed=True, logged_auc=float(complete[2]))

    by_key = defaultdict(list)
    for index, a in enumerate(attempts):
        a["attempt_id"] = index
        key = a["game"], a["variant"], a["individual_id"]
        by_key[key].append(a)
        target = next((r for r in data.get(key[:2], []) if r["individual_id"] == key[2]), None)
        a["matches_json"] = bool(target and a["completed"] and
                                  a.get("seed") == target["seed"] and a.get("config") == target["config"] and
                                  np.isclose(a["logged_auc"], target["auc"], rtol=1e-12, atol=1e-4) and
                                  [f for f, _ in a["evaluation"]] == list(range(50000,5000001,50000)) and
                                  np.allclose([v for _, v in a["evaluation"]], target["eval_trajectory"], rtol=0, atol=.00051))
        a["selected"] = False
        frames = [r[0] for r in a["loss"]]
        assert all(x < y for x, y in zip(frames, frames[1:])), key
    for group in by_key.values():
        candidates = [a for a in group if a["matches_json"]]
        if candidates:
            # Deterministic representative; duplicates are retained in the audit.
            candidates[0]["selected"] = True
    audit, bins = [], []
    for a in attempts:
        samples = np.array(a["loss"])
        if not len(samples):
            audit.append({k:v for k,v in a.items() if k not in ("loss", "evaluation", "config")} | dict(
                loss_samples=0, evaluation_points=len(a["evaluation"]), last_frame=0,
                nonfinite_loss=0, negative_loss=0, zero_loss=0))
            continue
        audit.append({k:v for k,v in a.items() if k not in ("loss", "evaluation", "config")} | dict(
            loss_samples=len(samples), evaluation_points=len(a["evaluation"]), last_frame=int(samples[-1,0]),
            nonfinite_loss=int((~np.isfinite(samples[:,1])).sum()), negative_loss=int((samples[:,1]<0).sum()),
            zero_loss=int((samples[:,1]==0).sum()), loss_median=float(np.median(samples[:,1])),
            loss_p95=float(np.percentile(samples[:,1],95)), loss_max=float(np.max(samples[:,1]))))
        # Non-overlapping 50k frame bins; median of logged minibatches, not all updates.
        bin_ids = ((samples[:,0]-1)//50000).astype(int)
        for b in np.unique(bin_ids):
            chunk = samples[bin_ids == b]
            bins.append(dict(attempt_id=a["attempt_id"], source=a["source"], game=a["game"], variant=a["variant"],
                             individual_id=a["individual_id"], selected=a["selected"], completed=a["completed"],
                             frame_bin_end=int((b+1)*50000), loss_median=float(np.median(chunk[:,1])),
                             loss_p95=float(np.percentile(chunk[:,1],95)), samples=len(chunk),
                             epsilon_last=float(chunk[-1,2])))
    pd.DataFrame(audit).to_csv(AUDIT / "log_attempts.csv", index=False)
    pd.DataFrame(jobs).to_csv(AUDIT / "log_jobs.csv", index=False)
    binned = pd.DataFrame(bins)
    binned.to_csv(TABLES / "loss_bins.csv.gz", index=False)
    coverage = []
    for game in GAMES:
        for v in VARIANTS:
            selected = [a for a in attempts if a["game"] == game and a["variant"] == v and a["selected"]]
            coverage.append(dict(game=game, variant=v, results_n=len(data[game,v]),
                                 matched_log_n=len(selected),
                                 missing_individual_ids=json.dumps(sorted(set(range(30))-{a["individual_id"] for a in selected}))))
    pd.DataFrame(coverage).to_csv(AUDIT / "log_coverage.csv", index=False)
    return attempts, binned, jobs


def loss_figures(attempts, bins):
    for game in GAMES:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for ax, v, label, color in zip(axes.flat, VARIANTS, LABELS, COLORS):
            subset = bins[(bins.game == game) & (bins.variant == v) & bins.selected]
            n = subset.individual_id.nunique()
            if not n:
                ax.text(.5,.5,"No matched completed loss history",ha="center",transform=ax.transAxes)
                ax.set_title(label)
                continue
            counts = subset.groupby("frame_bin_end").individual_id.nunique()
            assert len(counts) == 100 and (counts == n).all(), (game, v, counts)
            grouped = subset.groupby("frame_bin_end").loss_median
            q = grouped.quantile([.25,.5,.75]).unstack()
            x = q.index.to_numpy()/1e6
            # Zero-rounded values remain in tables and are masked only on log axes.
            positive = lambda a: np.where(np.asarray(a)>0, a, np.nan)
            ax.plot(x, positive(q[.5]), color=color, lw=1.6)
            ax.fill_between(x, positive(q[.25]), positive(q[.75]), color=color, alpha=.2)
            ax.set(yscale="log", title=label, xlabel="Training frames (millions)", ylabel="Logged minibatch loss")
            ax.grid(alpha=.15)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.02,.9,"50,000-frame bins\nMedian of logged loss within each run\nLine: median across matched runs\nBand: interquartile range, not a CI\n\nSeparate y-scales for diagnostics\nPER uses importance-weighted MSE\nLoss magnitude does not rank policies\n\nOnly completed attempts matching JSON\nRepeated attempts are counted once",va="top", fontsize=10, linespacing=1.6)
        fig.suptitle(f"{NAMES[game]}: training-loss diagnostics", fontsize=14)
        savefig(fig, f"{game}_loss")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, v, label, color in zip(axes.flat, VARIANTS, LABELS, COLORS):
        subset = bins[(bins.game == "seaquest") & (bins.variant == v)]
        for _, a in subset.groupby("attempt_id"):
            ax.plot(a.frame_bin_end/1e6, np.where(a.loss_median>0,a.loss_median,np.nan), color=color)
        ax.set(yscale="log",title=label,xlabel="Training frames (millions)",ylabel="Median logged loss / 50k bin")
        ax.set_xlim(0,5)
        ax.grid(alpha=.15)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02,.85,"INCOMPLETE RUNS\nOne individual per variant\nEach trace ends at the last log sample\nNo completed Seaquest result JSON\n\nNo cross-arm statistical comparison\nNo extrapolation to 5 million frames", va="top", fontsize=11, linespacing=1.8)
    fig.suptitle("Seaquest: partial loss traces for debugging only", fontsize=14)
    savefig(fig, "seaquest_partial_loss")


def main():
    for directory in (OUT, FIGURES, TABLES, AUDIT):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,"axes.spines.right":False})
    data = load_results()
    comparisons = result_figures(data)
    density_figures(data)
    from analysis.endpoints import analyze as analyze_endpoints, loss_diagnostics
    analyze_endpoints(data)
    attempts, bins, jobs = parse_logs(data)
    loss_diagnostics(bins)
    loss_figures(attempts, bins)
    summary = dict(result_files=len(data), completed_result_runs=sum(map(len,data.values())),
                   out_logs=len(jobs), err_logs=len({p for root in LOG_ROOTS for p in root.rglob("*.err")}),
                   unique_stdout_files=sum(not j["duplicate_stdout_of"] for j in jobs),
                   duplicate_stdout_files=sum(bool(j["duplicate_stdout_of"]) for j in jobs),
                   log_attempts=len(attempts), completed_log_attempts=sum(a["completed"] for a in attempts),
                   matching_log_attempts=sum(a["matches_json"] for a in attempts),
                   selected_unique_attempts=sum(a["selected"] for a in attempts),
                   logged_loss_samples=sum(len(a["loss"]) for a in attempts),
                   logged_eval_samples=sum(len(a["evaluation"]) for a in attempts),
                   incomplete_attempts=sum(not a["completed"] for a in attempts),
                   failed_shapiro_checks=sum(bool(r["shapiro_p"]<.05) for r in comparisons),
                   bootstrap_replicates=3000, bootstrap_seed=20260917,
                   versions=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__))
    (AUDIT/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
