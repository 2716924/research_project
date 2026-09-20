"""Paired final-return analysis and descriptive configuration diagnostics.

Run independently with ``python -m analysis.endpoints`` or via analysis.review.
The final checkpoint is the primary secondary endpoint; the last ten checkpoint
mean is an explicitly exploratory sensitivity endpoint, not ten extra runs.
"""
import hashlib
import json

# Configure the headless plotting backend and writable caches before pyplot.
from analysis import review

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METRICS = {
    "auc": "AUC (return × frames)",
    "final_return": "Final evaluation return (5 million frames)",
    "last_10_return": "Mean of last 10 evaluations (4.55–5 million frames)",
}


def endpoint(run, metric):
    if metric == "auc":
        return float(run["auc"])
    if metric == "final_return":
        return float(run["eval_trajectory"][-1])
    if metric == "last_10_return":
        return float(np.mean(run["eval_trajectory"][-10:]))
    raise ValueError(metric)


def paired_arrays(control, treatment, metric):
    c = {r["individual_id"]: r for r in control}
    t = {r["individual_id"]: r for r in treatment}
    if len(c) != len(control) or len(t) != len(treatment) or c.keys() != t.keys():
        raise ValueError("Endpoints require unique, identical paired IDs")
    ids = sorted(c)
    for i in ids:
        if (c[i]["seed"], c[i]["config"]) != (t[i]["seed"], t[i]["config"]):
            raise ValueError(f"Mismatched seed/configuration for individual {i}")
    return (np.array([endpoint(c[i], metric) for i in ids]),
            np.array([endpoint(t[i], metric) for i in ids]))


def paired_summary(c, t, rng):
    d = t-c
    n = len(d)
    if n < 3 or not np.isfinite(d).all():
        raise ValueError("At least three finite pairs are required")
    sd = d.std(ddof=1)
    radius = stats.t.ppf(1-.0125/2, n-1) * sd/np.sqrt(n)
    test = stats.ttest_rel(t, c)
    boot = d[rng.integers(0, n, (20000, n))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return dict(n_pairs=n, mean_control=c.mean(), mean_treatment=t.mean(),
                mean_difference=d.mean(), sd_difference=sd,
                ci_98_75_low=d.mean()-radius, ci_98_75_high=d.mean()+radius,
                bootstrap_95_low=low, bootstrap_95_high=high,
                percent_change_of_means=100*d.mean()/c.mean() if c.mean() else np.nan,
                paired_dz=d.mean()/sd if sd else np.nan,
                t_statistic=test.statistic, p_value=test.pvalue,
                p_bonferroni_4=min(1., 4*test.pvalue),
                p_bonferroni_16=min(1., 16*test.pvalue),
                p_bonferroni_32_auc_and_final=min(1., 32*test.pvalue),
                significant_per_game=bool(test.pvalue <= .0125),
                shapiro_p=stats.shapiro(d).pvalue if sd else np.nan,
                matched_win_fraction=(d>0).mean(), matched_tie_fraction=(d==0).mean())


def final_figures(data, comparisons):
    rng = np.random.default_rng(20260920)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, game in zip(axes.flat, review.GAMES):
        values = [[endpoint(r, "final_return") for r in data[game, v]] for v in review.VARIANTS]
        ax.boxplot(values, showfliers=False)
        for i, (sample, color) in enumerate(zip(values, review.COLORS), 1):
            ax.scatter(i+rng.uniform(-.13, .13, len(sample)), sample, color=color, alpha=.65, s=18)
        ax.set_xticks(range(1, 6), review.LABELS, rotation=15, fontsize=8)
        ax.set(title=review.NAMES[game], ylabel="Final evaluation return")
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Final return at 5 million frames: 30 configuration–seed runs per arm\nEach point is one run's mean over 10 fixed evaluation episodes")
    review.savefig(fig, "final_return_distributions")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, game in zip(axes.flat, review.GAMES):
        rows = comparisons[(comparisons.game == game) & (comparisons.metric == "final_return")]
        for i, row in enumerate(rows.itertuples()):
            ax.errorbar(row.mean_difference, i, xerr=row.ci_98_75_high-row.mean_difference,
                        fmt="o", color=review.COLORS[i+1], capsize=4)
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_yticks(range(4), review.LABELS[1:])
        ax.invert_yaxis()
        ax.set(title=review.NAMES[game], xlabel="Treatment − DQN final return")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Secondary endpoint: paired final-return effects and 98.75% t intervals\nFour comparisons per game; normality diagnostics and wider multiplicity in tables")
    review.savefig(fig, "paired_final_return_effects")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, game in zip(axes.flat, review.GAMES):
        for i, (variant, color) in enumerate(zip(review.VARIANTS, review.COLORS)):
            values = np.array([endpoint(r, "final_return") for r in data[game, variant]])
            baseline = 4-i
            if values.std() > 0:
                kernel = stats.gaussian_kde(values)
                bw = np.sqrt(kernel.covariance[0, 0])
                grid = np.linspace(values.min()-3*bw, values.max()+3*bw, 700)
                density = kernel(grid)
                density *= .82/density.max()
                ax.fill_between(grid, baseline, baseline+density, color=color, alpha=.3)
                ax.plot(grid, baseline+density, color=color)
            ax.plot(values, np.full(len(values), baseline-.05), "|", color=color)
        ax.set_yticks(range(5), review.LABELS[::-1])
        ax.set(title=review.NAMES[game], xlabel="Final evaluation return", ylim=(-.2, 5))
    fig.suptitle("Final-return densities: Gaussian KDE, Scott bandwidth\nRows scaled to equal peak height; ticks show runs; smoothing can extend below zero")
    review.savefig(fig, "final_return_densities")

    fig, axes = plt.subplots(4, 4, figsize=(15, 12), layout="constrained")
    for i, game in enumerate(review.GAMES):
        for j, variant in enumerate(review.VARIANTS[1:]):
            c, t = paired_arrays(data[game, "control"], data[game, variant], "final_return")
            d = t-c
            ax = axes[i, j]
            ax.hist(d, bins="auto", density=True, color=review.COLORS[j+1], alpha=.3)
            if d.std(ddof=1) > 0:
                grid = np.linspace(min(d.min(), d.mean()-3*d.std()), max(d.max(), d.mean()+3*d.std()), 300)
                ax.plot(grid, stats.norm.pdf(grid, d.mean(), d.std(ddof=1)), color=review.COLORS[j+1])
            ax.axvline(0, color="grey", ls="--")
            ax.set(title=f"{review.NAMES[game]} · {review.LABELS[j+1]}\nShapiro p={stats.shapiro(d).pvalue:.3g}",
                   xlabel="Paired final-return difference", ylabel="Density")
    fig.suptitle("Normality diagnostics of paired final-return differences\nNormal reference curves do not establish Gaussian outcomes")
    review.savefig(fig, "paired_final_return_gaussian_diagnostics")


def analyze(data):
    rng = np.random.default_rng(20260920)
    summaries, comparisons, groups, coverage = [], [], [], []
    for game in review.GAMES:
        for variant in review.VARIANTS:
            runs = data[game, variant]
            configurations = [json.dumps(r["config"], sort_keys=True) for r in runs]
            counts = pd.Series(configurations).value_counts()
            coverage.append(dict(game=game, variant=variant, runs=len(runs),
                                 unique_seeds=len({r["seed"] for r in runs}),
                                 unique_configurations=len(counts),
                                 configurations_with_repeated_runs=int((counts>1).sum()),
                                 max_runs_per_configuration=int(counts.max())))
            for metric in METRICS:
                values = np.array([endpoint(r, metric) for r in runs])
                sd = values.std(ddof=1)
                radius = stats.t.ppf(.975, len(values)-1)*sd/np.sqrt(len(values))
                summaries.append(dict(game=game, variant=variant, metric=metric, n=len(values),
                                      mean=values.mean(), sd=sd, variance=sd**2,
                                      median=np.median(values), q25=np.quantile(values,.25),
                                      q75=np.quantile(values,.75), minimum=values.min(), maximum=values.max(),
                                      mean_ci_95_low=values.mean()-radius, mean_ci_95_high=values.mean()+radius))
                if variant == "control":
                    continue
                c, t = paired_arrays(data[game, "control"], runs, metric)
                comparisons.append(dict(game=game, variant=variant, metric=metric,
                                        **paired_summary(c, t, rng)))
                if variant not in {"per", "nstep"}:
                    continue
                # Marginal groups are descriptive: other settings and seeds vary.
                ordered = sorted(runs, key=lambda r: r["individual_id"])
                for parameter in ordered[0]["config"]:
                    settings = np.array([r["config"][parameter] for r in ordered])
                    for setting in np.unique(settings):
                        mask = settings == setting
                        difference = (t-c)[mask]
                        groups.append(dict(game=game, variant=variant, metric=metric,
                                           parameter=parameter, setting=setting, n_pairs=int(mask.sum()),
                                           mean_control=c[mask].mean(), mean_treatment=t[mask].mean(),
                                           mean_difference=difference.mean(),
                                           percent_change_of_means=100*difference.mean()/c[mask].mean()
                                           if c[mask].mean() else np.nan,
                                           win_fraction=(difference>0).mean()))
    comparisons = pd.DataFrame(comparisons)
    summary = pd.DataFrame(summaries)
    summary.to_csv(review.TABLES / "endpoint_summary.csv", index=False)
    summary[summary.variant == "control"].to_csv(review.TABLES / "dqn_variability.csv", index=False)
    comparisons.to_csv(review.TABLES / "paired_endpoint_comparisons.csv", index=False)
    for metric in ("final_return", "last_10_return"):
        comparisons[comparisons.metric == metric].to_csv(review.TABLES / f"paired_{metric}_comparisons.csv", index=False)
    pd.DataFrame(groups).to_csv(review.TABLES / "hyperparameter_subgroups.csv", index=False)
    pd.DataFrame(coverage).to_csv(review.AUDIT / "configuration_seed_coverage.csv", index=False)
    provenance = dict(bootstrap_seed=20260920, bootstrap_replicates=20000,
                      bootstrap_interval="95% percentile, paired differences; descriptive sensitivity only",
                      final_return="Mean of 10 fixed-seed episodes at frame 5000000",
                      last_10_return="Mean of 10 checkpoint means at frames 4550000 through 5000000",
                      variance="Across configuration–seed individuals, not seed-only variance",
                      multiplicity="4 per game/endpoint; 16 across games/endpoint; 32 across AUC + final return. Last-10 exploratory.",
                      result_sha256={str(p.relative_to(review.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((review.ROOT / "results").rglob("*.json"))})
    (review.AUDIT / "endpoint_analysis.json").write_text(json.dumps(provenance, indent=2)+"\n")
    final_figures(data, comparisons)
    print(f"Endpoint analysis: {len(summary)} arm summaries, {len(comparisons)} paired comparisons")
    return comparisons


def main():
    for directory in (review.TABLES, review.FIGURES, review.AUDIT):
        directory.mkdir(parents=True, exist_ok=True)
    analyze(review.load_results())


def loss_diagnostics(bins):
    """Describe available logs without interpreting weighted loss as return."""
    rows = []
    selected = bins[bins.selected]
    for (game, variant), subset in selected.groupby(["game", "variant"]):
        early = subset[subset.frame_bin_end <= 500000].groupby("individual_id").loss_median.median()
        late = subset[subset.frame_bin_end > 4500000].groupby("individual_id").loss_median.median()
        rows.append(dict(game=game, variant=variant, n=len(early),
                         early_loss_median=early.median(), late_loss_median=late.median(),
                         late_early_ratio=late.median()/early.median() if early.median() else np.nan))
    pd.DataFrame(rows).to_csv(review.TABLES / "loss_early_late_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
