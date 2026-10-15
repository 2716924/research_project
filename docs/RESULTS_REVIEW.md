# Training results, log extraction, and proposal alignment

Review date: 17 September 2026. Inputs: `results/`, `slurm/logs/`, all subdirectories of `new_logs/`, and `logs/` when present, `Research_Proposal.pdf`, and the local training/analysis code. Findings concern the files currently present; this is not a full audit of algorithm implementations or cluster execution history.

## 1. What is available

The results contain **20 JSON files and 600 completed training runs**: five variants × four games × 30 individuals. The games are Breakout, Asterix, Freeway, and Space Invaders. Each run contains 100 evaluation means, one every 50,000 frames up to 5,000,000 frames. Each evaluation mean represents 10 fixed-seed greedy episodes. The JSON files also preserve individual ID, training seed, hyperparameters, and AUC.

The statistical unit is a **sampled configuration–seed individual**, with five paired algorithm outcomes. These are not 30 seed replicates at one fixed configuration. The ten evaluation episodes and 100 checkpoints must not be treated as independent training replicates.

The log scan includes **`slurm/logs/by-variant/`**, which holds the Breakout, Asterix, and Freeway histories, and `new_logs/`, which holds the newer Space Invaders and partial Seaquest histories. Duplicate stdout files across directories are identified by SHA-256 and parsed once, while every source path remains in the audit. There are no completed Seaquest JSON results in `results/`. The earlier review missed the ignored `slurm/logs/` directory; the coverage statements and figures here supersede that review.

The combined scan finds **68 stdout files (57 unique contents)** and 57 stderr files. Eleven stdout copies are byte-identical across directories and are parsed once. The unique logs contain **3,026,992 loss samples**, 60,576 evaluation samples, and 626 attempts: 590 completed and 36 partial. Three completed Space Invaders control individuals (IDs 27–29) appear in distinct jobs; selecting one matching attempt per individual leaves **587 completed loss histories**.

All 600 saved AUCs match recomputation and all paired seeds/configurations match. All 590 completed log attempts match their result JSONs. There are no nonfinite or negative losses; 136 samples print as zero and remain in the tables (log-scale plots mask zero values).

| Game | DQN | DDQN | PER | n-step | Averaged DQN |
|---|---:|---:|---:|---:|---:|
| Breakout | 30 | 30 | 30 | 30 | 29 |
| Asterix | 30 | 30 | 19 | 30 | 30 |
| Freeway | 30 | 30 | 30 | 30 | 29 |
| Space Invaders | 30 | 30 | 30 | 30 | 30 |

Counts above are **matched completed loss histories**, out of 30 result individuals in every cell. Missing histories: Breakout/Averaged ID 8; Asterix/PER IDs 15–25; Freeway/Averaged ID 7 (JSON IDs are zero-based). Result analyses still use all 30 individuals per cell. Loss plots with missing histories describe the available subset and may be affected by missingness; they must not be described as the full population. Seaquest has one partial individual per variant, separately plotted; five associated stderr files report time-limit cancellation.

## 2. Recommended figures, with literature support

**Follow-up repository search (18 September):** the 13 gaps mean missing
*complete* matching histories. Three have partial matching logs; ten have no
training history found. A search including ignored paths, differently named
files, archives, and locally available Git history found no additional complete
logs. See [the evidence and retrieval targets](MISSING_LOG_INVESTIGATION.md).

The layouts and metric choices below adapt published evaluation principles to this paired experiment; they are not claims that the cited papers used this exact experimental design.

| Priority | Figure | What it answers | Literature basis |
|---|---|---|---|
| Main text | Four-panel evaluation learning curves: one game per panel; five algorithm lines; uncertainty bands | How quickly does each variant learn, and does performance persist? | Young & Tian (2019), Fig. 3, plots return against training frames over 30 runs. Their curves use tuned step sizes and SEM; ours average the sampled population and use bootstrap intervals. [MinAtar paper](https://arxiv.org/pdf/1903.03176) |
| Main text | Paired AUC effect plot: treatment minus DQN, with zero reference and confidence intervals | How large is each modification's effect across matched individuals? | Patterson et al. (2024), §4.4 and Fig. 12, motivate visualizing paired differences; Ho et al. (2019) motivate effect sizes with uncertainty. A horizontal forest-style layout is our adaptation. [Empirical design](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf), [Estimation graphics](https://www.nature.com/articles/s41592-019-0470-3) |
| Main text | AUC dot-and-box plots, all 30 points visible | Is the mean driven by a few unusually successful configurations? | Distribution-aware reporting is supported by Agarwal et al. (2021); displaying observations alongside effects follows estimation graphics. [Statistical Precipice](https://arxiv.org/abs/2108.13264), [Ho et al.](https://www.nature.com/articles/s41592-019-0470-3) |
| Main text or appendix | Stacked AUC density curves, one row per algorithm | Where is the population concentrated, and are there several performance modes? | Adapted from Pretorius et al. (2020), Figure 3. We use Gaussian-kernel density estimation with Scott bandwidth and show all 30 observations as rug ticks. [Figure 3, preprint PDF p. 8](https://arxiv.org/pdf/1910.05725v2) |
| Diagnostic figure or appendix | Loss against frames, separate panels by variant, median and IQR | Are there spikes, plateaus, or unusual optimization behaviour? | This is a diagnostic recommendation based on the logged objective. Fujimoto et al. (2022) show that Bellman error can be a poor proxy for value accuracy; it should not substitute for policy evaluation. [Bellman-error study](https://arxiv.org/abs/2201.12417) |
| Optional appendix | Paired return-difference curves over training | When does a treatment help or hurt relative to its matched control? | Directly motivated by Patterson et al. (2024), Fig. 12. Pointwise intervals do not license repeated significance claims at every checkpoint. [Figure 12](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf) |
| Optional robustness figure | Per-game performance profiles: fraction of individuals exceeding an AUC threshold | How reliable is each variant across the sampled population? | Agarwal et al. (2021), §4.2, propose performance profiles. Here the distribution includes sampled hyperparameters, not just training randomness. [Performance profiles](https://arxiv.org/html/2108.13264v4#S4.SS2) |
| Optional exploratory figure | Paired AUC effects grouped by learning rate, replay size, or layer count | Which configurations appear to benefit? | Young & Tian (2019), Fig. 2, demonstrate hyperparameter-sensitivity plots. Our random sample is not their controlled sweep: show subgroup counts and avoid causal claims from small cells. [Figure 2](https://arxiv.org/pdf/1903.03176) |

Keep consistent variant colours and game ordering. Separate games because their score scales differ. Use an explicit legend distinguishing **confidence intervals** (uncertainty in a mean) from **IQR** (spread of individuals). Avoid a single raw-score average across games, unexplained difficulty rankings, or a best-run-only figure.

A cross-game IQM could be an exploratory supplement, after defining defensible per-game normalization. Agarwal et al. recommend IQM and bootstrap intervals, but that does not replace the proposal's paired mean-AUC estimand. No comparable random/human reference scores are supplied here. If aggregating games, preserve repeated individual IDs across games during resampling and state whether inference concerns these fixed games or a wider task population. [IQM and uncertainty](https://arxiv.org/abs/2108.13264)

The requested Gaussian-style figure is based on **Pretorius et al., Figure 3** (arXiv v2, PDF page 8): stacked density fits for experimental groups. The paper explicitly notes multimodal outcomes and uses rank-based nonparametric tests. It does not justify forcing the outcomes into a single bell curve. Our adaptation uses **Gaussian kernels** to estimate the AUC density, preserving multiple modes; the rug marks retain the observations. Bandwidth is Scott's rule, and each row is scaled to the same peak height so broad distributions remain visible beside narrow ones; compare horizontal position and spread, not vertical density heights. Smoothing tails below zero are displayed as a property of the estimator, not as negative observed AUCs. With only 30 observations, the precise density shape is exploratory. [Pretorius et al. (2020), §4 and Fig. 3](https://arxiv.org/pdf/1910.05725v2)

A separate diagnostic figure overlays a fitted normal density on each histogram of **paired AUC differences**, labelled with Shapiro p-values. It checks the assumption relevant to the proposal's paired t-tests. These normal reference curves are our addition; they are not a reproduction of the paper's statistical procedure. The proposal's paired t-tests remain the declared primary analysis.

## 3. Making the logs useful

Convert the console stream into three linked tables instead of presenting raw text:

| Table | Key fields | Purpose |
|---|---|---|
| Jobs | source file, header job ID, game, variant, completion count, timeout, warning, resubmission | Concise execution and coverage audit |
| Attempts | source file, individual ID, seed, configuration, attempt ID, completion marker, final frame, JSON match | Preserve retries without counting them as new individuals |
| Time series | attempt ID, frame, minibatch loss, epsilon, evaluation return | Produce curves and diagnostic summaries |

Read the game and variant from the log header, not just the filename or array index. Console individual numbers are one-based; JSON IDs are zero-based. A printed 'Saved results' line alone does not prove that the current individual completed.

The training code saves completed individuals, but it does **not** checkpoint a partial individual's network, optimizer, replay buffer, or RNG state. An interrupted individual restarts from the beginning. Therefore, do not concatenate its partial attempt with the next attempt into one curve. Keep retries in the audit, then select one completed attempt whose configuration, seed, AUC, evaluation frames, and evaluation trajectory match the JSON. Retain duplicate matches and their provenance in the audit.

The current loss line is one minibatch's MSE, sampled approximately every 1,000 frames; it is not the average of all intervening updates. For PER it is importance-weighted MSE. Different replay distributions and targets make absolute loss levels unsuitable for ranking algorithms. Values are printed to six decimal places, so zero may reflect rounding. No validation loss is recorded.

The example presentation uses non-overlapping **50,000-frame bins**. First calculate the median logged loss within each run/bin; then plot the median across runs with an IQR band. Use log-scaled loss axes and identify separate y-scales. Save zero/nonfinite counts and the within-bin 95th percentile so median smoothing does not hide spikes. Each completed-game loss plot includes only deduplicated, JSON-matching runs, with the available sample count printed in each panel. Seaquest partial curves are a separate diagnostic with no population uncertainty band or extrapolation.

Additional useful summaries are final frame, evaluation coverage, loss median/p95/max, zero or nonfinite counts, and interruption status. Timing is limited: the logs lack per-individual and per-event timestamps. Job start/end times can support job elapsed-time reporting, but exact per-run speed, GPU utilization, memory use, episode lengths, or reward components cannot be recovered from these logs.

For future runs, emit JSONL or CSV events containing run UUID, attempt UUID, individual ID, variant, game, frame, optimizer step, timestamp, and metric. Record window mean/median/p95 loss and sample counts, unweighted TD-error diagnostics alongside PER's weighted objective, per-episode evaluation returns, package versions, and git commit. Keep console output to periodic progress and warnings. These are recommendations; this review does not alter training.

## 4. What the proposal says to do with training data

Page references below give printed page first, then PDF viewer page.

| Proposal location | Planned use of data | Current position |
|---|---|---|
| §4.1–4.1.2, pp. 15–16 / PDF 24–25 | Sample at least 30 configurations; apply all five algorithms to each | 30 IDs per game/arm. The review checks that paired IDs have identical saved seeds and configurations. |
| §4.2.1, pp. 16–18 / PDF 25–27 | Evaluate Breakout, Asterix, and Freeway | All three have full results. Space Invaders is an additional environment; document the extension. Seaquest has partial logs only. |
| §4.3, p. 18 / PDF 27 | Train for 5 million frames; evaluate every 50,000 frames using ten fixed-seed greedy episodes; store their mean | Metadata and trajectories match this schedule. Individual episode returns were reduced to a mean and cannot now be recovered. |
| §4.4.1, pp. 18–19 / PDF 27–28 | Integrate each evaluation trajectory to produce AUC as the primary sample-efficiency measure | AUC is stored and recomputed from the saved trajectory by the review. |
| §4.4.2, p. 19 / PDF 28 | For each treatment and game, compute individual differences: treatment AUC minus control AUC | Pair by ID and validate seed/configuration; obtain 30 differences per comparison. |
| §4.4.2, p. 19 / PDF 28 | Shapiro–Wilk on differences, followed by paired t-tests; Bonferroni alpha = 0.05/4 = 0.0125 | The review exports these diagnostics, p-values, adjusted p-values, effect sizes, and intervals. |
| §3.2–3.4, pp. 13–14 / PDF 22–23 | Assess algorithmic efficacy and expose the limitations of isolated high scores | Present population distributions and matched effects alongside curves. Final-window scores are secondary summaries. |

The corresponding analysis is:

1. Validate completeness and pairing; keep the original inputs intact.
2. Use the unsmoothed evaluation trajectory to calculate one AUC per individual/arm/game.
3. Calculate the 30 paired differences against control.
4. Report mean difference, interval, paired standardized effect (d_z), Shapiro diagnostic, t statistic, and p-value.
5. Apply the declared multiple-comparison family. A significant **positive** difference supports improvement; a significant negative difference indicates deterioration.
6. Interpret the effect sizes and distributions in relation to the research questions, including limitations and inconclusive comparisons.

Several details need explicit wording in the dissertation:

- **AUC starts at 50,000 frames.** The current trapezoidal calculation covers 50,000–5,000,000, not 0–5,000,000. There is no frame-zero evaluation. Describe the observed interval; do not invent an initial score. Dividing AUC by 4,950,000 gives time-averaged evaluation return over that observed interval, not normalization across games.
- **Normality is not guaranteed by n=30.** Shapiro tests the distribution of differences, not the sampling distribution of their mean, and a nonsignificant result does not prove normality. Inspect Q–Q plots and influential pairs. Preserve the planned test and consider a paired-bootstrap sensitivity analysis; do not silently switch to whichever test gives significance. Wilcoxon has different assumptions and does not generally test the same mean-effect estimand.
- **Multiplicity across games is unspecified.** The stated 0.0125 threshold handles four treatments within one game. A single family containing the three planned games has 12 comparisons (0.05/12); all four games give 16 (0.003125). The review labels its primary calculations as per-game and also exports adjustment across 16 comparisons. Decide and explain the inferential family before making broad claims.
- **Confidence levels must match their purpose.** The effect plots use 98.75% two-sided t intervals, corresponding to four per-game comparisons at familywise 5%, under the t-test assumptions. Learning-curve bootstrap intervals are pointwise descriptive 95% intervals, not simultaneous bands.
- **Fixed evaluation seeds standardize evaluation conditions; they do not eliminate every source of uncertainty or establish generalization to unseen seeds.** One training seed per sampled configuration also prevents separating configuration sensitivity from within-configuration seed variation.
- **AUC does not separately identify stability.** Similar AUCs can arise from different learning dynamics. Keep the learning curves; any collapse or volatility metric needs its own definition and should be marked exploratory.
- **Research question 2 needs more than a t-test.** A within-dataset demonstration can compare final-window rankings, best-checkpoint rankings, and AUC rankings, then show how uncertain rankings become when subsampling matched individuals. It illustrates sensitivity; it does not independently prove estimator reliability. AUC cannot be compared numerically with published ALE score gains as though they were the same outcome.

## 5. Existing analysis scripts

The preliminary paired analysis illustrates what these figures reveal:

| Game | Result under the proposal's per-game threshold |
|---|---|
| Breakout | No treatment–control AUC difference passes 0.0125. This does not establish equivalence. |
| Asterix | Averaged DQN has higher mean AUC (+57.5%); n-step has lower mean AUC (−80.0%). |
| Freeway | Averaged DQN has higher mean AUC (+20.9%); PER (−28.6%) and n-step (−59.4%) have lower mean AUC. |
| Space Invaders | Averaged DQN (+19.5%) and DDQN (+11.4%) have higher mean AUC; PER (−52.0%) and n-step (−21.0%) have lower mean AUC. |

Percentages are the difference of arm means divided by the control mean, not the mean of per-individual percentage changes. These are preliminary t-test results, not final claims about algorithmic superiority. Shapiro p < 0.05 occurs for n-step on Breakout, Asterix, and Freeway, and DDQN on Space Invaders. The detailed CSV preserves all p-values and assumptions diagnostics.

The Space Invaders loss figure provides a particularly useful teaching example: PER's loss declines markedly while its mean AUC is about 52% below DQN's. This is an observation from these runs; it does not identify the cause. The weighted objective, replay distribution, and changing targets need to be considered before attributing the pattern to a mechanism.

`analyze_rct.py` is closer to the proposal: paired AUC tests, Shapiro, and Bonferroni. It currently matches IDs without validating seed/configuration equality, omits effect-size intervals, and needs explicit handling of incomplete or very small paired samples.

The archived `archive/legacy_analysis/analyse_results.py` tests **last-ten-checkpoint scores with unpaired Welch tests** at 0.05. Its current game list is Seaquest and Space Invaders, excluding the three proposal games. It also contains hard-coded expected rankings, approximate literature scores, and difficulty categories that should not be presented as verified literature findings. The archived `archive/legacy_analysis/results_summary.csv` contains older three-game summaries and is not the primary analysis required by §4.4.2.

The legacy script and its historical CSV have been archived together; `analyze_rct.py` remains the single-game statistics entry point. `analysis/review.py` is the main multi-game figure and log-analysis workflow. The training data and algorithm implementation are preserved.

## 6. Reproduce and use the outputs

From the project root:

```sh
.venv/bin/python -m analysis.review
```

Dependencies: NumPy, SciPy, pandas, Matplotlib, listed separately in `requirements-analysis.txt`. Install with `python -m pip install -r requirements-analysis.txt`. The script is intentionally scoped to the reviewed 30-individual, 5-million-frame dataset; incompatible inputs fail validation rather than being silently pooled.

- [Learning curves](../analysis_output/review/figures/learning_curves.png) and [vector PDF](../analysis_output/review/figures/learning_curves.pdf).
- [AUC distributions](../analysis_output/review/figures/auc_distributions.png) and [vector PDF](../analysis_output/review/figures/auc_distributions.pdf).
- [Paired AUC effects](../analysis_output/review/figures/paired_auc_effects.png) and [vector PDF](../analysis_output/review/figures/paired_auc_effects.pdf).
- [Breakout loss](../analysis_output/review/figures/breakout_loss.png), [Asterix loss](../analysis_output/review/figures/asterix_loss.png), and [Freeway loss](../analysis_output/review/figures/freeway_loss.png).
- [Pretorius-style AUC density curves](../analysis_output/review/figures/pretorius_auc_densities.png) and [Gaussian diagnostics of paired differences](../analysis_output/review/figures/paired_auc_gaussian_diagnostics.png).
- [Space Invaders loss](../analysis_output/review/figures/spaceinvaders_loss.png) and [vector PDF](../analysis_output/review/figures/spaceinvaders_loss.pdf).
- [Seaquest partial diagnostics](../analysis_output/review/figures/seaquest_partial_loss.png) and [vector PDF](../analysis_output/review/figures/seaquest_partial_loss.pdf).
- `tables/individual_metrics.csv`: all 600 run summaries, including separate final-checkpoint and last-ten-checkpoint means.
- `tables/paired_auc_comparisons.csv`: all 16 treatment comparisons and diagnostics.
- `tables/evaluation_curves.csv.gz`: 60,000 evaluation observations in long format.
- `audit/log_jobs.csv`, `audit/log_attempts.csv`, `audit/log_coverage.csv`: provenance, coverage, retries, completion and matching.
- `tables/loss_bins.csv.gz`: compact per-attempt loss bins, retaining selection flags.
- `audit/audit_summary.json`: validation counts and numerical-library versions.
- `audit/proposal_text.txt`: extracted proposal text with PDF page markers.

The learning curves, paired effects, density distributions, and loss diagnostics form a compact presentation. Keep partial-run diagnostics and detailed job audits in the appendix. Read the normality and execution-provenance findings alongside the preliminary statistical output before treating it as a final result.
