# Research figures and analysis outputs

Read the [methodology and literature review](../../docs/RESULTS_REVIEW.md).
See also the [implementation audit and final-return findings](../../RESEARCH_AUDIT.md).
Regenerate outputs with `python -m analysis.review` from the project root.

| Figure | PNG preview | Vector PDF |
|---|---|---|
| Evaluation learning curves | [PNG](figures/learning_curves.png) | [PDF](figures/learning_curves.pdf) |
| Individual AUC distributions | [PNG](figures/auc_distributions.png) | [PDF](figures/auc_distributions.pdf) |
| Paired AUC treatment effects | [PNG](figures/paired_auc_effects.png) | [PDF](figures/paired_auc_effects.pdf) |
| Final-return distributions | [PNG](figures/final_return_distributions.png) | [PDF](figures/final_return_distributions.pdf) |
| Paired final-return effects | [PNG](figures/paired_final_return_effects.png) | [PDF](figures/paired_final_return_effects.pdf) |
| Final-return density curves | [PNG](figures/final_return_densities.png) | [PDF](figures/final_return_densities.pdf) |
| Final-return normality diagnostics | [PNG](figures/paired_final_return_gaussian_diagnostics.png) | [PDF](figures/paired_final_return_gaussian_diagnostics.pdf) |
| Pretorius-style AUC density curves | [PNG](figures/pretorius_auc_densities.png) | [PDF](figures/pretorius_auc_densities.pdf) |
| Gaussian reference fits to paired differences | [PNG](figures/paired_auc_gaussian_diagnostics.png) | [PDF](figures/paired_auc_gaussian_diagnostics.pdf) |
| Breakout loss | [PNG](figures/breakout_loss.png) | [PDF](figures/breakout_loss.pdf) |
| Asterix loss | [PNG](figures/asterix_loss.png) | [PDF](figures/asterix_loss.pdf) |
| Freeway loss | [PNG](figures/freeway_loss.png) | [PDF](figures/freeway_loss.pdf) |
| Space Invaders loss | [PNG](figures/spaceinvaders_loss.png) | [PDF](figures/spaceinvaders_loss.pdf) |
| Partial Seaquest loss | [PNG](figures/seaquest_partial_loss.png) | [PDF](figures/seaquest_partial_loss.pdf) |

`tables/` contains run metrics, paired comparisons, evaluation trajectories,
density parameters, and binned losses. `audit/` contains source-file hashes,
attempt histories, missing-individual coverage, and validation counts. Generated
outputs are gitignored; source code and documentation are versioned.

New endpoint tables are `endpoint_summary.csv`, `paired_endpoint_comparisons.csv`,
`paired_final_return_comparisons.csv`, `paired_last_10_return_comparisons.csv`,
and `dqn_variability.csv`. Final return is the evaluation at 5M frames; the
last-ten mean is a separate sensitivity endpoint. Variability is across
configuration–seed individuals. `hyperparameter_subgroups.csv` is descriptive,
not a controlled hyperparameter experiment. `audit/endpoint_analysis.json`
records definitions, multiplicity, bootstrap settings and result hashes.

Loss figures use completed attempts matching the result JSONs. Current coverage
is 587/600: Asterix/PER has 19/30 histories; Breakout/Averaged and Freeway/Averaged
each have 29/30; other completed-game cells have 30/30. Result statistics still
use all 600 runs. See `audit/log_coverage.csv`.

Of the 13 missing complete histories, three have partial matching logs and ten
have no training history found. The [full repository investigation](../../docs/MISSING_LOG_INVESTIGATION.md)
records the exact individuals, files, and job transitions.

The Pretorius-style curves use Gaussian kernels and permit multiple peaks.
The separate normal-fit figure is an assumption diagnostic. Neither asserts that
the observed population is Gaussian.
