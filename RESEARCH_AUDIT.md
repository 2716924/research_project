# n-step/PER implementation audit and final-return analysis

For the subsequent all-arm audit, RCT interpretation and new resume/preflight
safeguards, see [DESIGN_AND_IMPLEMENTATION_AUDIT.md](DESIGN_AND_IMPLEMENTATION_AUDIT.md).

Audit date: 20 September 2026. Source checkout before edits: `8297823`.
All performance estimates below describe the **original saved experiments**,
including the historical PER implementation. Fixing source code does not repair
previously trained agents or their scores.

## Findings

| Question | Evidence-based answer |
|---|---|
| Is n-step broken? | No return-construction or bootstrap bug was found in the exercised MinAtar path. Analytic tests cover n=1/3/5, short episodes, terminal tails and episode boundaries. Poor performance alone does not establish a bug. |
| Is PER correct? | A confirmed maximum-priority bookkeeping bug was reproduced and fixed. The other tested sampling/importance-weight calculations match their analytic definitions. The bug's contribution to the score deficit is not identified by existing runs. |
| Are these published hyperparameters? | Some treatment defaults match papers, but the complete training recipes do not. In particular, optimizer settings, loss, learning rates, environment and evaluation differ. |
| Can we analyze final return and DQN variance? | Yes. Added paired final-checkpoint and final-window analyses, distributions, confidence intervals and all-arm/DQN variability tables. There are 30 runs per game/arm, covering 29 distinct configurations. |

The dataset contains four completed games × five arms × 30 individuals = 600
completed runs. Each trajectory has 100 evaluation means. Pairing checks validate
IDs, training seeds and configurations; saved AUCs are recomputed. Complete,
deduplicated logs match 587 runs. In particular, Asterix PER loss diagnostics have
19/30 histories, although its return statistics use all 30 results. Seaquest has
no completed result JSON in this dataset and is excluded from endpoint tests.

## 1. n-step implementation

`NStepCollector` stores the discounted reward sum and the state/done flag at the
end of the window. `nstep_target` adds `gamma**3 * max Q_target` only when the
transition is nonterminal. `train_individual` flushes the collector before reset.

Tests compare every emitted transition against an independent direct calculation
over episodes of lengths 1, 2, 3, 4, 7 and 20, for n=1, 3 and 5. They check reward,
state, action, successor, terminal flag and no mixing across episodes. Separate
tests exercise a continuing three-step window, target discounting and masking.
An integration check sets n=1 and compares network parameters against control
through the actual MinAtar training loop (120 frames, reduced warm-up, three
parameter snapshots; evaluation is replaced by snapshot collection). This is a
mechanism check, not a benchmark of learned performance.

The fixed exponent on short **terminal** tails is harmless: their bootstrap is
zero. At the training-budget boundary, at most two pending n=3 transitions are
not emitted; this is not a plausible explanation for a 50–80% score deficit.

There is a latent portability issue: the trainer treats `truncated` as terminal.
The installed MinAtar adapter returns `truncated=False`, and the four registered
environments have no Gymnasium `TimeLimit`. Thus this is not an observed cause
here. If external time limits are added, continuation discounts and the actual
tail horizon must be stored, while still flushing at episode reset. The local
environment check also found separate observation arrays across steps, removing
one potential source of corruption in queued n-step states. Historical package
versions are not saved, so this cannot certify the exact cluster environment.

The paper cited in this repo studies **Mountain Car**, including n in
{1,3,5,10,20}; it is not a prediction of these MinAtar scores. Its n-step
Q-learning formulation also omits off-policy correction and explicitly discusses
the resulting policy mismatch. Rainbow uses n=3 but reports that n is sensitive,
and its ablations concern a combined agent on ALE Atari. Neither establishes
that standalone n-step DQN must match or beat DQN for every configuration.
[Hernandez-Garcia & Sutton, §§2.1, 4](https://arxiv.org/pdf/1901.07510),
[Hessel et al., Table 1 and ablations](https://arxiv.org/pdf/1710.02298).

For another implementation of discounted replay trajectories and episode
boundaries, see [Google Dopamine's replay buffer](https://github.com/google/dopamine/blob/master/dopamine/tf/replay_memory/circular_replay_buffer.py).
It is a comparison reference, not a drop-in replacement or an assertion of
identical training behavior.

## 2. Confirmed PER bug and interpretation of loss

Previously, `update_priorities` stored `(abs(TD error)+eps)**alpha` in both the
tree and `max_priority`. But `push` raised `max_priority` to `alpha` again.
Only the **initial priority of newly inserted transitions** had this double
exponent; updates to sampled transitions already used the correct exponent.

For an observed absolute TD error of 100 and alpha=0.6:

| Quantity | Historical implementation | Correct calculation |
|---|---:|---:|
| Updated transition tree mass | 15.8489 | 15.8489 |
| Next inserted transition tree mass | 5.2481 | 15.8489 |

The fix keeps `max_priority` in raw units and applies alpha once when storing
tree mass. This follows the bookkeeping in
[OpenAI Baselines' PER implementation](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py).
The regression test failed before the fix and passed afterwards. Git history
contains the defect in the initial and later buffer implementations. Result
files omit source hashes, so exact run-to-commit attribution is unavailable.

Other checks passed: a partially filled non-power-of-two tree samples according
to expected probabilities; importance weights match global-minimum
normalization; alpha=0 produces uniform priorities/unit weights; beta anneals
as intended; ring overwrite and large-capacity partial fill return valid data.
These tests increase confidence in the mechanisms exercised; they do not prove
all possible numerical corner cases or establish long-run learning performance.

**PER loss is a different measured objective.** The logged quantity is
`mean(weight * TD_error**2)`, whereas other arms log unweighted MSE on their
sampled batches. PER samples a different distribution, and beta changes during
training. The paper defines normalized importance weights and anneals beta to
1. [Schaul et al., §3.4](https://arxiv.org/pdf/1511.05952).

Writing raw priority as p, the code's normalized weight is
`(p_min / p_i)**(alpha*beta)` when the probability floor is inactive. For example,
if one stored transition has zero TD error, eps=1e-6, and another has error near
1, the latter's weight falls from about 0.0363 at beta=.4 to 0.000251 at beta=1,
even if both errors remain fixed. This is an algebraic illustration, **not a
measurement of those priorities in the saved runs**. RMSprop also rescales
gradients, so weighted-loss shrinkage is not directly an equivalent shrinkage
in parameter updates.

In the observed Space Invaders logs, the median PER loss falls from 0.02910 in
the first 500k frames to 0.000456 in the last 500k, while its AUC is 52.0% below
DQN. DQN's corresponding loss is 0.14396 → 0.16197. These summaries first take
the median of existing 50k-bin medians within each run, then the median across
runs. They demonstrate why loss magnitude cannot rank policy quality.

Old logs do not contain unweighted PER errors or importance weights, so the
respective contributions of improved fitting, weighting and sample composition
cannot be recovered. New training logs now include sampled unweighted MSE,
sampled absolute TD error, mean/minimum importance weight and beta. The
unweighted PER diagnostic still uses a prioritized batch; a future common
uniform diagnostic batch would be needed for a shared sampling distribution.

## 3. Hyperparameter comparison and what the current data can identify

| Setting | This repo | Relevant reference |
|---|---|---|
| PER alpha, beta | .6; .4 → 1 | Matches proportional PER in Schaul et al.; Rainbow uses alpha=.5 |
| PER learning rate | Same as paired DQN: 1e-4, 5e-4 or 1e-3 | Schaul et al. reduced the baseline rate fourfold for PER |
| n-step horizon | 3 | Rainbow uses 3; the multi-step study tests several horizons |
| Shared loss | Unclipped MSE | MinAtar's example uses Huber loss |
| RMSprop | PyTorch defaults: alpha=.99, eps=1e-8, centered=False, momentum=0 | MinAtar example: alpha=.95, eps=.01, centered=True |
| Shared learning rate | 1e-4, 5e-4, 1e-3 | MinAtar example: 2.5e-4, absent from the sampled population |
| Replay capacity | 10k, 50k, 100k | MinAtar example: 100k |
| Batch, target update | 32/64; 500/1000/2000 steps | MinAtar example: 32; 1000 |
| Warm-up, gamma, updates | 5000; .99; one update/step | Matches these MinAtar example settings |
| Budget and exploration floor | 5M steps; .1 | Matches these MinAtar example settings |

Sources: [original PER experiments, §4](https://arxiv.org/pdf/1511.05952),
[Rainbow hyperparameters](https://arxiv.org/pdf/1710.02298),
[MinAtar authors' DQN example](https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py).
The optimizer/loss differences are substantial candidates for interaction with
n-step and PER; they are **not established causes** of the current deficits.
The verified sources replace the earlier unverified `CleanRL rainbow_atari.py`
attribution in the training-code comments.

Matching alpha/beta/n alone therefore does not reproduce a published experiment.
The shared sampled configuration design estimates robustness under **this
population of settings**, rather than each method's separately tuned optimum.
Published ALE scores, Mountain Car returns and these fixed-seed MinAtar
evaluation AUCs are not numerically interchangeable.

The new `hyperparameter_subgroups.csv` describes paired effects at each sampled
setting. For n-step, Asterix and Freeway mean AUC is lower at **all three**
learning rates. On Freeway the subgroup deficits are −92.2%, −49.7% and −36.9%
at 1e-4, 5e-4 and 1e-3, respectively. The smallest rate is not a demonstrated
rescue. PER also loses on Freeway and Space Invaders in every learning-rate
subgroup. These are marginal, post-hoc groups of 7, 13 and 10 individuals;
other hyperparameters and seeds vary. Do not infer a causal rate effect or pick
a winning subgroup as a confirmatory result.

To separate causes, the next performance experiment should hold configurations,
seeds, evaluation and budget fixed while changing one factor:

1. Historical PER versus fixed PER, with identical original settings, isolates
   the code repair. Use separate output directories; never resume old results
   into a mixed old/fixed PER cohort.
2. Under fixed PER, compare baseline rate versus rate/4, then investigate alpha
   and priority epsilon separately. Include a DQN baseline for each setting.
3. Compare DQN (n=1), n=3 and n=5 under the existing recipe and a separate
   MinAtar-reference recipe: lr=2.5e-4, Huber, centered RMSprop (.95, eps=.01),
   buffer=100k, batch=32, target interval=1000, one hidden layer.
4. Repeat each selected configuration with multiple training seeds (at least
   five for an initial diagnostic), prioritizing Asterix/Freeway for n-step and
   Space Invaders for PER. Use the full 5M budget for comparable final-return
   claims; short pilots only diagnose failures or early learning.

This audit did not launch full-budget retraining or reproduce published scores.
The source bug is established; its performance cost and hyperparameter causes
remain questions for those controlled reruns. Poor outcomes are retained and
reported, rather than changing settings to force a desired ranking.

## 4. Final return, AUC and variability

**AUC remains primary.** It integrates evaluations from 50k through 5M frames,
an observed interval of 4.95M frames. **Final return** is the saved mean over ten
fixed-seed greedy evaluation episodes at exactly 5M frames. The separate
last-ten-checkpoints mean covers 4.55M–5M and is an exploratory sensitivity
summary. It is neither the final single policy nor ten independent runs.

Both new endpoints use the same 30 matched individuals, paired t-tests,
Shapiro diagnostics, paired d_z, win fractions and 98.75% t intervals as AUC.
Tables additionally contain paired percentile-bootstrap 95% intervals (20,000
resamples of individuals), explicitly descriptive rather than multiplicity-
adjusted confirmation. Per-game Bonferroni uses four treatments (.0125);
additional columns adjust 16 tests per endpoint and 32 across AUC + final
return. The exploratory final window is outside that 32-test family.

The final-checkpoint Shapiro diagnostic is below .05 in 9/16 comparisons.
Retain the planned paired tests, inspect distributions and intervals, and avoid
treating nonsignificance as equivalence. The bootstrap is a sensitivity check,
not a way to select a more favorable significance result.

### Relative treatment effects

Values are percent differences of arm means relative to DQN; * indicates the
planned per-game paired test passes p ≤ .0125 for that endpoint.

| Game | PER AUC | PER final return | n-step AUC | n-step final return |
|---|---:|---:|---:|---:|
| Breakout | +0.1% | +11.1% | +4.5% | −2.5% |
| Asterix | −15.8% | −40.8% | −80.0%* | −79.3%* |
| Freeway | −28.6%* | −8.9% | −59.4%* | −49.9%* |
| Space Invaders | −52.0%* | −52.2%* | −21.0%* | −14.7% |

At the final checkpoint, the four significant per-game comparisons are n-step
below DQN on Asterix (p=.001291) and Freeway (p=.000203), PER below DQN on
Space Invaders (p=.0000146), and Averaged DQN above DQN on Freeway (p=.009160).
Under the 32-test correction only the first three remain significant. The
last-ten-checkpoint analysis has different outcomes; for example, PER's Freeway
deficit is −22.0% (p=.001711), illustrating endpoint sensitivity.

### DQN baseline uncertainty

SD is the sample SD across 30 configuration–seed runs, not a standard error and
not uncertainty across the ten evaluation episodes. AUC below is in millions
of return × frames.

| Game | DQN AUC mean ± SD | DQN final return mean ± SD |
|---|---:|---:|
| Breakout | 65.21 ± 16.65 | 14.45 ± 9.53 |
| Asterix | 16.77 ± 14.12 | 4.21 ± 4.69 |
| Freeway | 159.81 ± 58.89 | 35.55 ± 17.17 |
| Space Invaders | 168.84 ± 90.31 | 36.89 ± 25.95 |

Each game/arm has 30 distinct training seeds and **29 distinct configurations**.
IDs 20 and 22 (zero-based; seeds 1357 and 1359) happen to share one configuration.
Thus there is one two-seed replicate, not a systematic repeated-seed design.
The exported variance mixes configuration sensitivity and training randomness;
it cannot reliably decompose them. The same seed/configuration cohort is used
across games, so 120 DQN game-runs are not 120 replicates of one environment.
Evaluation episode returns were reduced to means and their within-checkpoint
variance cannot be recovered from the JSONs.

## Outputs and reproduction

- [Final-return distributions](analysis_output/review/figures/final_return_distributions.png)
- [Paired final-return effects](analysis_output/review/figures/paired_final_return_effects.png)
- [Final-return density curves](analysis_output/review/figures/final_return_densities.png)
- [Final-return normality diagnostics](analysis_output/review/figures/paired_final_return_gaussian_diagnostics.png)
- [All endpoint summaries](analysis_output/review/tables/endpoint_summary.csv)
- [All paired endpoint tests](analysis_output/review/tables/paired_endpoint_comparisons.csv)
- [DQN variance, SD and intervals](analysis_output/review/tables/dqn_variability.csv)
- [Hyperparameter subgroup diagnostics](analysis_output/review/tables/hyperparameter_subgroups.csv)
- [Early/late loss diagnostics](analysis_output/review/tables/loss_early_late_diagnostics.csv)
- [Endpoint definitions, random seed and input hashes](analysis_output/review/audit/endpoint_analysis.json)

Each new figure also has a vector PDF beside the PNG. Reproduce the complete
analysis with `.venv/bin/python -m analysis.review`, or just the endpoint
analysis with `.venv/bin/python -m analysis.endpoints`.

Run the unit checks with `.venv/bin/python -m unittest discover -s tests -v`
and the existing arm smoke checks with `.venv/bin/python verify_arms.py`.
Validation completed: 14 unittest checks, 11 arm-verifier checks and all three
legacy replay checks passed. The full analysis regenerated successfully over
all 600 saved runs. The n=1 integration check matched DQN parameters exactly
at all three snapshots; PER's short training smoke check remained finite.
The older replay checks still use pytest-style functions; they were executed
directly because pytest is not installed locally. Their observation fixtures
and the arm verifier were updated for the existing array-backed replay API.
A postponed-annotations import fixes an unrelated Python 3.9 import failure in
`rct/slurm_utils.py` so the actual training-loop check can run locally.
