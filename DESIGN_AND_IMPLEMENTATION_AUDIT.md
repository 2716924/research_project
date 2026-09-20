# Follow-up: RCT design, published settings and all-arm verification

20 September 2026. This extends [the first implementation audit](RESEARCH_AUDIT.md).
It separates methodological interpretation, current implementation checks and
what can be inferred about historical experiments.

## Does the hyperparameter design contradict the algorithm papers?

No. A population comparison and a tuned benchmark estimate different things.
Your proposal §4.1 explicitly asks whether a modification improves results
across a shared population of configurations. It intentionally includes both
strong and poor settings. Table 4.1 specifies the actual learning rates, replay
sizes, batch sizes, target intervals, exploration schedules and layer counts
used by the sampler. They are your study's choices, not numerical settings
prescribed by Pretorius for DQN.

Pretorius et al. §3.1 samples network designs and copies them across intervention
groups. The study concerns supervised ReLU networks and initialization, not DQN
hyperparameters. Appendix A balances design-component levels by concatenating
random permutations. [Pretorius et al., §3.1 and Appendix A](https://www.raillab.org/publication/pretorius-2020-if/pretorius-2020-if.pdf).

The repo implements **independent uniform draws** from your discrete population,
as specified in proposal §4.1.1. The same 30 configuration/seed pairs are copied
to all five arms, so each arm sees exactly the same sample, not just a similar
distribution on average. This is a paired computational experiment; the
randomness is in sampling configurations and training, rather than assigning
different configurations exclusively to different treatments.

The effect being estimated for a game is:

\[
\Delta_A(P)=\mathbb{E}_{h\sim P,\,s}
\left[Y(A,h,s)-Y(\mathrm{DQN},h,s)\right].
\]

Here h is a shared configuration, s denotes training randomness, P is the chosen
configuration population, and Y is AUC or the specified return endpoint. With
one seed for most configurations, the sample mixes both sources of variability.
This is a population-specific effect. There is no hyperparameter-independent
algorithm ranking that this design can recover automatically.

A method may benefit at its tuned settings and lose on average under P without
either experiment being wrong. Different environments, budgets, optimizers,
losses and evaluation policies also matter. Conversely, merely finding these
differences does not establish that they caused the observed deficits. The
confirmed PER defect remains a separate implementation issue.

The original Double-DQN paper itself includes a controlled comparison using
DQN's existing hyperparameters, alongside a separately tuned version. Sharing
settings is therefore not inherently contrary to algorithm evaluation.
[Van Hasselt et al., evaluation and Table 2](https://arxiv.org/pdf/1509.06461).

### Sampling limitation to disclose

This sampler is inspired by Pretorius but does not reproduce its balancing
procedure. In this 30-run cohort:

| Parameter | Values | Counts |
|---|---|---|
| Learning rate | .0001 / .0005 / .001 | 7 / 13 / 10 |
| Replay capacity | 10k / 50k / 100k | 15 / 5 / 10 |
| Batch size | 32 / 64 | 15 / 15 |
| Target interval | 500 / 1000 / 2000 | 8 / 11 / 11 |
| Hidden layers | 1 / 2 | 16 / 14 |

The finite sample is uneven, especially for replay capacity, although the
sampler is uniform in expectation. These imbalances do not create an imbalance
between treatment and control, because pairing is exact. They do limit coverage
and precision. Proposal §5.2 also envisages approximate marginal stratification
where possible; report that this cohort used independent sampling. Do not
retroactively change the cohort or discard poor configurations to match a
published ranking. A larger or balanced sample belongs in a separately defined
follow-up experiment. The full count table is
[hyperparameter_coverage.csv](analysis_output/review/audit/hyperparameter_coverage.csv).

## Which follow-up strategies are compatible with the RCT?

The following is an interpretation of your stated estimand and matched design:

| Strategy | What it estimates / how to report it |
|---|---|
| Same sampled settings for every arm | Retains the current question: effect of the algorithmic change across the stated population. |
| Fix PER's priority defect and repeat the same cohort | Corrects the intervention implementation. Keep historical results and corrected results explicitly separate. |
| Compare old and corrected PER with all settings fixed | Diagnostic effect of the code repair, not a hyperparameter comparison. |
| Change optimizer/loss for every arm in a new experiment | A second shared training protocol; compare within that protocol and examine algorithm-by-protocol interactions. |
| Set PER's rate to one quarter of each paired base rate | A defined package intervention: PER plus its rate rule. Valid as a separate arm, but its effect cannot be attributed solely to replay sampling. |
| Compare n=1, 3 and 5 | Explicit additional treatments/horizon sensitivity, with matched configurations and a declared comparison family. |
| Tune each method separately | A separate best-achievable-performance study; use comparable tuning budgets and fresh evaluation seeds/runs after selection. It is not the existing component-only contrast. |
| Select favorable configurations after inspecting results | Exploratory only; it cannot replace the original confirmatory population estimate. |

Additional sensitivity experiments are allowed by controlled experimental
reasoning, but they do not retroactively become part of the original protocol.
State the intervention, configuration population, seed replication, endpoint
and multiplicity before interpreting the new results. Independent evaluation
after tuning is discussed in [Patterson et al., hyperparameter selection](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf).

Suggested dissertation wording:

> We estimate the effect of each DQN modification over a prespecified shared
> hyperparameter population, rather than reproducing each method's separately
> tuned benchmark performance. Differences from published rankings therefore
> concern robustness under our experimental conditions and do not, by
> themselves, refute the original findings.

Use this wording only for correctly implemented arms. The historical PER
results must additionally disclose the identified implementation defect.

## Current implementation audit

| Arm | Mechanism verified | Qualification |
|---|---|---|
| DQN | Uniform replay, target-network maximum, terminal masking, periodic target synchronization, detached targets | A MinAtar-adapted DQN under the shared MSE/RMSprop recipe, not an exact Nature/ALE replication. |
| Double DQN | Online network chooses the action; target network evaluates that action; terminal rows ignore the bootstrap | A constructed example with conflicting online/target argmaxes detects swapped roles or accidental ordinary DQN. |
| PER | One priority exponent, normalized importance weights, annealing, priority-biased sampling and valid ring-buffer entries | Previous double-exponent initialization defect fixed; performance after correction still needs full-budget experiments. |
| n-step | Discounted reward windows, successor alignment, episode tails, gamma-to-the-horizon bootstrap | n=1 exactly matches DQN in the tested short training loop. External nonterminal truncations are now rejected explicitly. |
| Averaged target DQN | Mean Q-value per action across snapshots, then maximum; snapshot deep copy, bounded history, oldest eviction and target-update timing | Target averaging is correct. The shared behavior/evaluation policy remains the online network. |

All targets are computed without gradients through their bootstrap networks.
Integration checks exercise every dispatch branch and verify that target
parameters synchronize at the expected boundary without acquiring gradients.
The architecture has no dropout or batch normalization. Evaluation now still
explicitly enters eval mode and restores the prior mode, preventing that
assumption from silently breaking if the architecture is extended later.

### Averaged-DQN documentation discrepancy

Proposal §2.4.4 (printed p. 9, PDF p. 18) writes a mean of per-network maxima.
The original algorithm takes the maximum after averaging action values. The
current implementation uses the latter, correct ordering. For snapshots
Q1=[10,0], Q2=[0,10], the correct next-state value is 5, whereas averaging maxima
gives 10. A dedicated regression test distinguishes them.
[Anschel et al., Algorithm 2](https://proceedings.mlr.press/v70/anschel17a/anschel17a.pdf).

There is a further scope difference to document: Algorithm 2 outputs an averaged
Q function, while this project's shared evaluator uses the current online
network. Keeping the evaluator shared is consistent with isolating the target
change in proposal Table 4.2; call this the **Averaged-DQN target intervention**
rather than claiming an exact reproduction of the complete published agent.
The snapshot warm-up uses the available history, with a single-target fallback
before the first sync. These choices are now explicit in the audit/metadata.

The existing proposal PDF has been preserved. Its equation should be corrected
in the editable dissertation source. Code comments and reference notes now state
the actual averaging order.

## Additional defects and protections

1. **Unvalidated resumption:** previously an output filename and individual ID
   were enough to skip training. Source changes, budgets, evaluation schedules
   and even different cohorts could go undetected. New training resumes compare
   saved protocol, source hashes, package/Python versions, thread count,
   evaluation metadata, seeds and configurations. Legacy results remain
   readable for analysis, but cannot be silently resumed under new code.
2. **Completion boundary:** stopping before the last interaction could return
   `completed=True` for budgets whose last loop index is a time-check boundary.
   Completion now requires exhausting the loop. This is not evidence that the
   saved standard 5M-frame experiments suffered that edge case.
3. **Invalid or partial results:** saving/resuming now validates unique IDs,
   full evaluation count, finite returns, and AUC agreement. JSON serialization
   rejects NaN/Infinity before replacing an existing file. Trial budgets must
   include an evaluation at the final frame.
4. **Numerical failures:** nonfinite losses, gradients and evaluation Q-values
   or returns stop the run instead of silently yielding misleading results.
   Finite but poor or unstable learning remains possible and requires empirical
   diagnostics; these checks do not classify every learning failure as a bug.
5. **Unexpected time limits:** a nonterminal truncation fails explicitly because
   this MinAtar protocol does not store the continuation discount/horizon needed
   for correct truncated n-step bootstraps. Ordinary MinAtar terminations still
   work. Supporting other wrappers would require an explicit extension.
6. **Nonterminating greedy evaluation:** a real CLI smoke check hung on Seaquest.
   A no-op policy stays at the surface with full oxygen and no terminal signal;
   this was reproduced for 10,000 steps. Evaluation previously had no limit.
   The new 100,000-step safety guard raises an error, closes the environment and
   records no partial score. The limit and failure behavior are saved in the
   protocol. A real Seaquest regression uses a shorter limit to reproduce the
   failure quickly. Choosing to score capped episodes instead would define a
   different evaluation protocol and must be explicit. This is relevant to
   future Seaquest jobs, not an explanation of completed four-game returns.
7. **Preflight checks:** `run_rct.py` invokes `verify_arms.py` in a subprocess
   before training. The verifier now runs the original arm checks plus the
   algorithm and result-integrity regressions. Failures abort the job. The child
   process keeps test RNG/thread changes out of the real experiment.

Optimizer defaults, epsilon settings and gamma are now recorded explicitly in
`TRAINING_SETTINGS`. Their numerical values retain the previous recipe. These
changes do not silently switch to a different paper's tuned settings.

For new corrected experiments, use a **fresh output directory**. A source or
dependency change will intentionally block resuming a fingerprinted run; this
is conservative and can also reject harmless source edits. Historical files
are not modified or assigned invented provenance. Metadata cannot authenticate
unrecorded historical cluster code, nor guarantee identical floating-point
behavior across all hardware.

## Verification and remaining limits

The full unittest discovery passes 30 tests. The training preflight runs 11
original arm checks plus 23 algorithm/integrity unittest checks, including the
regression that fails on the old PER priority formula.

An additional real-environment smoke check trained each of the five arms for
600 Breakout steps using a reduced replay warm-up, performed actual fixed-seed
evaluations, saved results and resumed them with strict validation. All five
completed with finite returns and passed round-trip validation. The exact
configuration, results and source hashes are in
[implementation_smoke_checks.json](analysis_output/review/audit/implementation_smoke_checks.json).
This test is intentionally too short to assess convergence or comparative
performance. The first separate CLI check exposed the Seaquest evaluation hang
and was interrupted without writing a result. A Space Invaders CLI check
exercises automatic preflight and result writing outside the research results
directory after the safeguards were added.

The evidence supports: **the tested mechanisms are correct, the identified
defects are fixed, and known regressions are checked before new CLI training.**
It does not support “there are no remaining bugs” or “these changes guarantee
published performance.” Proposal §5.2 calls for learning/convergence and Q-value
validation on small-scale trials as well as unit tests. The smoke checks satisfy
execution checks, not that stronger convergence criterion. Full-budget matched
reruns and replicated pilot learning curves are still needed to measure the
PER fix's effect and investigate the n-step deficit.
