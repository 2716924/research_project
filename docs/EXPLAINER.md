# Explaining the project

This project asks which individual changes to DQN improve learning across a
population of plausible training configurations. Each of 30 sampled
configurations is trained with all five algorithms, using the same seed for
matched runs.

| Group | Change being tested |
|---|---|
| DQN control | Uniform replay and a one-step target |
| Double DQN | Separate action selection from target evaluation |
| PER | Prioritize replay transitions using TD error |
| n-step DQN | Accumulate several rewards before bootstrapping |
| Averaged DQN | Use historical target networks to form the target |

The proposal specifies Breakout, Asterix, and Freeway. The saved results also
contain Space Invaders. Seaquest currently has partial logs without completed
result JSONs. The current cluster scripts target Seaquest and Space Invaders;
this differs from the original three-game collection.

Every 50,000 training frames, we evaluate the greedy policy on ten fixed-seed
episodes. Integrating these mean returns gives one AUC per run. Higher AUC means
better return over the observed interval; the curve shows details the single
number loses.

We compare each treatment's AUC with its matched control. The proposal calls for
normality diagnostics, paired t-tests, and Bonferroni correction. Effect size
and uncertainty matter alongside statistical significance.

The presentation combines learning curves, individual AUC distributions,
paired effects, and loss diagnostics. Pretorius-style density curves show the
spread of outcomes. Gaussian kernels in those curves do not mean the outcomes
have a normal distribution.

Training loss describes the network update objective. Evaluation return measures
policy performance. Their trends can differ, particularly when replay sampling
and target definitions differ between variants.

The [research review](RESULTS_REVIEW.md) gives findings and citations. The
[figure index](../analysis_output/review/README.md) links generated plots.
The [README](../README.md) gives current CPU cluster and analysis commands.
Resubmission preserves completed individuals; partial individuals restart and
are treated as separate attempts in the log audit.
