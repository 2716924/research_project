# Trusted reference implementations used for this project

The [20 September implementation audit](../RESEARCH_AUDIT.md) supersedes the
earlier assumption that these attributions alone establish correctness. It
records a reproduced PER bug, its fix, verified primary sources and differences
from published training recipes. Saved experiment results predate that fix.

This document records where each algorithmic mechanism was sourced.
Implementations were adapted to our RCT design (one change per treatment arm,
shared Nature-DQN topology, MinAtar environments).

## Control group — Nature DQN

| Mechanism | Paper | Reference code |
|-----------|-------|----------------|
| Experience replay + 1-step target | Mnih et al. 2015 | Original Nature DQN; our baseline in `rct/trainer.py` |

## Trusted implementation sources

The treatment arms were informed by community PyTorch implementations and
research papers. Key sources include public repositories for Double DQN,
Rainbow/PER, and Averaged-DQN. These sources validate the design choices used
in this project and are reflected in the code structure below.

## Treatment arms

### Double DQN (DDQN)

| Mechanism | Paper | Reference code |
|-----------|-------|----------------|
| Online selects action, target evaluates | Van Hasselt et al. 2016 | [DQN_Family_PyTorch](https://github.com/kmdanielduan/DQN_Family_PyTorch) `agent.py` `_optimize_model()` with `DOUBLE=True` |
| | | [Rainbow-DQN-pytorch](https://github.com/Lizhi-sjtu/Rainbow-DQN-pytorch) |

**Our file:** `rct/targets.py` → `double_dqn_target()`

### Prioritised Experience Replay (PER)

| Mechanism | Paper | Reference code |
|-----------|-------|----------------|
| Priority ∝ \|TD error\|^α | Schaul et al. 2015 | [DQN_Family_PyTorch](https://github.com/kmdanielduan/DQN_Family_PyTorch) `PrioritizedReplayMemory` |
| Segment-tree sampling and raw maximum-priority bookkeeping | OpenAI Baselines | [Official replay buffer](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py) |
| Importance-sampling weights | Schaul et al. 2016, §3.4 | β annealed 0.4 → 1.0 |

**Our files:** `rct/replay.py` → `PrioritizedReplayBuffer`; `rct/segment_tree.py`

**Fixed hyperparameters (not in shared population):** α=0.6, β_start=0.4, ε=1e-6

### n-step DQN

| Mechanism | Paper | Reference code |
|-----------|-------|----------------|
| n-step return before bootstrap | Hernandez-Garcia & Sutton 2019 | [Rainbow-DQN-pytorch](https://github.com/Lizhi-sjtu/Rainbow-DQN-pytorch) n-step buffer |
| Horizon choice | Hessel et al. 2018 | [Rainbow Table 1: n=3](https://arxiv.org/pdf/1710.02298) |

**Our file:** `rct/replay.py` → `NStepCollector`

**Fixed hyperparameter:** n=3

### Averaged DQN

| Mechanism | Paper | Reference code |
|-----------|-------|----------------|
| Average Q-values per action, then take the maximum | Anschel et al. 2017 | [Original Algorithm 2](https://proceedings.mlr.press/v70/anschel17a/anschel17a.pdf) |
| | | [YiSyuanChen/Averaged-DQN](https://github.com/YiSyuanChen/Averaged-DQN) |

**Our file:** `rct/targets.py` → `AveragedTargetEnsemble`

**Fixed hyperparameter:** K=10 historical target networks

## Evaluation methodology

| Requirement | Proposal section | Our implementation |
|-------------|------------------|-------------------|
| Fixed eval seeds | §4.3 | `rct/constants.py` → `EVAL_SEEDS` |
| Greedy eval (ε=0) | §4.3 | `rct/evaluation.py` |
| AUC over training frames | §4.4.1 | `rct/evaluation.py` → `compute_auc()` |
| Paired t-tests + Bonferroni | §4.4.2 | `analyze_rct.py` |

## Primary papers

For the presentation and analysis workflow, see [RESULTS_REVIEW.md](RESULTS_REVIEW.md).
Its distribution figure adapts Figure 3 of Pretorius et al. (2020),
[If dropout limits trainable depth, does critical initialisation still matter?](https://arxiv.org/pdf/1910.05725v2).
That figure shows density fits, including multimodal distributions. Our
Gaussian-kernel implementation is an adaptation; the separate normal-fit
diagnostics do not claim to reproduce the paper's statistical tests.

1. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
2. Van Hasselt, H. et al. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
3. Schaul, T. et al. (2015). Prioritized experience replay. *arXiv:1511.05952*.
4. Hernandez-Garcia, J. F. & Sutton, R. S. (2019). Understanding multi-step deep RL. *arXiv:1901.07510*.
5. Anschel, O. et al. (2017). Averaged-DQN. *ICML*.
