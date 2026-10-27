# followup-02 — follow-up runs after the first report draft

Single-run follow-ups requested in the lecturer's feedback (Oct 2026). Every run is one arm × game ×
individual for 5M frames, with evaluation every 50k frames on greedy seeds 1000–1009, as in the main trial.
The groups and the code that trained them are in `provenance/FOLLOWUP_README.md` and `provenance/followup.py`.

| Group | Runs | Status | What it answers |
|---|---|---|---|
| `recipe-sampled` | 42 | complete (23 Oct 2026) | DQN/PER/n-step on Asterix + Freeway under the Obando-Ceron & Castro (2021) recipe, on the 7 individuals with lr 1e-4 |
| `recipe-reference` | 40 | complete (23 Oct 2026) | Their published MinAtar DQN configuration, seeds 1337–1341 (positive control) |
| `lr-low-dqn` | 56 | complete (24 Oct 2026) | DQN on the same 7 individuals at lr 3e-5 and 1e-5 |
| `seaquest-capped` | 150 | complete (archived 27 Oct 2026) | All arms × 30 individuals on Seaquest; greedy episodes cut at 100k steps |
| `lr-low-arms` | 224 | complete (archived 27 Oct 2026) | The four treatments at lr 3e-5 and 1e-5 |

Current coverage: `.venv/bin/python followup.py status --root $PWD/Results/followup-02` (from the repo root).

## Layout

The run folders keep the cluster layout, so `followup.py status` and `followup.py collect` work on this folder.

```
followup-02/
├── <group>/results/<arm>/<game>/<tag>/<arm>_results_MinAtar_<Game>-v1.json   final results (source of truth)
├── <group>/tracking/<arm>/<game>/<tag>/attempt-<id>.jsonl, wandb/             loss/TD/IS-weight metrics, offline W&B
├── logs/rct-followup_<job>_<task>.{out,err}                                     SLURM stdout/stderr per array task
├── logs/rows/row<N>_<job>.log                                                   training log per manifest row
├── provenance/                                                                  code, tests and slurm scripts as deployed
├── scripts/archive_followup.py                                                  moves cluster output in; verifies checksums
└── tables/file_moves.csv                                                        old path → new path, SHA-256, bytes
```

Tags: `iNN` is individual NN (seed 1337 + NN); `iNN-lr3e-05` / `iNN-lr1e-05` is the same individual at a lower
learning rate. In `recipe-reference`, `i00`–`i04` are seeds 1337–1341 with the published configuration, not
the sampled individuals 0–4. `.row.lock` files are empty locks written by `followup.py run`.

## Provenance

- Code in `provenance/` is byte-identical to the repository's `rct/`, `followup.py`, `run_rct.py`,
  `verify_arms.py` and `slurm/` at the time of the run. Python 3.11.14, torch 2.13.0, MinAtar 1.0.15, the same
  stack as `corrected-rct-01`.
- The main-trial partners of these runs (shared recipe, lr 1e-4) are in `corrected-rct-01/` and
  `original-run-2026-09/`; `analysis/followup02.py` asserts that each follow-up run has the same seed and
  sampled configuration as its partner.

## Integrity

Files were moved, never edited, merged or deleted. `scripts/archive_followup.py move` checksums every file
before and after its move and appends it to `tables/file_moves.csv`; first batch: 1,625 files from
`dist/followup/followup/` on 24 Oct 2026; second batch: 4,303 files from `followup/` on 27 Oct 2026, cross-checked against an independent `shasum -a 256` manifest
taken before the move. Re-check at any time:

```sh
python3 Results/followup-02/scripts/archive_followup.py verify
```

## Adding later output (seaquest-capped, lr-low-arms)

Unzip `followup-output.zip` into its own empty folder outside `Results/`, then:

```sh
python3 Results/followup-02/scripts/archive_followup.py move <folder> --dry-run
python3 Results/followup-02/scripts/archive_followup.py move <folder>
python3 Results/followup-02/scripts/archive_followup.py verify
```

Files already archived with the same SHA-256 (the earlier runs, repeated in the zip) are left in the
folder and reported as duplicates; if any archived file differs, nothing is moved.

## Analysis

`.venv/bin/python -m analysis.followup02` → `analysis_output/followup-02/tables/` (`runs.csv`, `paired.csv`,
`gap_change.csv`). Groups without results are skipped.
