# Repository cleanup — 17 September 2026

The cleanup preserves the proposal, training implementation, raw results, and
raw logs. It does not alter Git history, submit jobs, or rerun training.

| Previous location | Current location / action |
|---|---|
| Root `review_training_data.py` implementation | `analysis/review.py`; root command remains a compatibility wrapper |
| Root `analyse_results.py` | `archive/legacy_analysis/analyse_results.py`, preserved as historical code |
| `analysis_output/results_summary.csv` | `archive/legacy_analysis/results_summary.csv`, preserved as a historical report |
| Full `analysis_output/review/REVIEW.md` report | `docs/RESULTS_REVIEW.md`; old location links to the maintained report |
| Flat generated PNG/PDF/CSV/JSON files | `analysis_output/review/{figures,tables,audit}/` |
| Finder metadata and local Python bytecode | Removed from the working tree; the pre-existing root Finder file was backed up under `/private/tmp/rct_cleanup_*` |
| README and nontechnical guide | Updated to the actual CPU scripts, two-game current array, log sources, and restart behaviour |

Analysis dependencies are now listed in `requirements-analysis.txt`. Git ignore
rules cover downloaded `new_logs/`, generated analysis artifacts, and tool caches.
This follows the existing practice of keeping raw `results/` and `logs/` local.
Ignored files still need an independent backup; Git does not preserve them.

Both raw log locations remain in place. Duplicate stdout content is handled in
analysis through SHA-256 hashes, not by deleting raw files. Each source path and
its duplicate relationship is recorded in the generated `audit/log_jobs.csv`.

`analyze_rct.py` remains available for single-game statistics; its default now
uses the available Breakout results, and the help text reflects all completed
games. The statistical procedure is unchanged.

Validation: four parser regression tests cover copied log files, separate
restarted attempts, seed mismatches despite matching AUCs, and duplicate completed
individuals in distinct jobs. The full data workflow validates all 600 AUCs and
paired configurations and finds 587 matched completed loss histories. The report
explains the 13 missing histories. Generated plots are checked visually.
