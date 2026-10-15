# Missing loss-history investigation — 18 September 2026

The full repository search found **no additional complete loss histories**.
The correct description of the 13 gaps is **three partial histories and ten
individuals with no training history found**. All 13 have completed evaluation
trajectories in their result JSONs. The coverage table counts complete matching
loss histories, not every appearance of an individual in a log.

## Search scope

- Inventoried the entire working tree, including hidden and ignored paths:
  220 project files, 48,912 virtual-environment files, and 358 Git files at the
  start of the search. No filesystem traversal errors occurred.
- Searched project file contents for training loss/evaluation lines regardless
  of extension. All 68 real stdout files were already included in the audit;
  the only other match was synthetic test data in `tests/test_analysis_logs.py`.
- Checked archive signatures as well as names. The only project archives are
  the two generated CSV.gz analysis tables; no extra raw-log archive was found.
- Inspected file symlinks; these point to Python executables, not additional
  research-log directories. Archives inside the virtual environment are package
  fixtures and timezone/sample data.
- Searched history across all locally available Git refs for `.out`, `.err`,
  `.log`, `.zip`, `.tar`, `.gz`, and `.tgz` paths; none were recorded. There
  are no local stashes. No remote fetch or cluster search was performed.

## Evidence for the three partial histories

Console individual numbers below start at 1; JSON IDs start at 0.

| Run | Partial stdout | Last logged frame | Evaluation points matching saved JSON | Following job |
|---|---|---:|---:|---|
| Breakout / Averaged, individual 9 (ID 8, seed 1345) | `slurm/logs/by-variant/averaged/rct-bigbatch_30221_0.out`, individual begins at line 40843 | 1,860,000 / 5,000,000 | 37 / 100 | `rct-bigbatch_41929_0.out`, line 42 reports 9 completed; line 51 skips individual 9 |
| Freeway / Averaged, individual 8 (ID 7, seed 1344) | `slurm/logs/by-variant/averaged/rct-bigbatch_30221_2.out`, individual begins at line 35743 | 4,610,000 / 5,000,000 | 92 / 100 | `rct-bigbatch_41930_2.out`, line 42 reports 8 completed; line 50 skips individual 8 |
| Asterix / PER, individual 16 (ID 15, seed 1352) | `slurm/logs/by-variant/per/rct-bigbatch_23274_1.out`, individual begins at line 76543 | 700,000 / 5,000,000 | 14 / 100 | `rct-bigbatch_24102_1.out`, line 42 reports 26 completed; line 58 skips individual 16 |

The seeds, configurations, evaluation frames, and every available evaluation
score in these three partial attempts match the corresponding result JSON
prefixes. Each earlier file explicitly says that the individual is incomplete
because of the time limit. These are not complete histories rejected because
of a formatting or matching issue.

## The other ten individuals

For **Asterix / PER individuals 17–26** (JSON IDs 16–25; seeds 1353–1362),
the complete repository search found no training-attempt history. Lines 59–68
of `slurm/logs/by-variant/per/rct-bigbatch_24102_1.out` skip them as already
completed, and line 70 starts individual 27.

Thus the available earlier Asterix log finishes with 15 completed individuals
and a partial individual 16, while the next available log starts with 26
completed. The output covering that increase is absent from this checkout.
The analogous increases are 8 to 9 completed for Breakout/Averaged and 7 to 8
completed for Freeway/Averaged.

## Interpretation and retrieval targets

The saved results support that those runs completed. They do not contain the
missing minibatch losses, so the losses cannot be reconstructed from them.
The local evidence does not establish why the intervening output is absent
(for example, a missing download or overwritten job output); that remains
unresolved.

The most useful place to check next is the cluster's original job-output
directory and any backups, including earlier versions or other outputs around
these transitions:

- Averaged Breakout: `30221_0` to job `41929`, individual 9.
- Averaged Freeway: `30221_2` to job `41930`, individual 8.
- PER Asterix: `23274_1` to job `24102`, individuals 16–26.

Those are job transitions evidenced by the logs, not asserted filenames for an
unknown missing job. Full-history coverage remains **587/600**; a further three
individuals have partial matching traces. Current full-run loss figures keep
their original sample counts.

Machine-readable evidence is in
[`missing_log_investigation.csv`](../analysis_output/review/audit/missing_log_investigation.csv)
and [`repository_search.json`](../analysis_output/review/audit/repository_search.json).
