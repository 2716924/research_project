#!/usr/bin/env python3
"""
Follow-up experiments requested after the first report draft (Oct 2026).

Each manifest row is ONE training run (arm, game, individual) and is trained by
run_rct.py into its own result directory, so rows can run in parallel. Groups,
in priority order:

  recipe-sampled    DQN, PER, n-step under the Obando-Ceron & Castro (2021)
                    recipe on the seven lr=1e-4 individuals; Asterix, Freeway.
                    Paired with the existing shared-recipe runs of the same
                    individuals, so the recipe is the only difference.
  recipe-reference  Their published MinAtar DQN configuration (5 seeds) with
                    DQN, PER, n-step on Asterix and Freeway, plus DQN on
                    Breakout and Space Invaders as a sanity check of DQN.
  lr-low-dqn        DQN on the seven lr=1e-4 individuals at lr 3e-5 and 1e-5.
  seaquest-capped   All five arms x 30 individuals on Seaquest, with greedy
                    evaluation episodes truncated at the step limit.
  lr-low-arms       The four treatments at lr 3e-5 and 1e-5 (optional).

Commands:
  python followup.py tasks  [--per-task P]           SLURM array ranges per group
  python followup.py rows TASK [--per-task P]        rows trained by one array task
  python followup.py run ROW --root DIR [...]        train one row (used by SLURM)
  python followup.py status --root DIR [--per-task P]
  python followup.py pending TASK --root DIR [--per-task P]   exit 0 if TASK has unfinished rows
  python followup.py prepare --root DIR              create every output folder before sbatch
  python followup.py collect --root DIR --out runs.csv
"""

import argparse
import csv
import fcntl
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from rct.constants import GAMES, REFERENCE_CONFIGS, VARIANTS

ROOT = Path(__file__).resolve().parent
GAME = {g.split("/")[1].removesuffix("-v1").lower(): g for g in GAMES}
# Every individual whose sampled learning rate is 1e-4 (sample_individuals(30)).
LR_1E4_IDS = [12, 13, 18, 20, 22, 26, 28]
LOW_LRS = [3e-5, 1e-5]
DEFAULT_PER_TASK = 4


def _row(group, variant, game, individual_id, recipe="shared", eval_limit="raise", overrides=None):
    tag = f"i{individual_id:02d}"
    if overrides and "lr" in overrides and group.startswith("lr-low"):
        tag += f"-lr{overrides['lr']:g}"
    return dict(group=group, variant=variant, game=GAME[game], slug=game,
                individual_id=individual_id, recipe=recipe, eval_limit=eval_limit,
                overrides=overrides or {}, tag=tag)


def manifest():
    """All rows, grouped, in priority order. Row order is fixed: never reorder
    or edit groups once an experiment has started (row ids name the runs)."""
    groups = {}
    groups["recipe-sampled"] = [
        _row("recipe-sampled", v, g, j, recipe="obando2021")
        for g in ("asterix", "freeway") for j in LR_1E4_IDS for v in ("control", "per", "nstep")
    ]
    ref = REFERENCE_CONFIGS["obando2021"]
    groups["recipe-reference"] = [
        _row("recipe-reference", v, g, j, recipe="obando2021", overrides=ref)
        for g in ("asterix", "freeway") for j in range(5) for v in ("control", "per", "nstep")
    ] + [
        _row("recipe-reference", "control", g, j, recipe="obando2021", overrides=ref)
        for g in ("breakout", "spaceinvaders") for j in range(5)
    ]
    groups["lr-low-dqn"] = [
        _row("lr-low-dqn", "control", g, j, overrides={"lr": lr})
        for lr in LOW_LRS for g in ("breakout", "asterix", "freeway", "spaceinvaders") for j in LR_1E4_IDS
    ]
    groups["seaquest-capped"] = [
        _row("seaquest-capped", v, "seaquest", j, eval_limit="truncate")
        for j in range(30) for v in VARIANTS
    ]
    groups["lr-low-arms"] = [
        _row("lr-low-arms", v, g, j, overrides={"lr": lr})
        for lr in LOW_LRS for g in ("breakout", "asterix", "freeway", "spaceinvaders")
        for j in LR_1E4_IDS for v in VARIANTS if v != "control"
    ]
    rows = [row for group in groups.values() for row in group]
    for i, row in enumerate(rows):
        row["row"] = i
    return rows


def task_chunks(per_task):
    """Array task id -> row ids. Tasks never mix groups, so each group can be
    submitted on its own with the ranges printed by `tasks`."""
    if per_task < 1:
        raise ValueError("--per-task must be positive")
    chunks, current, group = [], [], None
    for row in manifest():
        if current and (row["group"] != group or len(current) == per_task):
            chunks.append((group, current))
            current = []
        group = row["group"]
        current.append(row["row"])
    chunks.append((group, current))
    return chunks


def run_dirs(root, row):
    base = Path(root) / row["group"]
    rel = Path(row["variant"]) / row["slug"] / row["tag"]
    return base / "results" / rel, base / "tracking" / rel


def ensure_dir(path, attempts=10):
    """Create a directory on shared storage, retrying: simultaneous creation of the
    same folder from several nodes can fail transiently ("mkdir: Already exists")."""
    path = Path(path)
    for attempt in range(attempts):
        if path.is_dir():
            return
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if attempt == attempts - 1:
                raise
            print(f"Retrying creation of {path}: {exc}", flush=True)
            time.sleep(random.uniform(1, 5))
    if not path.is_dir():
        raise OSError(f"Could not create {path}")


def result_file(root, row):
    from rct.results import result_path
    return Path(result_path(str(run_dirs(root, row)[0]), row["variant"], row["game"]))


def completed_run(root, row):
    path = result_file(root, row)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    runs = [r for r in data.get("results", []) if r["individual_id"] == row["individual_id"]]
    return runs[0] if runs else None


def cmd_tasks(args):
    spans = {}
    for task, (group, _) in enumerate(task_chunks(args.per_task)):
        spans.setdefault(group, [task, task])[1] = task
    counts = {}
    for row in manifest():
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(f"{'group':18} {'runs':>5}  sbatch --array (RCT_RUNS_PER_TASK={args.per_task})")
    for group, (lo, hi) in spans.items():
        print(f"{group:18} {counts[group]:5}  {lo}-{hi}")


def cmd_rows(args):
    chunks = task_chunks(args.per_task)
    if not 0 <= args.task < len(chunks):
        sys.exit(f"Array task {args.task} out of range 0-{len(chunks) - 1}")
    print(" ".join(str(r) for r in chunks[args.task][1]))


def cmd_run(args):
    rows = manifest()
    if not 0 <= args.row < len(rows):
        sys.exit(f"Row {args.row} out of range 0-{len(rows) - 1}")
    row = rows[args.row]
    if not os.path.isabs(args.root):
        sys.exit("--root must be an absolute path on shared storage")
    out_dir, tracking_dir = run_dirs(args.root, row)
    ensure_dir(out_dir)
    ensure_dir(tracking_dir)
    lock = open(out_dir / ".row.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Row {args.row} is already running ({out_dir})")
    print(f"Row {args.row}: {json.dumps({k: v for k, v in row.items() if k != 'row'})}", flush=True)
    cmd = [sys.executable, "-u", str(ROOT / "run_rct.py"),
           "--variant", row["variant"], "--game", row["game"],
           "--individual-ids", str(row["individual_id"]),
           "--recipe", row["recipe"], "--eval-limit", row["eval_limit"],
           "--total-frames", str(args.total_frames), "--eval-frequency", str(args.eval_frequency),
           "--num-eval-episodes", str(args.num_eval_episodes),
           "--output-dir", str(out_dir), "--tracking-dir", str(tracking_dir),
           "--wandb-mode", os.environ.get("WANDB_MODE", "offline"),
           "--experiment-id", f"{os.environ.get('RCT_EXPERIMENT_ID', 'followup')}/{row['group']}"]
    if row["overrides"]:
        cmd += ["--config-override", json.dumps(row["overrides"])]
    sys.exit(subprocess.run(cmd).returncode)


def cmd_status(args):
    rows = manifest()
    done = {row["row"] for row in rows if completed_run(args.root, row) is not None}
    by_group = {}
    for row in rows:
        total, finished = by_group.get(row["group"], (0, 0))
        by_group[row["group"]] = (total + 1, finished + (row["row"] in done))
    for group, (total, finished) in by_group.items():
        print(f"{group:18} {finished:4}/{total:<4} complete")
    missing = [task for task, (_, ids) in enumerate(task_chunks(args.per_task))
               if any(i not in done for i in ids)]
    if missing:
        print(f"Array tasks with unfinished rows (RCT_RUNS_PER_TASK={args.per_task}): "
              + ",".join(map(str, missing)))
    else:
        print("All rows complete.")


def cmd_prepare(args):
    """Create all output folders up front (on the login node), so jobs that start at the
    same moment never race to create a shared folder."""
    if not os.path.isabs(args.root):
        sys.exit("--root must be an absolute path on shared storage")
    ensure_dir(ROOT / "logs" / "rct-followup" / "rows")
    rows = manifest()
    for row in rows:
        for directory in run_dirs(args.root, row):
            ensure_dir(directory)
    print(f"Prepared {2 * len(rows)} run folders under {args.root} and logs/rct-followup/rows")


def cmd_pending(args):
    """Exit 0 while the array task still has unfinished rows (used for auto-resubmission)."""
    chunks = task_chunks(args.per_task)
    if not 0 <= args.task < len(chunks):
        sys.exit(f"Array task {args.task} out of range 0-{len(chunks) - 1}")
    rows = manifest()
    unfinished = [i for i in chunks[args.task][1] if completed_run(args.root, rows[i]) is None]
    print("Unfinished rows: " + (" ".join(map(str, unfinished)) or "none"))
    sys.exit(0 if unfinished else 1)


def cmd_collect(args):
    fields = ["row", "group", "variant", "game", "individual_id", "seed", "recipe", "eval_limit",
              "lr", "buffer_capacity", "batch_size", "target_update_freq", "epsilon_decay_steps",
              "num_layers", "replay_start_size", "auc", "last10_mean", "final_return",
              "truncated_eval_episodes", "result_file"]
    out_rows = []
    for row in manifest():
        run = completed_run(args.root, row)
        if run is None:
            continue
        trajectory = run["eval_trajectory"]
        out_rows.append(dict(
            row=row["row"], group=row["group"], variant=row["variant"], game=row["slug"],
            individual_id=row["individual_id"], seed=run["seed"], recipe=row["recipe"],
            eval_limit=row["eval_limit"], **run["config"], auc=run["auc"],
            last10_mean=sum(trajectory[-10:]) / len(trajectory[-10:]), final_return=trajectory[-1],
            truncated_eval_episodes=sum(run.get("eval_truncated_episodes", [])),
            result_file=str(result_file(args.root, row).relative_to(args.root)),
        ))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} completed runs to {args.out}")
    _print_summary(out_rows)


def _print_summary(out_rows):
    """Mean AUC (1e6 units) per group/game/arm and paired differences to DQN."""
    cells = {}
    for r in out_rows:
        key = (r["group"], r["game"], r["lr"])
        cells.setdefault(key, {}).setdefault(r["variant"], {})[r["individual_id"]] = r["auc"] / 1e6
    for (group, game, lr), arms in sorted(cells.items()):
        print(f"\n{group} / {game} / lr={lr:g}")
        control = arms.get("control", {})
        for variant in VARIANTS:
            if variant not in arms:
                continue
            values = list(arms[variant].values())
            line = f"  {variant:9} n={len(values):2} mean AUC {sum(values) / len(values):8.2f}"
            pairs = [arms[variant][j] - control[j] for j in arms[variant] if j in control]
            if variant != "control" and len(pairs) >= 2:
                mean = sum(pairs) / len(pairs)
                sd = math.sqrt(sum((d - mean) ** 2 for d in pairs) / (len(pairs) - 1))
                line += f" | vs DQN: mean diff {mean:+8.2f}, wins {sum(d > 0 for d in pairs)}/{len(pairs)}"
                try:
                    from scipy import stats
                    line += f", paired t p={stats.ttest_1samp(pairs, 0).pvalue:.3g}" if sd > 0 else ""
                except ImportError:
                    pass
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    per_task = dict(type=int, default=int(os.environ.get("RCT_RUNS_PER_TASK", DEFAULT_PER_TASK)))
    p = sub.add_parser("tasks")
    p.add_argument("--per-task", **per_task)
    p = sub.add_parser("rows")
    p.add_argument("task", type=int)
    p.add_argument("--per-task", **per_task)
    p = sub.add_parser("run")
    p.add_argument("row", type=int)
    p.add_argument("--root", required=True)
    p.add_argument("--total-frames", type=int, default=5_000_000)
    p.add_argument("--eval-frequency", type=int, default=50_000)
    p.add_argument("--num-eval-episodes", type=int, default=10)
    p = sub.add_parser("status")
    p.add_argument("--root", required=True)
    p.add_argument("--per-task", **per_task)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p = sub.add_parser("pending")
    p.add_argument("task", type=int)
    p.add_argument("--root", required=True)
    p.add_argument("--per-task", **per_task)
    p = sub.add_parser("collect")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"tasks": cmd_tasks, "rows": cmd_rows, "run": cmd_run,
     "status": cmd_status, "pending": cmd_pending, "prepare": cmd_prepare, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
