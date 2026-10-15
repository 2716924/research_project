"""Build plot-ready CSV tables for corrected-rct-01.

Reads results/, logs/ and tracking/ without modifying them and (re)writes
everything in tables/ except file_moves.csv. Standard library only:

    python3 Results/corrected-rct-01/scripts/build_tables.py
"""
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
TABLES = EXP / "tables"
ARMS = ["control", "ddqn", "per", "nstep", "averaged"]
GAMES = ["breakout", "asterix", "freeway", "spaceinvaders", "seaquest"]
CONFIG_KEYS = ["lr", "buffer_capacity", "batch_size", "target_update_freq",
               "epsilon_decay_steps", "num_layers", "replay_start_size"]
METRIC_KEYS = ["train/loss", "train/epsilon", "train/sampled_mse", "train/sampled_abs_td",
               "train/is_weight_mean", "train/is_weight_min", "train/beta", "eval/mean_return"]


def rel(p):
    return p.relative_to(EXP).as_posix()


def write_csv(name, header, rows, compress=False):
    path = TABLES / name
    opener = gzip.open if compress else open
    with opener(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        n = 0
        for row in rows:
            w.writerow(row)
            n += 1
    print(f"{rel(path)}: {n} rows")


def load_results():
    results = {}
    for path in sorted((EXP / "results").glob("*/*/*_results_*.json")):
        arm, game = path.parts[-3], path.parts[-2]
        results[arm, game] = (path, json.loads(path.read_text()))
    return results


def runs_rows(results):
    for (arm, game), (path, d) in sorted(results.items()):
        for r in sorted(d["results"], key=lambda r: r["individual_id"]):
            yield [arm, game, d["game_id"], r["individual_id"], r["seed"], r["attempt_id"],
                   r["auc"], r["final_return"], len(r["eval_trajectory"])] + \
                  [r["config"].get(k) for k in CONFIG_KEYS] + [rel(path)]


def curve_rows(results):
    for (arm, game), (_, d) in sorted(results.items()):
        freq = d["eval_frequency"]
        for r in sorted(d["results"], key=lambda r: r["individual_id"]):
            for i, value in enumerate(r["eval_trajectory"]):
                # Matches rct/evaluation.py:compute_auc — evaluation i happens at frame (i + 1) * eval_frequency.
                yield [arm, game, r["individual_id"], r["seed"], i, (i + 1) * freq, value]


def auc_wide_rows(results):
    by_pair = defaultdict(dict)
    for (arm, game), (_, d) in results.items():
        for r in d["results"]:
            by_pair[game, r["individual_id"], r["seed"]][arm] = r["auc"]
    for (game, ind, seed), aucs in sorted(by_pair.items(), key=lambda kv: (GAMES.index(kv[0][0]), kv[0][1])):
        yield [game, ind, seed] + [aucs.get(a, "") for a in ARMS]


def parse_log(out_path):
    text = out_path.read_text(errors="replace")
    err_path = out_path.with_suffix(".err")
    err = err_path.read_text(errors="replace").strip() if err_path.exists() else ""

    def g(pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""

    cluster, array_job, task = re.match(r"rct-five-games-([AB])_(\d+)_(\d+)", out_path.stem).groups()
    status = g(r"Status:\s+(.+?)\s+\(exit")
    if not text.strip():
        status = "DID NOT START"
    return [f"cluster-{cluster}", array_job, g(r"Job ID:\s+(\S+)"), int(task),
            out_path.parts[-3], out_path.parts[-2], g(r"Node:\s+(\S+)"),
            g(r"Started:\s+(.+)"), g(r"Finished:\s+(.+)"), g(r"Wall elapsed:\s+(.+)"),
            status, g(r"\(exit (\d+)\)"), "Saved results" in text,
            err.splitlines()[-1][:200] if err else "", rel(out_path), rel(err_path)]


def log_rows():
    paths = sorted((EXP / "logs").glob("*/*/*.out"))
    rows = [parse_log(p) for p in paths]
    return sorted(rows, key=lambda r: (ARMS.index(r[4]), GAMES.index(r[5]), r[7] or "~", r[1]))


def read_attempt(path):
    start, finish, n_metrics = {}, {}, 0
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["event"] == "start":
                start = e["config"]
            elif e["event"] == "finish":
                finish = e
            else:
                n_metrics += 1
    return start, finish, n_metrics


def attempt_rows(results):
    final_ids = {r["attempt_id"] for _, d in results.values() for r in d["results"]}
    for path in sorted((EXP / "tracking").glob("*/*/attempt-*.jsonl")):
        start, finish, n = read_attempt(path)
        attempt_id = path.stem.removeprefix("attempt-")
        yield [path.parts[-3], path.parts[-2], attempt_id, start.get("individual_id"), start.get("seed"),
               start.get("slurm_job_id"), start.get("slurm_array_task_id"), finish.get("status", "no finish event"),
               finish.get("completed_frames", ""), finish.get("auc", ""), finish.get("final_return", ""),
               attempt_id in final_ids, n, rel(path)]


def metric_rows():
    for path in sorted((EXP / "tracking").glob("*/*/attempt-*.jsonl")):
        arm, game, attempt_id = path.parts[-3], path.parts[-2], path.stem.removeprefix("attempt-")
        ind = seed = None
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e["event"] == "start":
                    ind, seed = e["config"].get("individual_id"), e["config"].get("seed")
                elif e["event"] == "metrics":
                    yield [arm, game, attempt_id, ind, seed, e.get("frames")] + [e.get(k, "") for k in METRIC_KEYS]


def coverage_rows(results, logs, attempts):
    jobs = defaultdict(list)
    for row in logs:
        jobs[row[4], row[5]].append(row[10])
    tracked = defaultdict(int)
    for row in attempts:
        tracked[row[0], row[1]] += 1
    for arm in ARMS:
        for game in GAMES:
            path, d = results.get((arm, game), (None, None))
            n = len(d["results"]) if d else 0
            statuses = jobs.get((arm, game), [])
            if n == 30:
                state = "complete"
            elif n:
                state = "partial"
            elif any(s.startswith("FAILED") for s in statuses):
                state = "failed"
            else:
                state = "missing"
            yield [arm, game, state, n, len(statuses), "; ".join(statuses), tracked[arm, game],
                   rel(path) if path else ""]


def main():
    TABLES.mkdir(exist_ok=True)
    results = load_results()
    logs = log_rows()
    attempts = list(attempt_rows(results))

    write_csv("coverage.csv", ["arm", "game", "status", "n_individuals", "n_slurm_jobs", "slurm_job_statuses",
                               "n_tracked_attempts", "result_file"], coverage_rows(results, logs, attempts))
    write_csv("runs.csv", ["arm", "game", "game_id", "individual_id", "seed", "attempt_id", "auc", "final_return",
                           "n_evals"] + CONFIG_KEYS + ["result_file"], runs_rows(results))
    write_csv("auc_wide.csv", ["game", "individual_id", "seed"] + [f"auc_{a}" for a in ARMS], auc_wide_rows(results))
    write_csv("learning_curves.csv", ["arm", "game", "individual_id", "seed", "eval_index", "frame", "eval_return"],
              curve_rows(results))
    write_csv("slurm_jobs.csv", ["cluster", "array_job_id", "job_id", "array_task", "arm", "game", "node", "started",
                                 "finished", "wall_elapsed", "status", "exit_code", "saved_results",
                                 "stderr_last_line", "stdout_file", "stderr_file"], logs)
    write_csv("attempts.csv", ["arm", "game", "attempt_id", "individual_id", "seed", "slurm_job_id", "slurm_array_task",
                               "status", "completed_frames", "auc", "final_return", "in_final_results",
                               "n_metric_rows", "tracking_file"], attempts)
    write_csv("training_metrics.csv.gz", ["arm", "game", "attempt_id", "individual_id", "seed", "frames", "loss",
                                          "epsilon", "sampled_mse", "sampled_abs_td", "per_is_weight_mean",
                                          "per_is_weight_min", "per_beta", "eval_mean_return"],
              metric_rows(), compress=True)


if __name__ == "__main__":
    main()
