"""Decode the offline W&B runs under tracking/ and cross-check them against the
attempt JSONLs. Read-only on tracking/; writes tables/wandb_runs.csv and
tables/wandb_system.csv.gz. Needs the wandb package (the repo .venv has it):

    .venv/bin/python Results/corrected-rct-01/scripts/extract_wandb.py
"""
import csv
import gzip
import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

EXP = Path(__file__).resolve().parents[1]
TABLES = EXP / "tables"
SYSTEM_KEYS = ["cpu", "proc.cpu.threads", "proc.memory.rssMB", "proc.memory.percent", "memory_percent"]


def records(path):
    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    while (data := ds.scan_data()) is not None:
        record = pb.Record()
        record.ParseFromString(data)
        yield record.WhichOneof("record_type"), record


def items(entries):
    return {e.key or "/".join(e.nested_key): json.loads(e.value_json) for e in entries}


def jsonl_metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["event"] == "metrics":
                rows.append({k: v for k, v in e.items() if k != "event"})
    return rows


def main():
    run_rows, system_rows = [], []
    for run_file in sorted((EXP / "tracking").glob("*/*/wandb/offline-run-*/run-*.wandb")):
        arm, game = run_file.parts[-5], run_file.parts[-4]
        run_id = run_file.stem.removeprefix("run-")
        history, stats, exit_code = [], [], ""
        for kind, r in records(run_file):
            if kind == "history":
                history.append(items(r.history.item))
            elif kind == "stats":
                s = items(r.stats.item)
                stats.append(s)
                system_rows.append([arm, game, run_id, r.stats.timestamp.seconds] + [s.get(k, "") for k in SYSTEM_KEYS])
            elif kind == "exit":
                exit_code = r.exit.exit_code
        jsonl = EXP / "tracking" / arm / game / f"attempt-{run_id}.jsonl"
        ref = jsonl_metrics(jsonl) if jsonl.exists() else []
        logged = [{k: v for k, v in h.items() if not k.startswith("_")} for h in history]
        matches = len(ref) == len(logged) and all(
            a.keys() == b.keys() and all(abs(a[k] - b[k]) <= 1e-9 * max(1, abs(b[k])) for k in a)
            for a, b in zip(logged, ref))
        times = [h["_timestamp"] for h in history if "_timestamp" in h]
        rss = [s["proc.memory.rssMB"] for s in stats if "proc.memory.rssMB" in s]
        cpu = [s["cpu"] for s in stats if "cpu" in s]
        run_rows.append([arm, game, run_id, len(history), len(ref), matches,
                         max((h.get("frames", 0) for h in history), default=0),
                         round(times[-1] - times[0], 1) if times else "",
                         round(sum(cpu) / len(cpu), 2) if cpu else "", round(max(rss), 1) if rss else "",
                         exit_code, run_file.relative_to(EXP).as_posix()])
    with open(TABLES / "wandb_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "game", "attempt_id", "n_history", "n_jsonl_metrics", "history_matches_jsonl",
                    "last_frame", "wall_seconds", "mean_cpu_percent", "peak_rss_mb", "exit_code", "wandb_file"])
        w.writerows(run_rows)
    with gzip.open(TABLES / "wandb_system.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "game", "attempt_id", "unix_time"] + SYSTEM_KEYS)
        w.writerows(system_rows)
    ok = sum(r[5] for r in run_rows)
    print(f"{len(run_rows)} W&B runs decoded; {ok} match their attempt JSONL exactly; "
          f"{len(system_rows)} system-stat samples")


if __name__ == "__main__":
    main()
