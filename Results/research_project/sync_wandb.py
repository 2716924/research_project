#!/usr/bin/env python3
"""Preview or upload finalized offline attempts, retaining all local files."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def pending_runs(root):
    for run_file in sorted(Path(root).rglob("run-*.wandb")):
        if not run_file.parent.name.startswith("offline-run-"):
            continue
        if Path(str(run_file) + ".synced").exists():
            continue
        attempt_id = run_file.stem.removeprefix("run-")
        journal = run_file.parent.parent.parent / f"attempt-{attempt_id}.jsonl"
        # Skip active or abruptly killed attempts; these need manual inspection.
        if not journal.is_file():
            continue
        try:
            with journal.open() as stream:
                last_line = None
                for last_line in stream:
                    pass
            final = json.loads(last_line or "{}")
        except (ValueError, OSError):
            continue
        if (final.get("event") == "finish" and final.get("wandb_finalized") is True
                and final.get("status") in {"completed", "interrupted", "failed"}):
            yield run_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Experiment directory (or its full/pilot subdirectory)")
    parser.add_argument("--entity", default="2716924-wits")
    parser.add_argument("--project", default="dqn-rct")
    parser.add_argument("--execute", action="store_true", help="Upload; otherwise preview only")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Experiment directory does not exist")
    runs = list(pending_runs(args.root))
    print(f"{len(runs)} finalized, unsynced offline attempts")
    for run_file in runs:
        command = [sys.executable, "-m", "wandb", "sync", "--entity", args.entity,
                   "--project", args.project, "--no-sync-tensorboard", str(run_file)]
        print(shlex.join(command), flush=True)
        if args.execute:
            environment = dict(os.environ, WANDB_MODE="online")
            subprocess.run(command, check=True, env=environment)
    if not args.execute:
        print("Preview only. Add --execute to upload after logging in with wandb login.")


if __name__ == "__main__":
    main()
