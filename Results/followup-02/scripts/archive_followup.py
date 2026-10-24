#!/usr/bin/env python3
"""
Move follow-up cluster output into Results/followup-02/ (standard library only).

    python3 Results/followup-02/scripts/archive_followup.py move SRC [--dry-run]
    python3 Results/followup-02/scripts/archive_followup.py verify

SRC is an unzipped follow-up folder: the bundle folder brought back from the
cluster, or an unzipped followup-output.zip. It may contain:

  SRC/experiments/followup-02/**   -> followup-02/**              (results, tracking)
  SRC/logs/rct-followup/**         -> followup-02/logs/**         (SLURM and row logs)
  SRC/followup_runs.csv            left in place (derived; `followup.py collect` rebuilds it)
  anything else (code, tests, ...) -> followup-02/provenance/**   (code as deployed)

Files are moved, never edited, merged or deleted. Each move is checksummed
before and after and appended to tables/file_moves.csv (old_path, new_path,
sha256, bytes). A file that is already archived with the same SHA-256 is left
in SRC and reported as a duplicate, so the same script can take a later
output zip that repeats earlier runs. If any destination exists with a
different SHA-256, nothing is moved. .DS_Store files are ignored.

`verify` re-hashes every recorded file and lists archived files that
file_moves.csv does not record.
"""

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parents[1]
PROJECT = DEST.parents[1]
MOVES = DEST / "tables" / "file_moves.csv"
FIELDS = ["old_path", "new_path", "sha256", "bytes"]
EXPERIMENT_ID = DEST.name
# Written by us, not moved in: never counted as unrecorded by verify.
OWN = {"README.md", "scripts", "tables"}
IGNORED = {".DS_Store"}
LEFT_IN_PLACE = {"followup_runs.csv"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def destination(src_root, path):
    parts = path.relative_to(src_root).parts
    if parts[:2] == ("experiments", EXPERIMENT_ID):
        return DEST.joinpath(*parts[2:])
    if parts[:1] == ("experiments",):
        return None  # another experiment's output: not ours
    if parts[:2] == ("logs", "rct-followup"):
        return DEST / "logs" / Path(*parts[2:])
    if len(parts) == 1 and parts[0] in LEFT_IN_PLACE:
        return None
    return DEST / "provenance" / Path(*parts)


def plan(src_root):
    moves, duplicates, conflicts, skipped = [], [], [], []
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames.sort()
        for name in sorted(filenames):
            src = Path(dirpath) / name
            if name in IGNORED:
                continue
            if src.is_symlink():
                sys.exit(f"Refusing to move symbolic link {src}")
            dst = destination(src_root, src)
            if dst is None:
                skipped.append(src)
                continue
            digest, size = sha256(src), src.stat().st_size
            if dst.exists():
                (duplicates if sha256(dst) == digest else conflicts).append((src, dst, digest, size))
            else:
                moves.append((src, dst, digest, size))
    return moves, duplicates, conflicts, skipped


def prune_empty_dirs(root):
    for dirpath, _, _ in sorted(os.walk(root), key=lambda w: -len(w[0])):
        try:
            os.rmdir(dirpath)  # fails, harmlessly, unless the folder is empty
        except OSError:
            pass


def cmd_move(args):
    src_root = Path(args.src).resolve()
    if not src_root.is_dir():
        sys.exit(f"{src_root} is not a folder")
    if DEST == src_root or DEST in src_root.parents:
        sys.exit("SRC must be outside the archive")
    moves, duplicates, conflicts, skipped = plan(src_root)
    print(f"{len(moves)} to move, {len(duplicates)} already archived (identical, left in SRC), "
          f"{len(conflicts)} conflicts, {len(skipped)} left in place")
    for src in skipped:
        print(f"  left in place: {rel(src)}")
    if conflicts:
        for src, dst, *_ in conflicts:
            print(f"  CONFLICT (archived copy differs): {rel(src)} -> {rel(dst)}")
        sys.exit("Nothing moved. Resolve the conflicts first.")
    if args.dry_run:
        for src, dst, *_ in moves[:20]:
            print(f"  {rel(src)} -> {rel(dst)}")
        if len(moves) > 20:
            print(f"  ... and {len(moves) - 20} more")
        return
    MOVES.parent.mkdir(parents=True, exist_ok=True)
    new_csv = not MOVES.exists()
    with open(MOVES, "a", newline="") as f:
        writer = csv.writer(f)
        if new_csv:
            writer.writerow(FIELDS)
        for src, dst, digest, size in moves:
            if dst.exists():
                sys.exit(f"{rel(dst)} appeared during the move; stopping (moves so far are recorded)")
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            if sha256(dst) != digest or dst.stat().st_size != size:
                sys.exit(f"Checksum changed while moving {rel(src)}; stopping")
            writer.writerow([rel(src), rel(dst), digest, size])
            f.flush()
    prune_empty_dirs(src_root)
    print(f"Moved {len(moves)} files; recorded in {rel(MOVES)}")


def cmd_verify(args):
    if not MOVES.exists():
        sys.exit(f"{rel(MOVES)} does not exist: nothing has been archived yet")
    with open(MOVES, newline="") as f:
        rows = list(csv.DictReader(f))
    bad = 0
    recorded = set()
    for row in rows:
        path = PROJECT / row["new_path"]
        recorded.add(path.resolve())
        if not path.exists():
            print(f"MISSING  {row['new_path']}")
            bad += 1
        elif path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            print(f"CHANGED  {row['new_path']}")
            bad += 1
    unrecorded = [p for p in DEST.rglob("*") if p.is_file() and p.name not in IGNORED
                  and p.relative_to(DEST).parts[0] not in OWN and p.resolve() not in recorded]
    for p in unrecorded:
        print(f"UNRECORDED  {rel(p)}")
    print(f"{len(rows)} recorded files: {len(rows) - bad} verified, {bad} missing or changed; "
          f"{len(unrecorded)} unrecorded")
    sys.exit(1 if bad or unrecorded else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("move")
    p.add_argument("src")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("verify")
    args = parser.parse_args()
    {"move": cmd_move, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()
