"""Build a small cluster-transfer archive using only explicitly selected files."""

from pathlib import Path
import argparse
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", metavar="NAME",
                        help="Also write dist/NAME.zip with everything under a NAME/ folder")
    args = parser.parse_args()
    files = [
        "run_rct.py",
        "check_env.py",
        "verify_arms.py",
        "sync_wandb.py",
        "analyze_rct.py",
        "requirements.txt",
        "CLUSTER_RUN_GUIDE.md",
        "slurm/common.sh",
        "slurm/run_rct_bigbatch_test.slurm",
        "slurm/run_rct_bigbatch_variant.slurm",
        "slurm/run_followup.slurm",
        "followup.py",
        "FOLLOWUP_README.md",
        # verify_arms.py loads the first four during every training preflight.
        "tests/test_algorithm_audit.py",
        "tests/test_remaining_arms.py",
        "tests/test_run_integrity.py",
        "tests/test_cluster_tracking.py",
        "tests/test_followup.py",
        "tests/test_replay.py",
        "tests/test_torch_setup.py",
    ]
    files.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "rct").glob("*.py")))
    for name in files:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Required cluster file missing: {name}")

    output = ROOT / "dist" / "rct-cluster.tar.gz"
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for name in files:
            archive.add(ROOT / name, arcname=f"research_project/{name}", recursive=False)
    print(f"Created {output} ({output.stat().st_size:,} bytes; {len(files)} files)")
    print("Transfer this archive, then extract it with: tar -xzf rct-cluster.tar.gz")
    if args.zip:
        zip_path = ROOT / "dist" / f"{args.zip}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(ROOT / name, arcname=f"{args.zip}/{name}")
        print(f"Created {zip_path} ({zip_path.stat().st_size:,} bytes; {len(files)} files)")


if __name__ == "__main__":
    main()
