"""Rebuild and check the paper figures."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
NUMERICAL = (
    "fig_bell_floor.pdf",
    "fig_multiblock_stagnation_v2.pdf",
    "fig_xxz_recovery_pilot.pdf",
    "fig_stagnation_genericity_distribution.pdf",
    "fig_multiblock_rank_detail.pdf",
    "fig_natural_hamiltonian_intervention.pdf",
)
STATIC = ("fig1_same_point_repair.png", "fig_shallow_cut_budget.pdf")
TABLES = (
    "multiblock_instance_correlations.csv",
    "multiblock_paired_comparisons.csv",
    "multiblock_random_resampling.csv",
    "xxz_integration_statistics.csv",
)
OUTPUTS = tuple(f"figures/{name}" for name in (*NUMERICAL, *STATIC))
OUTPUTS += tuple(f"statistics/{name}" for name in TABLES)
OUTPUTS += (
    "statistics/routing_summary.json",
    "statistics/multiblock_hierarchical_manifest.json",
    "checks.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_hashes(root=ROOT):
    hashes = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or name in hashes
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"Invalid checksum entry: {name}")
        hashes[name] = digest
    if not hashes:
        raise ValueError("Empty checksum list")
    return hashes


def verify_checksums(root=ROOT):
    hashes = read_hashes(root)
    for name, digest in hashes.items():
        if sha(root / name) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return hashes


def validate_output(output, root=ROOT):
    root, output = root.resolve(), output.resolve()
    protected = {(root / name).resolve() for name in read_hashes(root)}
    protected.add((root / "SHA256SUMS").resolve())
    reserved = ("data", "reference", "plots", "tests", ".github")
    if (
        output == root
        or root.is_relative_to(output)
        or any(output.is_relative_to(root / name) for name in reserved)
        or output in protected
        or any((output / name).resolve() in protected for name in OUTPUTS)
    ):
        raise ValueError("Output must not overwrite release files")


def compare_table(expected, actual):
    with expected.open(encoding="utf-8", newline="") as handle:
        left = list(csv.reader(handle))
    with actual.open(encoding="utf-8", newline="") as handle:
        right = list(csv.reader(handle))
    if len(left) != len(right):
        raise ValueError(f"Row count mismatch: {expected.name}")
    maximum = 0.0
    for old_row, new_row in zip(left, right):
        if len(old_row) != len(new_row):
            raise ValueError(f"Column count mismatch: {expected.name}")
        for old, new in zip(old_row, new_row):
            if old == new:
                continue
            try:
                a, b = float(old), float(new)
            except ValueError as exc:
                raise ValueError(f"Text mismatch: {expected.name}") from exc
            if math.isnan(a) and math.isnan(b):
                continue
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Value mismatch: {expected.name}")
            maximum = max(maximum, abs(a - b))
    return maximum


def rebuild(output):
    from plots import STAGES
    import matplotlib
    import numpy
    import scipy

    output = output.resolve()
    validate_output(output)
    before = verify_checksums()
    with tempfile.TemporaryDirectory(prefix="vqc-figures-") as directory:
        work = Path(directory)
        env = dict(
            os.environ,
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(work / ".mpl"),
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        for index, stage in enumerate(STAGES, 1):
            print(f"[{index}/{len(STAGES)}] {stage}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "plots", stage, str(work)],
                cwd=ROOT,
                env=env,
                check=True,
            )
        differences = {
            name: compare_table(
                ROOT / "reference/tables" / name, work / "statistics" / name
            )
            for name in TABLES
        }
        for name in NUMERICAL:
            path = work / "figures" / name
            if not path.is_file() or path.stat().st_size < 1000:
                raise ValueError(f"Missing figure: {name}")
        for name in STATIC:
            shutil.copy2(ROOT / "reference/figures" / name, work / "figures" / name)
        if verify_checksums() != before:
            raise ValueError("Release changed during reproduction")
        record = {
            "mode": "frozen-data reproduction",
            "figures": 8,
            "compared_tables": 4,
            "release_checksums": len(before),
            "inputs_unchanged": True,
            "table_max_absolute_difference": differences,
            "tolerance": {"absolute": 1e-12, "relative": 1e-10},
            "software": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "output_sha256": {
                name: sha(work / name) for name in OUTPUTS if name != "checks.json"
            },
        }
        (work / "checks.json").write_text(json.dumps(record, indent=2) + "\n")
        validate_output(output)
        for name in OUTPUTS:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / name, target)
    verify_checksums()
    print("8 figures; 4 tables verified; inputs unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("figures", "check"), default="figures"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    if args.command == "check":
        print(f"{len(verify_checksums())} checksums passed.")
    else:
        rebuild(args.output_dir)


if __name__ == "__main__":
    main()
