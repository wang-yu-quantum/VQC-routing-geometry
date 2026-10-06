"""Run an experiment in a new, isolated directory."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
COMMANDS = {
    "bell": ("bell_floor_figure.py", []),
    "controlled": ("multiblock_stagnation_intervention.py", ["--seeds", "5", "--pool-size", "12"]),
    "tfim": ("natural_hamiltonian_intervention.py", ["--seeds-per-size", "5", "--pool-size", "16"]),
    "heisenberg": ("xxz_recovery_pilot.py", ["--sizes", "8", "10", "12", "--seeds-per-size", "5", "--candidates", "12"]),
    "routing": ("stagnation_genericity_pilot.py", []),
}


def validate_output(output):
    output = output.resolve()
    if output == REPOSITORY or REPOSITORY.is_relative_to(output):
        raise ValueError("Choose a new run directory.")
    for name in ("data", "reference", "plots", "tests", "experiments", ".github"):
        if output.is_relative_to(REPOSITORY / name):
            raise ValueError("Output cannot be inside release inputs or source directories.")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory.")
    return output


def command_for(task, smoke):
    script, arguments = COMMANDS[task]
    arguments = list(arguments)
    if smoke:
        if task == "routing":
            arguments.append("--estimate-only")
        elif task != "bell":
            arguments.append("--smoke")
    return [sys.executable, script, *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=COMMANDS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run the original small diagnostic configuration.")
    mode.add_argument("--full", action="store_true", help="Run the paper configuration; may be slow.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare source and command without running.")
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True)
    work = output / "work"
    shutil.copytree(ROOT / "src", work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if args.task == "routing":
        shutil.copyfile(ROOT / "configs/routing.json", work / "stagnation_genericity_manifest.json")
    command = command_for(args.task, args.smoke)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR=str(output / ".mpl"), OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    record = {
        "task": args.task,
        "mode": "smoke" if args.smoke else "full",
        "status": "prepared",
        "command": ["python", *command[1:]],
        "working_directory": "work",
        "python": platform.python_version(),
        "platform": platform.system(),
        "software": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
        "logical_cpus": os.cpu_count(),
        "free_disk_bytes": shutil.disk_usage(output).free,
        "threads": 1,
        "source_sha256": {
            str(path.relative_to(work)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(work.rglob("*")) if path.is_file()
        },
    }
    manifest = output / "run.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print("Command (inside work/):", " ".join(record["command"]), flush=True)
    if args.prepare_only:
        return
    record.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    try:
        with (output / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode,
                      status="completed" if result.returncode == 0 else "failed")
        if result.returncode == 0 and not args.smoke:
            from render import export_and_render
            export_and_render(args.task, output)
    except BaseException:
        record["status"] = "interrupted_or_failed"
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(f"Experiment failed; see {output / 'run.log'}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
