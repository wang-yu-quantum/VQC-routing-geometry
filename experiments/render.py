"""Render current paper panels from newly generated experiment outputs."""

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plots import read_rows, read_traces

FILES = {
    "bell": {"results.csv": "bell_floor_state_metric.csv"},
    "controlled": {
        "results.csv": "multiblock_stagnation_results.csv",
        "candidates.csv": "multiblock_stagnation_candidates.csv",
        "traces.npz": "VQC/routing_geometry/outputs/multiblock_stagnation_traces.npz",
    },
    "tfim": {
        "results.csv": "natural_hamiltonian_results.csv",
        "candidates.csv": "natural_hamiltonian_candidates.csv",
        "summary.json": "natural_hamiltonian_manifest.json",
        "traces.npz": "VQC/routing_geometry/outputs/natural_hamiltonian_traces.npz",
    },
    "heisenberg": {
        "checkpoints.csv": "xxz_checkpoints.csv",
        "methods.csv": "xxz_method_results.csv",
        "candidates.csv": "xxz_candidates.csv",
        "traces.npz": "xxz_traces.npz",
    },
    "routing": {"results.csv": "stagnation_genericity_results.csv"},
}


def export_and_render(task, output):
    data = output / "data" / task
    figures = output / "figures"
    statistics = output / "statistics"
    for directory in (data, figures, statistics):
        directory.mkdir(parents=True, exist_ok=True)
    for name, source in FILES[task].items():
        shutil.copyfile(output / "work" / source, data / name)
    if task == "bell":
        from plots import bell
        integers = {"m", "n", "k", "state_schmidt_rank"}
        rows = [{k: (int(v) if k in integers else float(v)) if v else ""
                 for k, v in row.items()} for row in read_rows(data / "results.csv")]
        bell.make_figure(rows, figures)
    elif task == "controlled":
        from plots import controlled
        controlled.run(data, output)
    elif task == "tfim":
        from plots import tfim
        tfim.make_figure(read_rows(data / "results.csv"), read_rows(data / "candidates.csv"),
                         json.loads((data / "summary.json").read_text())["instances"],
                         read_traces(data / "traces.npz"), figures)
    elif task == "heisenberg":
        from plots import heisenberg, heisenberg_statistics
        heisenberg_statistics.run(data, statistics)
        heisenberg.make_figure(read_rows(data / "checkpoints.csv"),
                               read_rows(data / "candidates.csv"), read_rows(data / "methods.csv"),
                               read_traces(data / "traces.npz"), figures)
    elif task == "routing":
        from plots import routing, check_events
        rows = routing.read_results(data / "results.csv")
        protocol = json.loads((output / "work/stagnation_genericity_manifest.json").read_text())
        check_events(rows, protocol["events"])
        routing.make_figure(rows, output_dir=figures)
        (statistics / "routing_summary.json").write_text(json.dumps(routing.statistics(rows), indent=2) + "\n")
