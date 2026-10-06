"""Paper figures from frozen results."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("bell", "controlled", "heisenberg_statistics", "selectors", "routing")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_traces(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def check_events(rows, events):
    counts = {}
    for level, threshold in events.items():
        count = 0
        for row in rows:
            event = (
                row["parameter_gradient_norm"] <= threshold["parameter_gradient_max"]
                and row["ambient_gradient_norm"] >= threshold["ambient_gradient_min"]
                and row["chi_ratio"] <= threshold["chi_ratio_max"]
                and row["energy_error"] >= threshold["energy_error_min"]
            )
            if int(event) != row[f"event_{level}"]:
                raise ValueError(f"Event mismatch: {row['route_id']}, {level}")
            count += int(event)
        counts[level] = count
    return counts


def draw(stage, output):
    figures = output / "figures"
    statistics = output / "statistics"
    figures.mkdir(parents=True, exist_ok=True)
    statistics.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data"

    if stage == "bell":
        from . import bell

        rows = [
            {
                key: (
                    int(value)
                    if key in {"m", "n", "k", "state_schmidt_rank"}
                    else float(value)
                )
                if value
                else ""
                for key, value in row.items()
            }
            for row in read_rows(data / "bell/results.csv")
        ]
        bell.make_figure(rows, figures)

    elif stage == "controlled":
        from . import controlled

        controlled.run(data / "controlled", output)

    elif stage == "heisenberg_statistics":
        from . import heisenberg_statistics

        heisenberg_statistics.run(data / "heisenberg", statistics)

    elif stage == "selectors":
        from . import heisenberg, tfim

        tfim.make_figure(
            read_rows(data / "tfim/results.csv"),
            read_rows(data / "tfim/candidates.csv"),
            json.loads((data / "tfim/summary.json").read_text())["instances"],
            read_traces(data / "tfim/traces.npz"),
            figures,
        )
        heisenberg.make_figure(
            read_rows(data / "heisenberg/checkpoints.csv"),
            read_rows(data / "heisenberg/candidates.csv"),
            read_rows(data / "heisenberg/methods.csv"),
            read_traces(data / "heisenberg/traces.npz"),
            figures,
        )

    elif stage == "routing":
        from . import routing

        rows = routing.read_results(data / "routing/results.csv")
        if len(rows) != 180:
            raise ValueError("Expected 180 routing instances")
        protocol = json.loads((data / "provenance.json").read_text())[
            "routing_protocol"
        ]
        counts = check_events(rows, protocol["events"])
        if counts != {"relaxed": 151, "primary": 85, "strict": 3}:
            raise ValueError("Unexpected event counts")
        routing.make_figure(rows, output_dir=figures)
        summary = routing.statistics(rows)
        summary["event_counts"] = counts
        summary["columns"] = {
            "by_size": ["n", "family", "events", "wilson95", "cluster95", "routes"],
            "by_family": [
                "family",
                "events",
                "cluster95",
                "median_coverage",
                "coverage_iqr",
            ],
            "sensitivity": ["definition", "events", "routes"],
            "rank_correlations": [
                "n",
                "budget_vs_coverage",
                "rank_vs_coverage",
                "error_vs_coverage",
            ],
        }
        (statistics / "routing_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")
