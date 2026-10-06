import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("experiment_runner", ROOT / "experiments/run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("name", ["data/run", "reference/run", "experiments/run", "plots/run", "."])
def test_runner_protects_release(name):
    with pytest.raises(ValueError):
        runner.validate_output(ROOT / name)


@pytest.mark.parametrize("task", runner.COMMANDS)
def test_prepare_only_never_runs_optimizer(task, tmp_path):
    output = tmp_path / task
    subprocess.run([sys.executable, str(ROOT / "experiments/run.py"), task,
                    "--full", "--prepare-only", "--output-dir", str(output)], check=True)
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "prepared"
    assert not (output / "run.log").exists()
    assert not list((output / "work").rglob("*.csv"))
    if task == "routing":
        config = json.loads((output / "work/stagnation_genericity_manifest.json").read_text())
        assert config["status"] == "configured"
        assert config["execution"] == {}


def test_bell_data_generation_and_export(tmp_path):
    import reproduce

    output = tmp_path / "bell"
    before = reproduce.verify_checksums()
    subprocess.run([sys.executable, str(ROOT / "experiments/run.py"), "bell",
                    "--full", "--output-dir", str(output)], check=True)
    assert json.loads((output / "run.json").read_text())["status"] == "completed"
    reproduce.compare_table(ROOT / "data/bell/results.csv", output / "data/bell/results.csv")
    assert (output / "figures/fig_bell_floor.pdf").is_file()
    assert reproduce.verify_checksums() == before
