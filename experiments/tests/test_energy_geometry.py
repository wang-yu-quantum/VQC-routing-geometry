import json
from pathlib import Path

import numpy as np
import pytest

import multiblock_stagnation_intervention as circuit
import natural_hamiltonian_intervention as tfim
import stagnation_genericity_pilot as routing
import xxz_recovery_pilot as heisenberg

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("model", [tfim, heisenberg])
def test_energy_gradient_and_tangent_contraction(n, model):
    rng = np.random.default_rng(20261007 + n)
    ops, count = tfim.build_old_ansatz(n)
    theta = rng.normal(0.0, 0.2, count)
    hamiltonian = tfim.tfim_hamiltonian(n) if model is tfim else heisenberg.xxz_hamiltonian(n)
    energy, gradient, state = model.energy_and_gradient(ops, theta, hamiltonian, n)
    evaluate = tfim.energy_geometry if model is tfim else heisenberg.state_geometry
    geometry = evaluate(ops, theta, hamiltonian, n)
    np.testing.assert_allclose(gradient, geometry.parameter_gradient, atol=2e-10, rtol=2e-9)
    step = 1e-6
    for index in np.linspace(0, count - 1, 7, dtype=int):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += step
        minus[index] -= step
        difference = (model.energy_and_gradient(ops, plus, hamiltonian, n)[0]
                      - model.energy_and_gradient(ops, minus, hamiltonian, n)[0]) / (2 * step)
        np.testing.assert_allclose(gradient[index], difference, atol=2e-8, rtol=2e-6)
    h_state = hamiltonian @ state
    variance = float(np.vdot(h_state, h_state).real - energy**2)
    np.testing.assert_allclose(geometry.ambient_norm**2, 4 * variance, atol=2e-11)
    assert geometry.projected_norm <= geometry.ambient_norm + 1e-10


@pytest.mark.parametrize("n", [4, 6])
def test_same_point_and_residual_score(n):
    rng = np.random.default_rng(20261017 + n)
    ops = []
    count = circuit.add_local_layer(ops, n, 0)
    theta = rng.normal(0, 0.3, count)
    hamiltonian = heisenberg.xxz_hamiltonian(n)
    old = heisenberg.state_geometry(ops, theta, hamiltonian, n)
    insertion = len(ops)
    sequence = tuple((q, q + 1) for q in range(n - 1))
    augmented = heisenberg.build_augmented_ops(ops, insertion, sequence, n, count)
    full_theta = np.r_[theta, np.zeros(3 * n)]
    state, tangent = heisenberg.candidate_tangents_at_zero(ops, insertion, sequence, theta, n)
    full_state, full_tangent = circuit.simulate_with_tangents(augmented, full_theta, n, full_theta.size)
    np.testing.assert_allclose(state, old.state, atol=1e-12)
    np.testing.assert_allclose(full_state, old.state, atol=1e-12)
    np.testing.assert_allclose(tangent, full_tangent[:, count:], atol=1e-12)
    _, old_gradient, _ = heisenberg.energy_and_gradient(ops, theta, hamiltonian, n)
    _, full_gradient, _ = heisenberg.energy_and_gradient(augmented, full_theta, hamiltonian, n)
    np.testing.assert_allclose(old_gradient, full_gradient[:count], atol=1e-11)
    candidate = circuit.realify(circuit.horizontalize(state, tangent))
    metrics = heisenberg.tangent_metrics(old.tangent_basis, candidate)
    residual = old.ambient_real - old.tangent_basis @ (old.tangent_basis.T @ old.ambient_real)
    score = np.linalg.norm(metrics["residual_basis"].T @ residual)**2
    union = np.linalg.norm(metrics["union_basis"].T @ old.ambient_real)**2 - old.projected_squared
    np.testing.assert_allclose(score, union, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_routing_construction_matches_frozen_architectures(n):
    from plots.routing import read_results
    frozen = {row["route_id"]: row for row in read_results(ROOT / "data/routing/results.csv")
              if int(row["n"]) == n}
    routes = routing.build_routes(n)
    assert len(routes) == 20
    for route in routes:
        row = frozen[route.route_id]
        assert [list(map(list, block)) for block in route.blocks] == json.loads(row["routing_sequences"])
        assert list(route.cumulative_cut_profile) == json.loads(row["cumulative_cut_profile"])
        assert route.total_cut_budget == row["total_cut_budget"]
        assert route.paired_route_id == row["paired_route_id"]


def test_clean_reports_keep_numeric_outputs(tmp_path, monkeypatch):
    from plots import read_rows
    from plots.routing import read_results
    summaries = json.loads((ROOT / "data/tfim/summary.json").read_text())["instances"]
    monkeypatch.setattr(tfim, "REPORT_FILE", tmp_path / "tfim.json")
    tfim.write_report(read_rows(ROOT / "data/tfim/results.csv"),
                      read_rows(ROOT / "data/tfim/candidates.csv"), summaries)
    report = json.loads((tmp_path / "tfim.json").read_text())
    assert report["instances"] == 15
    assert len(report["score_correlations"]) == 15 * 3 * 4
    monkeypatch.setattr(heisenberg, "REPORT_FILE", tmp_path / "heisenberg.json")
    heisenberg.write_report(read_rows(ROOT / "data/heisenberg/checkpoints.csv"),
                            read_rows(ROOT / "data/heisenberg/candidates.csv"),
                            read_rows(ROOT / "data/heisenberg/methods.csv"), False)
    assert json.loads((tmp_path / "heisenberg.json").read_text())["instances"] == 15
    config = json.loads((ROOT / "experiments/configs/routing.json").read_text())
    routing.write_report(read_results(ROOT / "data/routing/results.csv"), config, output_dir=tmp_path)
    report = json.loads((tmp_path / routing.REPORT_PATH.name).read_text())
    assert report["primary_instances"] == 85
    assert report["instances"] == 180
