from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata, spearmanr

from multiblock_stagnation_intervention import (
    Gate,
    add_cnot_sequence,
    add_local_layer,
    apply_gate,
    apply_one_qubit,
    build_gadget_ops,
    horizontalize,
    identity_sequence,
    one_qubit_matrix,
    PAULI,
    realify,
    simulate,
    simulate_with_tangents,
    zero_state,
)
from VQC.routing_geometry.core import kappa, routing_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
SNAPSHOT_DIR = (
    ROOT / "VQC" / "routing_geometry" / "outputs" / "natural_hamiltonian_snapshots"
)
RESULTS_CSV = ROOT / "natural_hamiltonian_results.csv"
CANDIDATES_CSV = ROOT / "natural_hamiltonian_candidates.csv"
REPORT_FILE = ROOT / 'natural_hamiltonian_intervention_statistics.json'
MANIFEST_FILE = ROOT / "natural_hamiltonian_manifest.json"
TRACE_FILE = (
    ROOT / "VQC" / "routing_geometry" / "outputs" / "natural_hamiltonian_traces.npz"
)
FIGURE_PDF = FIGURE_DIR / "fig_natural_hamiltonian_intervention.pdf"
FIGURE_PNG = FIGURE_DIR / "fig_natural_hamiltonian_intervention.png"

BASE_SEED = 20260723
J_COUPLING = 1.0
TRANSVERSE_FIELD = 1.0
OLD_BLOCKS = 2
CNOTS_PER_CANDIDATE = 4
INTERVENTION_STEPS = 50
CHECKPOINTS = (1, 5, 20, 50)
METHODS = ("identity", "random", "osr_only", "pauli_only", "projection", "oracle")


@dataclass
class EnergyGeometry:
    energy: float
    ambient_norm: float
    projected_norm: float
    projected_squared: float
    chi: float
    state: np.ndarray
    gram: np.ndarray
    tangent_real: np.ndarray
    ambient_real: np.ndarray
    parameter_gradient: np.ndarray


@dataclass
class AdamRecord:
    theta: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    step: int
    energies: np.ndarray


@dataclass
class NaturalCandidate:
    index: int
    name: str
    sequence: tuple[tuple[int, int], ...]
    matrix: np.ndarray
    osr_score: float = 0.0
    pauli_score: float = 0.0
    full_score: float = 0.0
    delta: float = 0.0
    old_rank: int = 0
    candidate_rank: int = 0
    union_rank: int = 0
    novel_dimension: int = 0
    intersection_dimension: int = 0
    overlap_fraction: float = 0.0
    principal_angle_min_deg: float = 90.0
    principal_angle_median_deg: float = 90.0
    principal_angle_max_deg: float = 90.0
    principal_angle_obliqueness_deg: float = 0.0
    projector_commutator_fro: float = 0.0
    gram_condition_number: float = float("inf")
    same_state_error: float = 0.0
    same_energy_error: float = 0.0
    same_old_gradient_error: float = 0.0
    same_ambient_error: float = 0.0
    energies: np.ndarray | None = None


def tfim_hamiltonian(
    n: int,
    coupling: float = J_COUPLING,
    field: float = TRANSVERSE_FIELD,
) -> sp.csr_matrix:
    dimension = 2**n
    indices = np.arange(dimension, dtype=np.int64)
    diagonal = np.zeros(dimension, dtype=float)
    for qubit in range(n - 1):
        left = (indices >> (n - 1 - qubit)) & 1
        right = (indices >> (n - 2 - qubit)) & 1
        diagonal += -coupling * np.where(left == right, 1.0, -1.0)
    hamiltonian = sp.diags(diagonal, format="csr", dtype=np.complex128)
    for qubit in range(n):
        mask = 1 << (n - 1 - qubit)
        flipped = indices ^ mask
        hamiltonian += sp.csr_matrix(
            (
                np.full(dimension, -field, dtype=np.complex128),
                (indices, flipped),
            ),
            shape=(dimension, dimension),
        )
    return hamiltonian


def exact_ground_state(hamiltonian: sp.csr_matrix) -> tuple[float, np.ndarray]:
    values, vectors = eigsh(hamiltonian, k=1, which="SA", tol=1e-12)
    state = vectors[:, 0]
    phase_index = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[phase_index]))
    return float(values[0].real), state


def energy_and_gradient(
    ops: list[Gate],
    theta: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    forward = [zero_state(n)]
    for gate in ops:
        forward.append(apply_gate(forward[-1], gate, theta, n))
    state = forward[-1]
    h_state = np.asarray(hamiltonian @ state).reshape(-1)
    energy = float(np.real(np.vdot(state, h_state)))
    gradient = np.zeros(theta.size, dtype=float)
    co_state = h_state.copy()
    for index in range(len(ops) - 1, -1, -1):
        gate = ops[index]
        if gate.param >= 0:
            derivative_matrix = (
                -0.5j * PAULI[gate.kind[1:]]
            ) @ one_qubit_matrix(gate, theta)
            derivative_after_gate = apply_one_qubit(
                forward[index], derivative_matrix, gate.q0, n
            )
            gradient[gate.param] += 2.0 * float(
                np.real(np.vdot(co_state, derivative_after_gate))
            )
        co_state = apply_gate(co_state, gate, theta, n, dagger=True)
    return energy, gradient, state


def orthonormal_basis(
    vectors: np.ndarray,
    relative_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    if vectors.size == 0:
        return np.empty((vectors.shape[0], 0)), np.empty(0)
    left, singular_values, _ = np.linalg.svd(vectors, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return np.empty((vectors.shape[0], 0)), singular_values
    rank = int(np.sum(singular_values > relative_tolerance * singular_values[0]))
    return left[:, :rank], singular_values[:rank]


def projected_norm(tangent_real: np.ndarray, ambient_real: np.ndarray) -> float:
    basis, _ = orthonormal_basis(tangent_real)
    if basis.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(basis.T @ ambient_real))


def energy_geometry(
    ops: list[Gate],
    theta: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
) -> EnergyGeometry:
    state, tangents = simulate_with_tangents(ops, theta, n, theta.size)
    h_state = np.asarray(hamiltonian @ state).reshape(-1)
    energy = float(np.real(np.vdot(state, h_state)))
    ambient = 2.0 * (h_state - energy * state)
    tangent_real = realify(horizontalize(state, tangents))
    ambient_real = realify(ambient[:, None]).reshape(-1)
    projection = projected_norm(tangent_real, ambient_real)
    ambient_norm = float(np.linalg.norm(ambient_real))
    parameter_gradient = tangent_real.T @ ambient_real
    return EnergyGeometry(
        energy=energy,
        ambient_norm=ambient_norm,
        projected_norm=projection,
        projected_squared=projection**2,
        chi=(projection**2 / ambient_norm**2) if ambient_norm > 1e-15 else 0.0,
        state=state,
        gram=tangent_real.T @ tangent_real,
        tangent_real=tangent_real,
        ambient_real=ambient_real,
        parameter_gradient=parameter_gradient,
    )


def dimer_sequence(n: int, layer: int) -> tuple[tuple[int, int], ...]:
    if layer % 2 == 0:
        return tuple((qubit, qubit + 1) for qubit in range(0, n - 1, 2))
    return tuple((qubit + 1, qubit) for qubit in range(0, n - 1, 2))


def build_old_ansatz(n: int, blocks: int = OLD_BLOCKS) -> tuple[list[Gate], int]:
    ops: list[Gate] = []
    parameter = add_local_layer(ops, n, 0)
    for layer in range(blocks):
        add_cnot_sequence(ops, dimer_sequence(n, layer))
        parameter = add_local_layer(ops, n, parameter)
    return ops, parameter


def initial_product_theta(
    n: int,
    blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    theta = rng.normal(0.0, 0.025, size=3 * n * (blocks + 1))
    for qubit in range(n):
        theta[3 * qubit + 1] += np.pi / 2.0
    return theta


def adam_optimize(
    ops: list[Gate],
    initial_theta: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
    steps: int,
    learning_rate: float,
    initial_first: np.ndarray | None = None,
    initial_second: np.ndarray | None = None,
    initial_step: int = 0,
    cosine_decay: bool = False,
) -> AdamRecord:
    theta = initial_theta.copy()
    first = (
        np.zeros_like(theta) if initial_first is None else initial_first.copy()
    )
    second = (
        np.zeros_like(theta) if initial_second is None else initial_second.copy()
    )
    beta1, beta2 = 0.9, 0.999
    energies = []
    for local_step in range(1, steps + 1):
        energy, gradient, _ = energy_and_gradient(ops, theta, hamiltonian, n)
        energies.append(energy)
        global_step = initial_step + local_step
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        first_hat = first / (1.0 - beta1**global_step)
        second_hat = second / (1.0 - beta2**global_step)
        rate = learning_rate
        if cosine_decay:
            rate *= 0.002 + 0.998 * 0.5 * (
                1.0 + np.cos(np.pi * (local_step - 1) / steps)
            )
        theta -= rate * first_hat / (np.sqrt(second_hat) + 1e-8)
    final_energy, _, _ = energy_and_gradient(ops, theta, hamiltonian, n)
    energies.append(final_energy)
    return AdamRecord(
        theta=theta,
        first_moment=first,
        second_moment=second,
        step=initial_step + steps,
        energies=np.asarray(energies),
    )


def optimize_old_point(
    ops: list[Gate],
    initial_theta: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
    smoke: bool,
) -> tuple[np.ndarray, AdamRecord, dict[str, object]]:
    warmup = adam_optimize(
        ops,
        initial_theta,
        hamiltonian,
        n,
        steps=80 if smoke else 600,
        learning_rate=0.035,
        cosine_decay=True,
    )

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        energy, gradient, _ = energy_and_gradient(ops, theta, hamiltonian, n)
        return energy, gradient

    result = minimize(
        objective,
        warmup.theta,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 80 if smoke else 700,
            "gtol": 1e-9 if smoke else 1e-11,
            "ftol": 1e-12 if smoke else 1e-15,
            "maxls": 40,
        },
    )
    metadata: dict[str, object] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "final_gradient_norm": float(np.linalg.norm(result.jac)),
        "warmup_final_energy": float(warmup.energies[-1]),
    }
    return np.asarray(result.x, dtype=float), warmup, metadata


def finite_difference_gradient_check() -> dict[str, float]:
    n = 4
    rng = np.random.default_rng(99117)
    hamiltonian = tfim_hamiltonian(n)
    ops, parameter_count = build_old_ansatz(n)
    theta = rng.normal(0.0, 0.2, size=parameter_count)
    _, analytic, _ = energy_and_gradient(ops, theta, hamiltonian, n)
    epsilon = 1e-6
    numerical = np.zeros_like(theta)
    for index in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        e_plus = energy_and_gradient(ops, plus, hamiltonian, n)[0]
        e_minus = energy_and_gradient(ops, minus, hamiltonian, n)[0]
        numerical[index] = (e_plus - e_minus) / (2.0 * epsilon)
    absolute = float(np.max(np.abs(analytic - numerical)))
    relative = absolute / max(1e-15, float(np.max(np.abs(numerical))))
    if absolute > 2e-7 or relative > 2e-6:
        raise AssertionError(
            f"Energy gradient check failed: absolute={absolute}, relative={relative}."
        )
    return {"maximum_absolute_error": absolute, "maximum_relative_error": relative}


def candidate_pool(
    n: int,
    pool_size: int,
    rng: np.random.Generator,
) -> list[NaturalCandidate]:
    identity = identity_sequence(n)
    candidates = [
        NaturalCandidate(
            index=0,
            name="identity",
            sequence=identity,
            matrix=routing_matrix(n, identity),
        )
    ]
    seen = {np.eye(n, dtype=np.uint8).tobytes()}
    attempts = 0
    while len(candidates) < pool_size + 1 and attempts < 200000:
        attempts += 1
        sequence = tuple(
            tuple(int(value) for value in rng.choice(n, size=2, replace=False))
            for _ in range(CNOTS_PER_CANDIDATE)
        )
        matrix = routing_matrix(n, sequence)
        key = matrix.tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            NaturalCandidate(
                index=len(candidates),
                name=f"B{len(candidates):02d}",
                sequence=sequence,
                matrix=matrix,
            )
        )
    if len(candidates) != pool_size + 1:
        raise RuntimeError("Could not generate the requested fixed-cost candidate pool.")
    return candidates


def contiguous_osr_score(matrix: np.ndarray, n: int) -> float:
    return float(sum(kappa(matrix, tuple(range(size))) for size in range(1, n)))


def tangent_overlap_metrics(
    old_tangent: np.ndarray,
    new_tangent: np.ndarray,
) -> dict[str, float | int]:
    old_basis, old_singular = orthonormal_basis(old_tangent)
    new_basis, new_singular = orthonormal_basis(new_tangent)
    union_basis, union_singular = orthonormal_basis(
        np.concatenate((old_tangent, new_tangent), axis=1)
    )
    old_rank = old_basis.shape[1]
    new_rank = new_basis.shape[1]
    union_rank = union_basis.shape[1]
    if old_rank and new_rank:
        singular = np.clip(
            np.linalg.svd(old_basis.T @ new_basis, compute_uv=False),
            0.0,
            1.0,
        )
    else:
        singular = np.empty(0)
    padded = np.zeros(new_rank, dtype=float)
    padded[: singular.size] = np.clip(singular, 0.0, 1.0)
    angles = np.degrees(np.arccos(padded)) if new_rank else np.array([90.0])
    intersection = int(np.sum(singular > 1.0 - 1e-8))
    overlap_fraction = (
        float(np.sum(singular**2) / new_rank) if new_rank else 0.0
    )
    obliqueness = (
        float(np.max(np.minimum(angles, 90.0 - angles))) if new_rank else 0.0
    )
    commutator_fro = float(
        np.sqrt(2.0 * np.sum(singular**2 * (1.0 - singular**2)))
    )
    if union_singular.size:
        gram_condition = float(
            (union_singular[0] / union_singular[-1]) ** 2
        )
    else:
        gram_condition = float("inf")
    return {
        "old_rank": old_rank,
        "candidate_rank": new_rank,
        "union_rank": union_rank,
        "novel_dimension": union_rank - old_rank,
        "intersection_dimension": intersection,
        "overlap_fraction": overlap_fraction,
        "principal_angle_min_deg": float(np.min(angles)),
        "principal_angle_median_deg": float(np.median(angles)),
        "principal_angle_max_deg": float(np.max(angles)),
        "principal_angle_obliqueness_deg": obliqueness,
        "projector_commutator_fro": commutator_fro,
        "gram_condition_number": gram_condition,
    }


def score_candidates(
    candidates: list[NaturalCandidate],
    old_ops: list[Gate],
    old_theta: np.ndarray,
    old_geometry: EnergyGeometry,
    hamiltonian: sp.csr_matrix,
    n: int,
) -> dict[int, tuple[list[Gate], np.ndarray, EnergyGeometry]]:
    cache: dict[int, tuple[list[Gate], np.ndarray, EnergyGeometry]] = {}
    old_energy, old_gradient, old_state = energy_and_gradient(
        old_ops, old_theta, hamiltonian, n
    )
    for candidate in candidates:
        candidate.osr_score = contiguous_osr_score(candidate.matrix, n)
        ops = build_gadget_ops(candidate.sequence, old_ops, n)
        theta = np.concatenate((np.zeros(3 * n), old_theta))
        candidate_geometry = energy_geometry(ops, theta, hamiltonian, n)
        full_energy, full_gradient, state = energy_and_gradient(
            ops, theta, hamiltonian, n
        )
        candidate.same_state_error = float(np.linalg.norm(state - old_state))
        candidate.same_energy_error = abs(full_energy - old_energy)
        candidate.same_old_gradient_error = float(
            np.linalg.norm(full_gradient[3 * n :] - old_gradient)
        )
        candidate.same_ambient_error = float(
            np.linalg.norm(candidate_geometry.ambient_real - old_geometry.ambient_real)
        )
        new_tangent = candidate_geometry.tangent_real[:, : 3 * n]
        candidate.pauli_score = projected_norm(
            new_tangent, old_geometry.ambient_real
        ) ** 2
        candidate.full_score = candidate_geometry.projected_squared
        candidate.delta = max(
            0.0, candidate.full_score - old_geometry.projected_squared
        )
        metrics = tangent_overlap_metrics(old_geometry.tangent_real, new_tangent)
        for key, value in metrics.items():
            setattr(candidate, key, value)
        if (
            candidate.same_state_error > 2e-11
            or candidate.same_energy_error > 2e-11
            or candidate.same_old_gradient_error > 2e-9
            or candidate.same_ambient_error > 2e-10
        ):
            raise AssertionError(
                f"Same-point invariance failed for {candidate.name}: "
                f"state={candidate.same_state_error}, "
                f"energy={candidate.same_energy_error}, "
                f"gradient={candidate.same_old_gradient_error}, "
                f"ambient={candidate.same_ambient_error}."
            )
        cache[candidate.index] = (ops, theta, candidate_geometry)
    return cache


def train_candidates(
    candidates: list[NaturalCandidate],
    cache: dict[int, tuple[list[Gate], np.ndarray, EnergyGeometry]],
    hamiltonian: sp.csr_matrix,
    n: int,
    smoke: bool,
) -> None:
    steps = 8 if smoke else INTERVENTION_STEPS
    for candidate in candidates:
        ops, theta, _ = cache[candidate.index]
        trained = adam_optimize(
            ops,
            theta,
            hamiltonian,
            n,
            steps=steps,
            learning_rate=0.025,
            cosine_decay=False,
        )
        if smoke and steps < INTERVENTION_STEPS:
            padded = np.pad(
                trained.energies,
                (0, INTERVENTION_STEPS + 1 - trained.energies.size),
                mode="edge",
            )
            candidate.energies = padded
        else:
            candidate.energies = trained.energies


def select_methods(
    candidates: list[NaturalCandidate],
) -> dict[str, int]:
    nonidentity = candidates[1:]
    improvements = {
        candidate.index: float(candidate.energies[0] - candidate.energies[-1])
        for candidate in candidates
        if candidate.energies is not None
    }
    def stable_argmax(attribute: str) -> int:
        values = np.asarray(
            [float(getattr(candidate, attribute)) for candidate in nonidentity]
        )
        maximum = float(np.max(values))
        tolerance = 1e-8 * max(abs(maximum), 1e-12)
        tied = [
            candidate.index
            for candidate, value in zip(nonidentity, values)
            if value >= maximum - tolerance
        ]
        return min(tied)

    return {
        "identity": 0,
        "osr_only": stable_argmax("osr_score"),
        "pauli_only": stable_argmax("pauli_score"),
        "projection": stable_argmax("delta"),
        "oracle": max(
            candidates, key=lambda item: (improvements[item.index], -item.index)
        ).index,
    }


def diagnostic_pair(
    candidates: list[NaturalCandidate],
) -> tuple[int, int, float]:
    nonidentity = candidates[1:]
    pairs = []
    epsilon = 1e-16
    for left_index, left in enumerate(nonidentity):
        for right in nonidentity[left_index + 1 :]:
            pauli_gap = abs(
                np.log10(left.pauli_score + epsilon)
                - np.log10(right.pauli_score + epsilon)
            )
            overlap_gap = abs(left.overlap_fraction - right.overlap_fraction)
            pairs.append((pauli_gap, -overlap_gap, left, right))
    pairs.sort(key=lambda item: (item[0], item[1]))
    minimum_gap = pairs[0][0]
    shortlist = [item for item in pairs if item[0] <= minimum_gap + 1e-8]
    chosen = min(shortlist, key=lambda item: item[1])
    return chosen[2].index, chosen[3].index, float(chosen[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_instance(
    n: int,
    seed: int,
    pool_size: int,
    random_draws: int,
    smoke: bool,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, np.ndarray],
]:
    rng = np.random.default_rng(seed)
    hamiltonian = tfim_hamiltonian(n)
    ground_energy, ground_state = exact_ground_state(hamiltonian)
    old_ops, parameter_count = build_old_ansatz(n)
    initial_theta = initial_product_theta(n, OLD_BLOCKS, rng)
    old_theta, warmup, optimizer_metadata = optimize_old_point(
        old_ops, initial_theta, hamiltonian, n, smoke
    )
    old_geometry = energy_geometry(old_ops, old_theta, hamiltonian, n)
    old_energy, old_gradient, old_state = energy_and_gradient(
        old_ops, old_theta, hamiltonian, n
    )
    old_energy_error = old_energy - ground_energy
    state_distance = float(np.linalg.norm(old_state - zero_state(n)))
    chi_limit = 2e-2 if smoke else 1e-4
    if old_geometry.ambient_norm <= 1e-4:
        raise AssertionError(
            f"Ambient gradient vanished for n={n}, seed={seed}: "
            f"{old_geometry.ambient_norm}."
        )
    if old_geometry.chi >= chi_limit:
        raise AssertionError(
            f"Old point is not sufficiently stationary for n={n}, seed={seed}: "
            f"chi={old_geometry.chi}, limit={chi_limit}."
        )
    if old_energy_error <= 1e-6:
        raise AssertionError(
            f"Old ansatz reached the exact ground energy for n={n}, seed={seed}."
        )
    if state_distance <= 1e-2:
        raise AssertionError(f"Old physical state is too close to |0> for n={n}.")

    candidates = candidate_pool(n, pool_size, rng)
    cache = score_candidates(
        candidates, old_ops, old_theta, old_geometry, hamiltonian, n
    )
    train_candidates(candidates, cache, hamiltonian, n, smoke)
    selected = select_methods(candidates)
    by_index = {candidate.index: candidate for candidate in candidates}
    oracle = by_index[selected["oracle"]]
    assert oracle.energies is not None
    oracle_gain = float(oracle.energies[0] - oracle.energies[INTERVENTION_STEPS])
    nonidentity = candidates[1:]
    nonidentity_gains = np.array(
        [
            float(candidate.energies[0] - candidate.energies[INTERVENTION_STEPS])
            for candidate in nonidentity
            if candidate.energies is not None
        ]
    )
    random_indices = rng.integers(0, len(nonidentity), size=random_draws)
    random_candidates = [nonidentity[index] for index in random_indices]
    random_gains = np.array(
        [
            float(candidate.energies[0] - candidate.energies[INTERVENTION_STEPS])
            for candidate in random_candidates
            if candidate.energies is not None
        ]
    )
    near_oracle_probability = float(
        np.mean(random_gains >= 0.9 * oracle_gain - 1e-12)
    )
    pair_left, pair_right, pair_pauli_gap = diagnostic_pair(candidates)
    instance_id = f"tfim_n{n}_t{OLD_BLOCKS}_s{seed}"

    result_rows: list[dict[str, object]] = []
    for method in ("identity", "osr_only", "pauli_only", "projection", "oracle"):
        candidate = by_index[selected[method]]
        assert candidate.energies is not None
        gain = float(
            candidate.energies[0] - candidate.energies[INTERVENTION_STEPS]
        )
        row: dict[str, object] = {
            "row_type": "method_summary",
            "instance": instance_id,
            "seed": seed,
            "n": n,
            "t": OLD_BLOCKS,
            "task": "open_TFIM_J1_h1",
            "method": method,
            "candidate_index": candidate.index,
            "candidate_name": candidate.name,
            "parameters_added": 3 * n,
            "routing_slots": CNOTS_PER_CANDIDATE,
            "gadget_cnot_count": 2 * CNOTS_PER_CANDIDATE,
            "candidate_pool_size": pool_size,
            "ground_energy": ground_energy,
            "old_energy": old_energy,
            "old_energy_error": old_energy_error,
            "old_ambient_norm": old_geometry.ambient_norm,
            "old_projected_norm": old_geometry.projected_norm,
            "old_chi": old_geometry.chi,
            "osr_score": candidate.osr_score,
            "pauli_score": candidate.pauli_score,
            "full_projected_score": candidate.full_score,
            "delta_B": candidate.delta,
            "overlap_fraction": candidate.overlap_fraction,
            "intersection_dimension": candidate.intersection_dimension,
            "novel_dimension": candidate.novel_dimension,
            "gram_condition_number": candidate.gram_condition_number,
            "final_energy": float(candidate.energies[-1]),
            "final_energy_error": float(candidate.energies[-1] - ground_energy),
            "gain_50": gain,
            "gain_per_added_cnot_50": gain / (2 * CNOTS_PER_CANDIDATE),
            "random_percentile_50": float(
                100.0 * np.mean(nonidentity_gains <= gain + 1e-15)
            ),
            "random_near_oracle_probability": near_oracle_probability,
        }
        for checkpoint in CHECKPOINTS:
            row[f"energy_after_{checkpoint}"] = float(
                candidate.energies[checkpoint]
            )
            row[f"gain_after_{checkpoint}"] = float(
                candidate.energies[0] - candidate.energies[checkpoint]
            )
        result_rows.append(row)

    random_row: dict[str, object] = {
        "row_type": "method_summary",
        "instance": instance_id,
        "seed": seed,
        "n": n,
        "t": OLD_BLOCKS,
        "task": "open_TFIM_J1_h1",
        "method": "random",
        "candidate_index": -1,
        "candidate_name": "repeated_random_mean",
        "parameters_added": 3 * n,
        "routing_slots": CNOTS_PER_CANDIDATE,
        "gadget_cnot_count": 2 * CNOTS_PER_CANDIDATE,
        "candidate_pool_size": pool_size,
        "ground_energy": ground_energy,
        "old_energy": old_energy,
        "old_energy_error": old_energy_error,
        "old_ambient_norm": old_geometry.ambient_norm,
        "old_projected_norm": old_geometry.projected_norm,
        "old_chi": old_geometry.chi,
        "final_energy": float(
            np.mean([candidate.energies[-1] for candidate in random_candidates])
        ),
        "final_energy_error": float(
            np.mean([candidate.energies[-1] for candidate in random_candidates])
            - ground_energy
        ),
        "gain_50": float(np.mean(random_gains)),
        "gain_per_added_cnot_50": float(
            np.mean(random_gains) / (2 * CNOTS_PER_CANDIDATE)
        ),
        "random_percentile_50": 50.0,
        "random_near_oracle_probability": near_oracle_probability,
        "random_gain_q025": float(np.quantile(random_gains, 0.025)),
        "random_gain_q975": float(np.quantile(random_gains, 0.975)),
    }
    for checkpoint in CHECKPOINTS:
        energies = np.array(
            [candidate.energies[checkpoint] for candidate in random_candidates]
        )
        random_row[f"energy_after_{checkpoint}"] = float(np.mean(energies))
        random_row[f"gain_after_{checkpoint}"] = float(old_energy - np.mean(energies))
    result_rows.append(random_row)

    for draw_index, candidate in enumerate(random_candidates):
        assert candidate.energies is not None
        result_rows.append(
            {
                "row_type": "random_draw",
                "instance": instance_id,
                "seed": seed,
                "n": n,
                "t": OLD_BLOCKS,
                "task": "open_TFIM_J1_h1",
                "method": "random",
                "draw": draw_index,
                "candidate_index": candidate.index,
                "candidate_name": candidate.name,
                "gain_50": float(
                    candidate.energies[0]
                    - candidate.energies[INTERVENTION_STEPS]
                ),
                "final_energy": float(candidate.energies[-1]),
                "final_energy_error": float(candidate.energies[-1] - ground_energy),
                "near_oracle": int(
                    candidate.energies[0] - candidate.energies[-1]
                    >= 0.9 * oracle_gain - 1e-12
                ),
            }
        )

    selected_reverse: dict[int, list[str]] = defaultdict(list)
    for method, index in selected.items():
        selected_reverse[index].append(method)
    candidate_rows: list[dict[str, object]] = []
    for candidate in candidates:
        assert candidate.energies is not None
        row = {
            "instance": instance_id,
            "seed": seed,
            "n": n,
            "t": OLD_BLOCKS,
            "candidate_index": candidate.index,
            "candidate_name": candidate.name,
            "selected_by": ";".join(selected_reverse[candidate.index]),
            "diagnostic_pair_member": int(
                candidate.index in (pair_left, pair_right)
            ),
            "sequence": json.dumps(candidate.sequence, separators=(",", ":")),
            "matrix": json.dumps(candidate.matrix.tolist(), separators=(",", ":")),
            "osr_score": candidate.osr_score,
            "pauli_score": candidate.pauli_score,
            "full_projected_score": candidate.full_score,
            "delta_B": candidate.delta,
            "old_rank": candidate.old_rank,
            "candidate_rank": candidate.candidate_rank,
            "union_rank": candidate.union_rank,
            "novel_dimension": candidate.novel_dimension,
            "intersection_dimension": candidate.intersection_dimension,
            "overlap_fraction": candidate.overlap_fraction,
            "principal_angle_min_deg": candidate.principal_angle_min_deg,
            "principal_angle_median_deg": candidate.principal_angle_median_deg,
            "principal_angle_max_deg": candidate.principal_angle_max_deg,
            "principal_angle_obliqueness_deg": (
                candidate.principal_angle_obliqueness_deg
            ),
            "projector_commutator_fro": candidate.projector_commutator_fro,
            "gram_condition_number": candidate.gram_condition_number,
            "same_state_error": candidate.same_state_error,
            "same_energy_error": candidate.same_energy_error,
            "same_old_gradient_error": candidate.same_old_gradient_error,
            "same_ambient_error": candidate.same_ambient_error,
            "initial_energy": float(candidate.energies[0]),
            "final_energy": float(candidate.energies[-1]),
            "final_energy_error": float(candidate.energies[-1] - ground_energy),
            "gain_per_added_cnot_50": float(
                (candidate.energies[0] - candidate.energies[-1])
                / (2 * CNOTS_PER_CANDIDATE)
            ),
        }
        for checkpoint in CHECKPOINTS:
            row[f"energy_after_{checkpoint}"] = float(
                candidate.energies[checkpoint]
            )
            row[f"gain_after_{checkpoint}"] = float(
                candidate.energies[0] - candidate.energies[checkpoint]
            )
        candidate_rows.append(row)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SNAPSHOT_DIR / f"{instance_id}.npz",
        hamiltonian_data=hamiltonian.data,
        hamiltonian_indices=hamiltonian.indices,
        hamiltonian_indptr=hamiltonian.indptr,
        hamiltonian_shape=np.asarray(hamiltonian.shape),
        ground_energy=np.asarray(ground_energy),
        ground_state=ground_state,
        old_state=old_state,
        old_energy=np.asarray(old_energy),
        old_theta=old_theta,
        old_gradient=old_gradient,
        old_ambient_real=old_geometry.ambient_real,
        old_tangent_real=old_geometry.tangent_real,
        old_gram=old_geometry.gram,
        warmup_first_moment=warmup.first_moment,
        warmup_second_moment=warmup.second_moment,
        warmup_step=np.asarray(warmup.step),
        warmup_energies=warmup.energies,
        final_optimizer_gradient=np.asarray(optimizer_metadata["final_gradient_norm"]),
        optimizer_metadata=np.asarray(json.dumps(optimizer_metadata, ensure_ascii=False)),
    )
    trace_data = {
        f"{instance_id}__{candidate.name}": candidate.energies
        for candidate in candidates
        if candidate.energies is not None
    }
    trace_data[f"{instance_id}__ground_energy"] = np.asarray([ground_energy])
    summary: dict[str, object] = {
        "instance": instance_id,
        "seed": seed,
        "n": n,
        "t": OLD_BLOCKS,
        "ground_energy": ground_energy,
        "old_energy": old_energy,
        "old_energy_error": old_energy_error,
        "old_ambient_norm": old_geometry.ambient_norm,
        "old_projected_norm": old_geometry.projected_norm,
        "old_chi": old_geometry.chi,
        "old_parameter_gradient_norm": float(np.linalg.norm(old_gradient)),
        "old_state_distance_from_zero": state_distance,
        "oracle_gain_50": oracle_gain,
        "random_mean_gain_50": float(np.mean(random_gains)),
        "random_near_oracle_probability": near_oracle_probability,
        "selectors": selected,
        "diagnostic_pair": [pair_left, pair_right],
        "diagnostic_pair_log10_pauli_gap": pair_pauli_gap,
        "maximum_same_state_error": max(
            candidate.same_state_error for candidate in candidates
        ),
        "maximum_same_energy_error": max(
            candidate.same_energy_error for candidate in candidates
        ),
        "maximum_same_old_gradient_error": max(
            candidate.same_old_gradient_error for candidate in candidates
        ),
        "maximum_same_ambient_error": max(
            candidate.same_ambient_error for candidate in candidates
        ),
        "optimizer": optimizer_metadata,
    }
    return result_rows, candidate_rows, summary, trace_data


def finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def quartiles(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return tuple(float(value) for value in np.quantile(array, (0.25, 0.5, 0.75)))


def score_correlations(
    candidate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        if int(row["candidate_index"]) != 0:
            grouped[str(row["instance"])].append(row)
    outputs: list[dict[str, object]] = []
    score_fields = (
        ("OSR", "osr_score"),
        ("Pauli", "pauli_score"),
        ("projection", "delta_B"),
    )
    for instance, rows in grouped.items():
        for score_name, score_field in score_fields:
            x = np.asarray([float(row[score_field]) for row in rows])
            for checkpoint in CHECKPOINTS:
                y = np.asarray(
                    [float(row[f"gain_after_{checkpoint}"]) for row in rows]
                )
                if np.unique(x).size < 2 or np.unique(y).size < 2:
                    rho = float("nan")
                else:
                    rho = float(spearmanr(x, y).statistic)
                outputs.append(
                    {
                        "instance": instance,
                        "n": int(rows[0]["n"]),
                        "score": score_name,
                        "checkpoint": checkpoint,
                        "spearman_rho": rho,
                    }
                )
    return outputs


def method_summaries(
    result_rows: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    rows = [row for row in result_rows if row["row_type"] == "method_summary"]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output: dict[str, dict[str, float]] = {}
    for method, method_rows in grouped.items():
        gains = [float(row["gain_50"]) for row in method_rows]
        errors = [float(row["final_energy_error"]) for row in method_rows]
        percentiles = [
            float(row["random_percentile_50"])
            for row in method_rows
            if row.get("random_percentile_50", "") != ""
        ]
        output[method] = {
            "mean_gain_50": float(np.mean(gains)),
            "median_gain_50": float(np.median(gains)),
            "median_final_energy_error": float(np.median(errors)),
            "median_random_percentile": (
                float(np.median(percentiles)) if percentiles else float("nan")
            ),
        }
    return output


def make_figure(
    result_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    traces: dict[str, np.ndarray],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.4,
            "axes.labelsize": 9,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.2,
            "figure.dpi": 180,
        }
    )
    colors = {
        "identity": "#777777",
        "random": "#e69f00",
        "osr_only": "#2a9d8f",
        "pauli_only": "#5b8ff9",
        "projection": "#c44e7d",
        "oracle": "#222222",
    }
    labels = {
        "identity": "identity",
        "random": "random",
        "osr_only": "OSR-only",
        "pauli_only": "candidate-only",
        "projection": "incremental",
        "oracle": "oracle",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.25, 6.35), constrained_layout=True)
    method_rows = [
        row for row in result_rows if row["row_type"] == "method_summary"
    ]
    representative_summary = sorted(
        [summary for summary in summaries if int(summary["n"]) == 8],
        key=lambda item: int(item["seed"]),
    )[0]
    representative = str(representative_summary["instance"])
    representative_methods = {
        str(row["method"]): row
        for row in method_rows
        if row["instance"] == representative
    }
    representative_candidates = [
        row for row in candidate_rows if row["instance"] == representative
    ]
    ground_energy = float(representative_summary["ground_energy"])
    nonidentity_traces = np.stack(
        [
            traces[f"{representative}__{row['candidate_name']}"]
            for row in representative_candidates
            if int(row["candidate_index"]) != 0
        ]
    )
    steps = np.arange(INTERVENTION_STEPS + 1)
    random_error = nonidentity_traces - ground_energy
    axes[0, 0].fill_between(
        steps,
        np.quantile(random_error, 0.1, axis=0),
        np.quantile(random_error, 0.9, axis=0),
        color=colors["random"],
        alpha=0.16,
        label="random 10--90%",
    )
    axes[0, 0].plot(
        steps,
        np.mean(random_error, axis=0),
        color=colors["random"],
        linewidth=1.2,
        label="random mean",
    )
    for method in ("identity", "osr_only", "pauli_only", "projection", "oracle"):
        row = representative_methods[method]
        candidate_name = str(row["candidate_name"])
        values = traces[f"{representative}__{candidate_name}"] - ground_energy
        axes[0, 0].plot(
            steps,
            values,
            color=colors[method],
            linewidth=1.8 if method in ("projection", "oracle") else 1.15,
            linestyle="--" if method == "oracle" else "-",
            label=labels[method],
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Optimization step")
    axes[0, 0].set_ylabel(r"Energy error $E-E_0$")
    axes[0, 0].set_title(
        "(a) Same-point TFIM intervention\n"
        f"n=8, t={OLD_BLOCKS}"
    )
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")

    nonidentity_rows = [
        row for row in representative_candidates if int(row["candidate_index"]) != 0
    ]
    scatter = axes[0, 1].scatter(
        [max(float(row["pauli_score"]), 1e-16) for row in nonidentity_rows],
        [max(float(row["delta_B"]), 1e-16) for row in nonidentity_rows],
        c=[float(row["overlap_fraction"]) for row in nonidentity_rows],
        cmap="viridis",
        s=31,
        edgecolor="white",
        linewidth=0.5,
    )
    minimum = min(
        min(float(row["pauli_score"]) for row in nonidentity_rows),
        min(float(row["delta_B"]) for row in nonidentity_rows),
    )
    maximum = max(
        max(float(row["pauli_score"]) for row in nonidentity_rows),
        max(float(row["delta_B"]) for row in nonidentity_rows),
    )
    minimum = max(minimum, 1e-16)
    axes[0, 1].plot(
        [minimum, maximum],
        [minimum, maximum],
        color="#888888",
        linewidth=0.8,
        linestyle=":",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(
        r"Candidate-only score $\|\Pi_{\mathcal{D}_B}G\|^2$"
    )
    axes[0, 1].set_ylabel(r"Incremental gain $\Delta_B$")
    axes[0, 1].set_title("(b) Candidate-only and incremental alignment")
    colorbar = fig.colorbar(scatter, ax=axes[0, 1], pad=0.01)
    colorbar.set_label("overlap fraction with old tangent")

    plot_methods = ("random", "osr_only", "pauli_only", "projection")
    ns = (4, 6, 8)
    width = 0.18
    for method_index, method in enumerate(plot_methods):
        for n_index, n in enumerate(ns):
            values = np.asarray(
                [
                    float(row["gain_50"])
                    for row in method_rows
                    if row["method"] == method and int(row["n"]) == n
                ]
            )
            center = n_index + (method_index - 1.5) * width
            axes[1, 0].bar(
                center,
                float(np.mean(values)),
                width=width * 0.88,
                color=colors[method],
                alpha=0.7,
                label=labels[method] if n_index == 0 else None,
            )
            jitter = np.linspace(-0.045, 0.045, len(values))
            axes[1, 0].scatter(
                center + jitter,
                values,
                s=13,
                facecolor="white",
                edgecolor=colors[method],
                linewidth=0.8,
                zorder=3,
            )
    axes[1, 0].set_xticks(range(len(ns)), [f"n={n}" for n in ns])
    axes[1, 0].set_ylabel("50-step energy decrease")
    axes[1, 0].set_title("(c) Selection gains after retraining")
    axes[1, 0].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )

    correlations = score_correlations(candidate_rows)
    score_order = ("OSR", "Pauli", "projection")
    score_colors = ("#2a9d8f", "#5b8ff9", "#c44e7d")
    positions = np.arange(len(score_order))
    for position, (score, color) in enumerate(zip(score_order, score_colors)):
        values = np.asarray(
            [
                float(row["spearman_rho"])
                for row in correlations
                if row["score"] == score and int(row["checkpoint"]) == 20
            ]
        )
        jitter = np.linspace(-0.12, 0.12, len(values))
        axes[1, 1].scatter(
            position + jitter,
            values,
            s=15,
            color=color,
            alpha=0.65,
        )
        axes[1, 1].plot(
            [position - 0.18, position + 0.18],
            [np.median(values), np.median(values)],
            color="#222222",
            linewidth=1.2,
        )
    axes[1, 1].axhline(0.0, color="#888888", linewidth=0.7)
    axes[1, 1].set_xticks(positions, ["OSR-only", "candidate-only", "incremental"])
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set_ylabel("Per-instance Spearman $\\rho$")
    axes[1, 1].set_title(
        "(d) Score vs 20-step decrease\n"
        f"{len(summaries)} instances, {len(nonidentity_rows)} candidates each"
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF)
    fig.savefig(FIGURE_PNG, dpi=300)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    result_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> None:
    method_summary = method_summaries(result_rows)
    correlations = score_correlations(candidate_rows)
    method_rows = [
        row for row in result_rows if row["row_type"] == "method_summary"
    ]
    method_table = []
    for method in METHODS:
        values = method_summary[method]
        method_table.append(
            [
                method,
                f"{values['mean_gain_50']:.6f}",
                f"{values['median_gain_50']:.6f}",
                f"{values['median_final_energy_error']:.6f}",
                f"{values['median_random_percentile']:.1f}",
            ]
        )
    correlation_table = []
    for score in ("OSR", "Pauli", "projection"):
        for checkpoint in CHECKPOINTS:
            values = [
                float(row["spearman_rho"])
                for row in correlations
                if row["score"] == score and int(row["checkpoint"]) == checkpoint
            ]
            q1, median, q3 = quartiles(values)
            correlation_table.append(
                [
                    score,
                    checkpoint,
                    f"{median:.3f}",
                    f"[{q1:.3f}, {q3:.3f}]",
                    f"[{min(values):.3f}, {max(values):.3f}]",
                ]
            )
    by_instance: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in method_rows:
        by_instance[str(row["instance"])][str(row["method"])] = row
    projection_vs_random = [
        float(rows["projection"]["gain_50"]) - float(rows["random"]["gain_50"])
        for rows in by_instance.values()
    ]
    pauli_vs_random = [
        float(rows["pauli_only"]["gain_50"]) - float(rows["random"]["gain_50"])
        for rows in by_instance.values()
    ]
    projection_vs_pauli = [
        float(rows["projection"]["gain_50"])
        - float(rows["pauli_only"]["gain_50"])
        for rows in by_instance.values()
    ]
    selector_differences = sum(
        int(rows["projection"]["candidate_index"])
        != int(rows["pauli_only"]["candidate_index"])
        for rows in by_instance.values()
    )
    tolerance = 1e-10

    def wtl(values: list[float]) -> tuple[int, int, int]:
        return (
            sum(value > tolerance for value in values),
            sum(abs(value) <= tolerance for value in values),
            sum(value < -tolerance for value in values),
        )

    pair_rows = []
    candidate_by_instance: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in candidate_rows:
        candidate_by_instance[str(row["instance"])][int(row["candidate_index"])] = row
    for summary in summaries:
        left_index, right_index = [
            int(value) for value in summary["diagnostic_pair"]
        ]
        left = candidate_by_instance[str(summary["instance"])][left_index]
        right = candidate_by_instance[str(summary["instance"])][right_index]
        pair_rows.append(
            [
                summary["instance"],
                f"{left['candidate_name']}/{right['candidate_name']}",
                f"{float(summary['diagnostic_pair_log10_pauli_gap']):.3f}",
                f"{float(left['overlap_fraction']):.3f}/{float(right['overlap_fraction']):.3f}",
                f"{float(left['delta_B']):.3e}/{float(right['delta_B']):.3e}",
                f"{float(left['gain_after_50']):.4f}/{float(right['gain_after_50']):.4f}",
            ]
        )
    old_chi = [float(summary["old_chi"]) for summary in summaries]
    old_ambient = [float(summary["old_ambient_norm"]) for summary in summaries]
    old_error = [float(summary["old_energy_error"]) for summary in summaries]
    random_probability = [
        float(summary["random_near_oracle_probability"]) for summary in summaries
    ]
    same_point_maxima = {
        key: max(float(summary[key]) for summary in summaries)
        for key in (
            "maximum_same_state_error",
            "maximum_same_energy_error",
            "maximum_same_old_gradient_error",
            "maximum_same_ambient_error",
        )
    }
    nonidentity_candidates = [
        row for row in candidate_rows if int(row["candidate_index"]) != 0
    ]
    relative_score_differences = [
        abs(float(row["delta_B"]) - float(row["pauli_score"]))
        / max(abs(float(row["delta_B"])), abs(float(row["pauli_score"])), 1e-16)
        for row in nonidentity_candidates
    ]
    maximum_obliqueness = max(
        float(row["principal_angle_obliqueness_deg"])
        for row in nonidentity_candidates
    )
    maximum_commutator = max(
        float(row["projector_commutator_fro"])
        for row in nonidentity_candidates
    )
    projection_wtl = wtl(projection_vs_random)
    pauli_wtl = wtl(pauli_vs_random)
    projection_pauli_wtl = wtl(projection_vs_pauli)
    projection_extra = (
        selector_differences > len(summaries) // 3
        and np.mean(projection_vs_pauli) > 1e-4
        and projection_pauli_wtl[0] > projection_pauli_wtl[2]
    )

    text = json.dumps({
        "instances": len(summaries),
        "method_summary": method_summary,
        "score_correlations": correlations,
        "same_point_maxima": same_point_maxima,
        "old_coverage_squared_range": [min(old_chi), max(old_chi)],
        "old_ambient_norm_range": [min(old_ambient), max(old_ambient)],
        "old_energy_error_range": [min(old_error), max(old_error)],
        "incremental_minus_random": projection_vs_random,
        "candidate_only_minus_random": pauli_vs_random,
        "incremental_minus_candidate_only": projection_vs_pauli,
        "selector_differences": selector_differences,
        "incremental_vs_random_wtl": projection_wtl,
        "candidate_only_vs_random_wtl": pauli_wtl,
        "incremental_vs_candidate_only_wtl": projection_pauli_wtl,
        "tie_tolerance": tolerance,
        "random_near_oracle_probabilities": random_probability,
        "relative_score_difference_denominator_floor": 1e-16,
        "maximum_relative_score_difference": max(relative_score_differences),
        "maximum_obliqueness_deg": maximum_obliqueness,
        "maximum_projector_commutator_fro": maximum_commutator,
        "diagnostic_pair_columns": ["instance", "candidates", "log10_score_gap", "overlap_fractions", "incremental_scores", "gains_50"],
        "diagnostic_pairs": pair_rows,
    }, indent=2) + "\n"
    REPORT_FILE.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-point routing intervention on a natural TFIM task."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds-per-size", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--random-draws", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    gradient_check = finite_difference_gradient_check()
    sizes = (4,) if args.smoke else (4, 6, 8)
    seeds_per_size = 1 if args.smoke else args.seeds_per_size
    pool_size = 4 if args.smoke else args.pool_size
    random_draws = 100 if args.smoke else args.random_draws
    all_results: list[dict[str, object]] = []
    all_candidates: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    traces: dict[str, np.ndarray] = {}
    for n in sizes:
        for seed_index in range(seeds_per_size):
            seed = BASE_SEED + 1000 * n + 37 * seed_index
            result_rows, candidate_rows, summary, trace_data = run_instance(
                n=n,
                seed=seed,
                pool_size=pool_size,
                random_draws=random_draws,
                smoke=args.smoke,
            )
            all_results.extend(result_rows)
            all_candidates.extend(candidate_rows)
            summaries.append(summary)
            traces.update(trace_data)
            print(
                f"finished {summary['instance']}: "
                f"old error={summary['old_energy_error']:.6g}, "
                f"chi={summary['old_chi']:.3e}, "
                f"projection={summary['selectors']['projection']}, "
                f"pauli={summary['selectors']['pauli_only']}",
                flush=True,
            )
    write_csv(RESULTS_CSV, all_results)
    write_csv(CANDIDATES_CSV, all_candidates)
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRACE_FILE, **traces)
    if not args.smoke:
        make_figure(all_results, all_candidates, summaries, traces)
        write_report(all_results, all_candidates, summaries)
    manifest = {
        "experiment": "natural_hamiltonian_same_point_intervention",
        "task": {
            "family": "1D open-boundary transverse-field Ising model",
            "coupling_J": J_COUPLING,
            "transverse_field_h": TRANSVERSE_FIELD,
            "sizes": list(sizes),
        },
        "old_ansatz": {
            "blocks": OLD_BLOCKS,
            "routing": "repeated disjoint dimers",
            "local_layer": "Rx-Ry-Rz on every qubit",
        },
        "candidate_protocol": {
            "pool_size_nonidentity": pool_size,
            "cnot_slots": CNOTS_PER_CANDIDATE,
            "gadget_cnot_count": 2 * CNOTS_PER_CANDIDATE,
            "parameters_added": "3n",
            "intervention_steps": INTERVENTION_STEPS,
            "random_draws_per_instance": random_draws,
            "candidate_generation_uses_ground_state": False,
        },
        "instances": summaries,
        "gradient_check": gradient_check,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "runtime_seconds": time.time() - started,
        "smoke": bool(args.smoke),
        "outputs": {},
    }
    output_paths = [RESULTS_CSV, CANDIDATES_CSV, TRACE_FILE]
    if not args.smoke:
        output_paths.extend([REPORT_FILE, FIGURE_PDF, FIGURE_PNG])
    manifest["outputs"] = {
        str(path.relative_to(ROOT)): {
            "sha256": hash_file(path),
            "bytes": path.stat().st_size,
        }
        for path in output_paths
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {len(summaries)} TFIM instances in "
        f"{time.time() - started:.1f} seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
