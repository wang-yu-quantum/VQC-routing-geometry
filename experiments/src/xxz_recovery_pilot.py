from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata, spearmanr

from multiblock_stagnation_intervention import (
    Gate,
    PAULI,
    add_cnot_sequence,
    add_local_layer,
    apply_gate,
    apply_one_qubit,
    horizontalize,
    realify,
    simulate_with_tangents,
    zero_state,
)
from VQC.routing_geometry.core import routing_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DELTA = 1.0
BASE_SEED = 20260804
SIZES = (8, 10, 12)
SEEDS_PER_SIZE = 5
BASELINE_BLOCKS = 2
DEFAULT_CANDIDATES = 12
POST_STEPS = 50
CHECKPOINT_WINDOW = 20
METHODS = ("no_augmentation", "random", "pauli", "gram", "oracle")

CHECKPOINT_CSV = HERE / "xxz_checkpoints.csv"
CANDIDATE_CSV = HERE / "xxz_candidates.csv"
METHOD_CSV = HERE / "xxz_method_results.csv"
SUMMARY_CSV = HERE / "xxz_summary_table.csv"
SUMMARY_TEX = HERE / "xxz_summary_table.tex"
TRACE_FILE = HERE / "xxz_traces.npz"
FIGURE_PDF = HERE / "fig_xxz_recovery_pilot.pdf"
FIGURE_PNG = HERE / "fig_xxz_recovery_pilot.png"
REPORT_FILE = HERE / 'xxz_recovery_pilot_statistics.json'
MANIFEST_FILE = HERE / "xxz_manifest.json"
CACHE_DIR = HERE / "cache"


@dataclass
class OptimizationTrace:
    theta: np.ndarray
    energies: np.ndarray
    gradients: np.ndarray
    thetas: list[np.ndarray]
    optimizer_message: str


@dataclass
class Checkpoint:
    theta: np.ndarray
    energy: float
    gradient: np.ndarray
    trajectory_step: int
    gain_20: float
    variation_20: float
    threshold_20: float
    energies: np.ndarray
    optimizer_message: str


@dataclass
class StateGeometry:
    state: np.ndarray
    energy: float
    ambient: np.ndarray
    ambient_real: np.ndarray
    ambient_norm: float
    tangent_real: np.ndarray
    tangent_basis: np.ndarray
    tangent_rank: int
    projected_squared: float
    projected_norm: float
    chi: float
    parameter_gradient: np.ndarray


@dataclass
class Candidate:
    index: int
    name: str
    sequence: tuple[tuple[int, int], ...]
    matrix: np.ndarray
    raw_gradient_squared: float = 0.0
    pauli_score: float = 0.0
    gram_score: float = 0.0
    union_score: float = 0.0
    candidate_rank: int = 0
    residual_rank: int = 0
    union_rank: int = 0
    overlap_fraction: float = 0.0
    min_angle_deg: float = 90.0
    median_angle_deg: float = 90.0
    max_angle_deg: float = 90.0
    obliqueness_deg: float = 0.0
    projector_commutator_fro: float = 0.0
    gram_condition_number: float = float("inf")
    same_state_error: float = 0.0
    same_energy_error: float = 0.0
    same_old_gradient_error: float = 0.0
    energies: np.ndarray | None = None


def xxz_hamiltonian(n: int, delta: float = DELTA) -> sp.csr_matrix:
    """Open XXZ chain in the same big-endian qubit convention as the circuit code."""
    dimension = 2**n
    indices = np.arange(dimension, dtype=np.int64)
    diagonal = np.zeros(dimension, dtype=float)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for qubit in range(n - 1):
        left_mask = 1 << (n - 1 - qubit)
        right_mask = 1 << (n - 2 - qubit)
        left = (indices & left_mask) != 0
        right = (indices & right_mask) != 0
        opposite = left != right
        diagonal += delta * np.where(opposite, -1.0, 1.0)
        active_rows = indices[opposite]
        rows.append(active_rows)
        cols.append(active_rows ^ left_mask ^ right_mask)
        data.append(np.full(active_rows.size, 2.0, dtype=np.complex128))
    hamiltonian = sp.diags(diagonal, format="csr", dtype=np.complex128)
    hamiltonian += sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dimension, dimension),
    )
    return hamiltonian


def exact_ground_energy(
    hamiltonian: sp.csr_matrix,
    n: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(880301 + n)
    v0 = rng.normal(size=hamiltonian.shape[0])
    v0 /= np.linalg.norm(v0)
    values, vectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        v0=v0.astype(np.complex128),
        tol=2e-12,
        maxiter=20000,
    )
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
    for gate_index in range(len(ops) - 1, -1, -1):
        gate = ops[gate_index]
        if gate.param >= 0:
            axis = gate.kind[1:]
            matrix = (-0.5j * PAULI[axis]) @ rotation_matrix(gate, theta)
            derivative = apply_one_qubit(forward[gate_index], matrix, gate.q0, n)
            gradient[gate.param] += 2.0 * float(np.real(np.vdot(co_state, derivative)))
        co_state = apply_gate(co_state, gate, theta, n, dagger=True)
    return energy, gradient, state


def rotation_matrix(gate: Gate, theta: np.ndarray) -> np.ndarray:
    angle = theta[gate.param] if gate.param >= 0 else gate.angle
    pauli = PAULI[gate.kind[1:]]
    return np.cos(angle / 2.0) * np.eye(2) - 1j * np.sin(angle / 2.0) * pauli


def brickwall_sequence(n: int, layer: int) -> tuple[tuple[int, int], ...]:
    parity = layer % 2
    edges = [(qubit, qubit + 1) for qubit in range(parity, n - 1, 2)]
    if layer % 4 in (1, 2):
        edges = [(target, control) for control, target in edges]
    return tuple(edges)


def build_connected_ansatz(n: int) -> tuple[list[Gate], int, int]:
    ops: list[Gate] = []
    parameter = add_local_layer(ops, n, 0)
    add_cnot_sequence(ops, brickwall_sequence(n, 0))
    parameter = add_local_layer(ops, n, parameter)
    insertion_index = len(ops)
    add_cnot_sequence(ops, brickwall_sequence(n, 1))
    parameter = add_local_layer(ops, n, parameter)
    return ops, parameter, insertion_index


def initial_neel_theta(n: int, parameter_count: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.normal(0.0, 0.045, size=parameter_count)
    for qubit in range(1, n, 2):
        theta[3 * qubit + 1] += np.pi
    return theta


def adam_fixed_steps(
    ops: list[Gate],
    theta0: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = theta0.copy()
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    initial_energy = energy_and_gradient(ops, theta, hamiltonian, n)[0]
    energies = [initial_energy]
    for step in range(1, steps + 1):
        _, gradient, _ = energy_and_gradient(ops, theta, hamiltonian, n)
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        decay = 0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * (step - 1) / steps))
        theta -= learning_rate * decay * first_hat / (np.sqrt(second_hat) + 1e-8)
        energies.append(energy_and_gradient(ops, theta, hamiltonian, n)[0])
    return theta, np.asarray(energies)


def baseline_optimization(
    ops: list[Gate],
    initial_theta: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
    ground_energy: float,
    smoke: bool,
) -> Checkpoint:
    warmup_steps = 35 if smoke else 140
    theta, warmup_energies = adam_fixed_steps(
        ops,
        initial_theta,
        hamiltonian,
        n,
        steps=warmup_steps,
        learning_rate=0.025,
    )
    thetas = [theta.copy()]
    energy, gradient, _ = energy_and_gradient(ops, theta, hamiltonian, n)
    lbfgs_energies = [energy]
    lbfgs_gradients = [gradient.copy()]

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        value, derivative, _ = energy_and_gradient(ops, values, hamiltonian, n)
        return value, derivative

    def callback(values: np.ndarray) -> None:
        value, derivative, _ = energy_and_gradient(ops, values, hamiltonian, n)
        thetas.append(np.asarray(values, dtype=float).copy())
        lbfgs_energies.append(value)
        lbfgs_gradients.append(derivative.copy())

    result = minimize(
        objective,
        theta,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": 35 if smoke else 260,
            "gtol": 2e-7 if smoke else 2e-9,
            "ftol": 1e-11 if smoke else 2e-14,
            "maxls": 40,
        },
    )
    final_theta = np.asarray(result.x, dtype=float)
    final_energy, final_gradient, _ = energy_and_gradient(
        ops, final_theta, hamiltonian, n
    )
    if not np.array_equal(thetas[-1], final_theta):
        thetas.append(final_theta.copy())
        lbfgs_energies.append(final_energy)
        lbfgs_gradients.append(final_gradient.copy())

    # Confirm a 20-step plateau with ordinary small gradient-descent refinements.
    polish_steps = CHECKPOINT_WINDOW + 4
    theta = final_theta.copy()
    for _ in range(polish_steps):
        energy, gradient, _ = energy_and_gradient(ops, theta, hamiltonian, n)
        trial = theta - 0.02 * gradient
        trial_energy, _, _ = energy_and_gradient(ops, trial, hamiltonian, n)
        if trial_energy <= energy + 1e-14:
            theta = trial
        value, derivative, _ = energy_and_gradient(ops, theta, hamiltonian, n)
        thetas.append(theta.copy())
        lbfgs_energies.append(value)
        lbfgs_gradients.append(derivative.copy())

    energies = np.asarray(lbfgs_energies)
    gradients = np.stack(lbfgs_gradients)
    error_floor = max(1e-3, 1e-3 * (n - 1))
    selected = None
    for index in range(CHECKPOINT_WINDOW, energies.size):
        error = energies[index] - ground_energy
        gain = energies[index - CHECKPOINT_WINDOW] - energies[index]
        window = energies[index - CHECKPOINT_WINDOW : index + 1]
        variation = float(np.max(window) - np.min(window))
        threshold = max(2e-7, 2e-4 * error)
        if (
            error > error_floor
            and gain <= threshold
            and variation <= 5.0 * threshold
            and np.linalg.norm(gradients[index]) <= 2e-3
        ):
            selected = index
            break
    if selected is None:
        index = energies.size - 1
        error = energies[index] - ground_energy
        gain = (
            energies[index - CHECKPOINT_WINDOW] - energies[index]
            if index >= CHECKPOINT_WINDOW
            else float("inf")
        )
        threshold = max(2e-7, 2e-4 * error)
        raise RuntimeError(
            "No audited stagnation checkpoint: "
            f"n={n}, error={error:.4e}, gain20={gain:.4e}, "
            f"threshold={threshold:.4e}, grad={np.linalg.norm(gradients[index]):.4e}."
        )
    window = energies[selected - CHECKPOINT_WINDOW : selected + 1]
    return Checkpoint(
        theta=thetas[selected].copy(),
        energy=float(energies[selected]),
        gradient=gradients[selected].copy(),
        trajectory_step=warmup_steps + selected,
        gain_20=float(energies[selected - CHECKPOINT_WINDOW] - energies[selected]),
        variation_20=float(np.max(window) - np.min(window)),
        threshold_20=max(2e-7, 2e-4 * (energies[selected] - ground_energy)),
        energies=np.concatenate((warmup_energies, energies[1:])),
        optimizer_message=str(result.message),
    )


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


def state_geometry(
    ops: list[Gate],
    theta: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
) -> StateGeometry:
    state, tangents = simulate_with_tangents(ops, theta, n, theta.size)
    h_state = np.asarray(hamiltonian @ state).reshape(-1)
    energy = float(np.real(np.vdot(state, h_state)))
    ambient = 2.0 * (h_state - energy * state)
    tangent_real = realify(horizontalize(state, tangents))
    ambient_real = realify(ambient[:, None]).reshape(-1)
    basis, _ = orthonormal_basis(tangent_real)
    projected = float(np.linalg.norm(basis.T @ ambient_real)) if basis.size else 0.0
    ambient_norm = float(np.linalg.norm(ambient_real))
    return StateGeometry(
        state=state,
        energy=energy,
        ambient=ambient,
        ambient_real=ambient_real,
        ambient_norm=ambient_norm,
        tangent_real=tangent_real,
        tangent_basis=basis,
        tangent_rank=basis.shape[1],
        projected_squared=projected**2,
        projected_norm=projected,
        chi=(projected**2 / ambient_norm**2) if ambient_norm > 1e-15 else 0.0,
        parameter_gradient=tangent_real.T @ ambient_real,
    )


def candidate_pool(
    n: int,
    pool_size: int,
    rng: np.random.Generator,
) -> list[Candidate]:
    edges = [(qubit, qubit + 1) for qubit in range(n - 1)]
    candidates: list[Candidate] = []
    seen: set[bytes] = set()
    attempts = 0
    while len(candidates) < pool_size and attempts < 100000:
        attempts += 1
        order = rng.permutation(n - 1)
        sequence = []
        for edge_index in order:
            left, right = edges[int(edge_index)]
            if int(rng.integers(0, 2)):
                sequence.append((right, left))
            else:
                sequence.append((left, right))
        sequence_tuple = tuple(sequence)
        matrix = routing_matrix(n, sequence_tuple)
        key = matrix.tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                index=len(candidates),
                name=f"B{len(candidates):02d}",
                sequence=sequence_tuple,
                matrix=matrix,
            )
        )
    if len(candidates) != pool_size:
        raise RuntimeError(f"Could not create {pool_size} connected fixed-cost candidates.")
    return candidates


def build_augmented_ops(
    old_ops: list[Gate],
    insertion_index: int,
    sequence: tuple[tuple[int, int], ...],
    n: int,
    old_parameter_count: int,
) -> list[Gate]:
    ops = list(old_ops[:insertion_index])
    add_cnot_sequence(ops, sequence)
    add_local_layer(ops, n, old_parameter_count)
    add_cnot_sequence(ops, reversed(sequence))
    ops.extend(old_ops[insertion_index:])
    return ops


def candidate_tangents_at_zero(
    old_ops: list[Gate],
    insertion_index: int,
    sequence: tuple[tuple[int, int], ...],
    old_theta: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_state = zero_state(n)
    for gate in old_ops[:insertion_index]:
        prefix_state = apply_gate(prefix_state, gate, old_theta, n)
    routed_state = prefix_state
    for control, target in sequence:
        routed_state = apply_gate(
            routed_state, Gate("CNOT", control, target), old_theta, n
        )
    tangents = np.empty((2**n, 3 * n), dtype=np.complex128)
    column = 0
    for qubit in range(n):
        for axis in ("X", "Y", "Z"):
            tangents[:, column] = apply_one_qubit(
                routed_state, -0.5j * PAULI[axis], qubit, n
            )
            column += 1
    state = routed_state
    for control, target in reversed(sequence):
        gate = Gate("CNOT", control, target)
        state = apply_gate(state, gate, old_theta, n)
        tangents = apply_gate(tangents, gate, old_theta, n)
    for gate in old_ops[insertion_index:]:
        state = apply_gate(state, gate, old_theta, n)
        tangents = apply_gate(tangents, gate, old_theta, n)
    return state, tangents


def tangent_metrics(
    old_basis: np.ndarray,
    candidate_real: np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    candidate_basis, candidate_singular = orthonormal_basis(candidate_real)
    residual = candidate_real - old_basis @ (old_basis.T @ candidate_real)
    residual_basis, _ = orthonormal_basis(residual)
    union_basis, union_singular = orthonormal_basis(
        np.concatenate((old_basis, candidate_real), axis=1)
    )
    if old_basis.shape[1] and candidate_basis.shape[1]:
        cosines = np.clip(
            np.linalg.svd(old_basis.T @ candidate_basis, compute_uv=False), 0.0, 1.0
        )
    else:
        cosines = np.empty(0)
    padded = np.zeros(candidate_basis.shape[1])
    padded[: cosines.size] = cosines
    angles = np.degrees(np.arccos(padded)) if padded.size else np.array([90.0])
    overlap = (
        float(np.sum(cosines**2) / candidate_basis.shape[1])
        if candidate_basis.shape[1]
        else 0.0
    )
    commutator = float(
        np.sqrt(2.0 * np.sum(cosines**2 * (1.0 - cosines**2)))
    )
    condition = float("inf")
    if union_singular.size and union_singular[-1] > 0:
        condition = float((union_singular[0] / union_singular[-1]) ** 2)
    return {
        "candidate_basis": candidate_basis,
        "residual_basis": residual_basis,
        "union_basis": union_basis,
        "candidate_rank": candidate_basis.shape[1],
        "residual_rank": residual_basis.shape[1],
        "union_rank": union_basis.shape[1],
        "overlap_fraction": overlap,
        "min_angle_deg": float(np.min(angles)),
        "median_angle_deg": float(np.median(angles)),
        "max_angle_deg": float(np.max(angles)),
        "obliqueness_deg": float(np.max(np.minimum(angles, 90.0 - angles))),
        "projector_commutator_fro": commutator,
        "gram_condition_number": condition,
        "candidate_singular_min": (
            float(candidate_singular[-1]) if candidate_singular.size else 0.0
        ),
    }


def score_candidates(
    candidates: list[Candidate],
    old_ops: list[Gate],
    insertion_index: int,
    old_theta: np.ndarray,
    old_geometry: StateGeometry,
    hamiltonian: sp.csr_matrix,
    n: int,
) -> dict[int, list[Gate]]:
    augmented: dict[int, list[Gate]] = {}
    old_energy, old_gradient, old_state = energy_and_gradient(
        old_ops, old_theta, hamiltonian, n
    )
    ambient_residual = old_geometry.ambient_real - old_geometry.tangent_basis @ (
        old_geometry.tangent_basis.T @ old_geometry.ambient_real
    )
    for candidate in candidates:
        ops = build_augmented_ops(
            old_ops, insertion_index, candidate.sequence, n, old_theta.size
        )
        theta = np.concatenate((old_theta, np.zeros(3 * n)))
        state, tangent = candidate_tangents_at_zero(
            old_ops,
            insertion_index,
            candidate.sequence,
            old_theta,
            n,
        )
        tangent_real = realify(horizontalize(old_state, tangent))
        metrics = tangent_metrics(old_geometry.tangent_basis, tangent_real)
        candidate_basis = metrics.pop("candidate_basis")
        residual_basis = metrics.pop("residual_basis")
        union_basis = metrics.pop("union_basis")
        for key, value in metrics.items():
            if hasattr(candidate, key):
                setattr(candidate, key, value)
        coordinate_gradient = tangent_real.T @ old_geometry.ambient_real
        candidate.raw_gradient_squared = float(coordinate_gradient @ coordinate_gradient)
        candidate.pauli_score = float(
            np.linalg.norm(candidate_basis.T @ old_geometry.ambient_real) ** 2
        )
        candidate.gram_score = float(
            np.linalg.norm(residual_basis.T @ ambient_residual) ** 2
        )
        candidate.union_score = float(
            np.linalg.norm(union_basis.T @ old_geometry.ambient_real) ** 2
            - old_geometry.projected_squared
        )
        full_energy, full_gradient, full_state = energy_and_gradient(
            ops, theta, hamiltonian, n
        )
        candidate.same_state_error = max(
            float(np.linalg.norm(state - old_state)),
            float(np.linalg.norm(full_state - old_state)),
        )
        candidate.same_energy_error = abs(full_energy - old_energy)
        candidate.same_old_gradient_error = float(
            np.linalg.norm(full_gradient[: old_theta.size] - old_gradient)
        )
        if (
            candidate.same_state_error > 3e-11
            or candidate.same_energy_error > 3e-11
            or candidate.same_old_gradient_error > 3e-9
            or abs(candidate.gram_score - candidate.union_score) > 2e-8
        ):
            raise AssertionError(
                f"Same-point/Gram audit failed for {candidate.name}: "
                f"state={candidate.same_state_error:.3e}, "
                f"energy={candidate.same_energy_error:.3e}, "
                f"old_grad={candidate.same_old_gradient_error:.3e}, "
                f"delta={abs(candidate.gram_score-candidate.union_score):.3e}."
            )
        augmented[candidate.index] = ops
    return augmented


def train_all_candidates(
    candidates: list[Candidate],
    augmented_ops: dict[int, list[Gate]],
    old_theta: np.ndarray,
    hamiltonian: sp.csr_matrix,
    n: int,
    smoke: bool,
) -> None:
    steps = 5 if smoke else POST_STEPS
    for candidate in candidates:
        theta = np.concatenate((old_theta, np.zeros(3 * n)))
        _, energies = adam_fixed_steps(
            augmented_ops[candidate.index],
            theta,
            hamiltonian,
            n,
            steps=steps,
            learning_rate=0.02,
        )
        if smoke:
            energies = np.pad(energies, (0, POST_STEPS + 1 - energies.size), mode="edge")
        candidate.energies = energies


def finite_difference_gradient_check() -> dict[str, float]:
    n = 4
    rng = np.random.default_rng(99117)
    hamiltonian = xxz_hamiltonian(n)
    ops, count, _ = build_connected_ansatz(n)
    theta = rng.normal(0.0, 0.2, size=count)
    _, analytic, _ = energy_and_gradient(ops, theta, hamiltonian, n)
    epsilon = 1e-6
    numerical = np.zeros_like(theta)
    for index in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (
            energy_and_gradient(ops, plus, hamiltonian, n)[0]
            - energy_and_gradient(ops, minus, hamiltonian, n)[0]
        ) / (2.0 * epsilon)
    absolute = float(np.max(np.abs(analytic - numerical)))
    relative = absolute / max(float(np.max(np.abs(numerical))), 1e-15)
    if absolute > 3e-7 or relative > 3e-6:
        raise AssertionError(
            f"Gradient check failed: absolute={absolute}, relative={relative}."
        )
    singlet_energy = exact_ground_energy(xxz_hamiltonian(2), 2)[0]
    if abs(singlet_energy + 3.0) > 1e-10:
        raise AssertionError(f"Two-qubit XXZ check failed: E0={singlet_energy}.")
    connected_edges = {
        tuple(sorted(edge))
        for layer in range(BASELINE_BLOCKS)
        for edge in brickwall_sequence(n, layer)
    }
    expected_edges = {(qubit, qubit + 1) for qubit in range(n - 1)}
    if connected_edges != expected_edges:
        raise AssertionError("Brick-wall union does not connect the open chain.")
    return {
        "maximum_absolute_gradient_error": absolute,
        "maximum_relative_gradient_error": relative,
        "two_qubit_ground_energy_error": abs(singlet_energy + 3.0),
    }


def safe_spearman(x: list[float], y: list[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if np.unique(x_array).size < 2 or np.unique(y_array).size < 2:
        return float("nan")
    return float(spearmanr(x_array, y_array).statistic)


def run_instance(
    n: int,
    seed: int,
    pool_size: int,
    smoke: bool,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, np.ndarray],
]:
    started = time.time()
    rng = np.random.default_rng(seed)
    hamiltonian = xxz_hamiltonian(n)
    ground_energy, _ = exact_ground_energy(hamiltonian, n)
    old_ops, parameter_count, insertion_index = build_connected_ansatz(n)
    initial_theta = initial_neel_theta(n, parameter_count, rng)
    checkpoint = baseline_optimization(
        old_ops, initial_theta, hamiltonian, n, ground_energy, smoke
    )
    old_geometry = state_geometry(old_ops, checkpoint.theta, hamiltonian, n)
    gradient_crosscheck = float(
        np.linalg.norm(old_geometry.parameter_gradient - checkpoint.gradient)
    )
    if gradient_crosscheck > 3e-8:
        raise AssertionError(
            f"State-tangent/adjoint gradient mismatch for n={n}, seed={seed}: "
            f"{gradient_crosscheck:.3e}."
        )
    if old_geometry.ambient_norm <= 1e-5:
        raise AssertionError(f"Ambient state gradient vanished for n={n}, seed={seed}.")
    candidates = candidate_pool(n, pool_size, rng)
    augmented_ops = score_candidates(
        candidates,
        old_ops,
        insertion_index,
        checkpoint.theta,
        old_geometry,
        hamiltonian,
        n,
    )
    train_all_candidates(
        candidates, augmented_ops, checkpoint.theta, hamiltonian, n, smoke
    )
    _, no_energies = adam_fixed_steps(
        old_ops,
        checkpoint.theta,
        hamiltonian,
        n,
        steps=5 if smoke else POST_STEPS,
        learning_rate=0.02,
    )
    if smoke:
        no_energies = np.pad(
            no_energies, (0, POST_STEPS + 1 - no_energies.size), mode="edge"
        )

    gains = {
        candidate.index: float(candidate.energies[0] - candidate.energies[-1])
        for candidate in candidates
        if candidate.energies is not None
    }
    random_index = int(rng.integers(0, len(candidates)))

    def stable_argmax(attribute: str) -> int:
        values = np.asarray([float(getattr(candidate, attribute)) for candidate in candidates])
        maximum = float(np.max(values))
        tolerance = 1e-10 * max(1.0, abs(maximum))
        return min(
            candidate.index
            for candidate, value in zip(candidates, values)
            if value >= maximum - tolerance
        )

    selected = {
        "random": random_index,
        "pauli": stable_argmax("pauli_score"),
        "gram": stable_argmax("gram_score"),
        "oracle": max(candidates, key=lambda item: (gains[item.index], -item.index)).index,
    }
    by_index = {candidate.index: candidate for candidate in candidates}
    oracle_gain = gains[selected["oracle"]]
    no_gain = float(no_energies[0] - no_energies[-1])
    recovery_threshold = max(1e-5, 0.01 * (checkpoint.energy - ground_energy))
    instance = f"xxz_n{n}_s{seed}"

    candidate_rows: list[dict[str, object]] = []
    for candidate in candidates:
        assert candidate.energies is not None
        gain = gains[candidate.index]
        candidate_rows.append(
            {
                "instance": instance,
                "n": n,
                "seed": seed,
                "candidate_index": candidate.index,
                "candidate_name": candidate.name,
                "sequence": json.dumps(candidate.sequence, separators=(",", ":")),
                "routing_matrix": json.dumps(candidate.matrix.tolist(), separators=(",", ":")),
                "raw_zero_gradient_squared": candidate.raw_gradient_squared,
                "pauli_score": candidate.pauli_score,
                "gram_residual_score": candidate.gram_score,
                "gram_union_crosscheck": candidate.union_score,
                "candidate_rank": candidate.candidate_rank,
                "residual_rank": candidate.residual_rank,
                "union_rank": candidate.union_rank,
                "overlap_fraction": candidate.overlap_fraction,
                "principal_angle_min_deg": candidate.min_angle_deg,
                "principal_angle_median_deg": candidate.median_angle_deg,
                "principal_angle_max_deg": candidate.max_angle_deg,
                "principal_angle_obliqueness_deg": candidate.obliqueness_deg,
                "projector_commutator_fro": candidate.projector_commutator_fro,
                "gram_condition_number": candidate.gram_condition_number,
                "same_state_error": candidate.same_state_error,
                "same_energy_error": candidate.same_energy_error,
                "same_old_gradient_error": candidate.same_old_gradient_error,
                "gain_50": gain,
                "final_energy": float(candidate.energies[-1]),
                "final_energy_error": float(candidate.energies[-1] - ground_energy),
                "gain_per_added_cnot": gain / (2 * (n - 1)),
                "oracle_regret": oracle_gain - gain,
                "selected_random": int(candidate.index == selected["random"]),
                "selected_pauli": int(candidate.index == selected["pauli"]),
                "selected_gram": int(candidate.index == selected["gram"]),
                "selected_oracle": int(candidate.index == selected["oracle"]),
            }
        )

    method_rows: list[dict[str, object]] = []
    method_candidates: dict[str, Candidate | None] = {
        "no_augmentation": None,
        "random": by_index[selected["random"]],
        "pauli": by_index[selected["pauli"]],
        "gram": by_index[selected["gram"]],
        "oracle": by_index[selected["oracle"]],
    }
    for method, candidate in method_candidates.items():
        energies = no_energies if candidate is None else candidate.energies
        assert energies is not None
        gain = float(energies[0] - energies[-1])
        method_rows.append(
            {
                "instance": instance,
                "n": n,
                "seed": seed,
                "method": method,
                "candidate_index": -1 if candidate is None else candidate.index,
                "candidate_name": "none" if candidate is None else candidate.name,
                "parameters_added": 0 if candidate is None else 3 * n,
                "cnot_added": 0 if candidate is None else 2 * (n - 1),
                "post_steps": POST_STEPS,
                "initial_energy": float(energies[0]),
                "final_energy": float(energies[-1]),
                "final_energy_error": float(energies[-1] - ground_energy),
                "gain_50": gain,
                "gain_per_added_cnot": (
                    float("nan") if candidate is None else gain / (2 * (n - 1))
                ),
                "oracle_regret": oracle_gain - gain,
                "recovery_success": int(gain > no_gain + recovery_threshold),
            }
        )

    candidate_gains = [gains[candidate.index] for candidate in candidates]
    pauli_rho = safe_spearman(
        [candidate.pauli_score for candidate in candidates], candidate_gains
    )
    gram_rho = safe_spearman(
        [candidate.gram_score for candidate in candidates], candidate_gains
    )
    raw_rho = safe_spearman(
        [candidate.raw_gradient_squared for candidate in candidates], candidate_gains
    )
    pauli_ranks = rankdata([candidate.pauli_score for candidate in candidates])
    gram_ranks = rankdata([candidate.gram_score for candidate in candidates])
    ranking_changes = int(np.sum(pauli_ranks != gram_ranks))
    max_rank_shift = float(np.max(np.abs(pauli_ranks - gram_ranks)))
    elapsed = time.time() - started
    checkpoint_row: dict[str, object] = {
        "instance": instance,
        "n": n,
        "seed": seed,
        "delta": DELTA,
        "ground_energy": ground_energy,
        "checkpoint_energy": checkpoint.energy,
        "checkpoint_energy_error": checkpoint.energy - ground_energy,
        "checkpoint_parameter_gradient_norm": float(np.linalg.norm(checkpoint.gradient)),
        "checkpoint_ambient_gradient_norm": old_geometry.ambient_norm,
        "checkpoint_projected_gradient_norm": old_geometry.projected_norm,
        "checkpoint_chi": old_geometry.chi,
        "state_tangent_gradient_crosscheck": gradient_crosscheck,
        "checkpoint_trajectory_step": checkpoint.trajectory_step,
        "checkpoint_gain_20": checkpoint.gain_20,
        "checkpoint_variation_20": checkpoint.variation_20,
        "checkpoint_threshold_20": checkpoint.threshold_20,
        "old_tangent_rank": old_geometry.tangent_rank,
        "candidate_pool_size": len(candidates),
        "candidate_cnot_each_side": n - 1,
        "candidate_parameters_added": 3 * n,
        "same_state_error_max": max(item.same_state_error for item in candidates),
        "same_energy_error_max": max(item.same_energy_error for item in candidates),
        "same_old_gradient_error_max": max(
            item.same_old_gradient_error for item in candidates
        ),
        "spearman_raw_gradient_gain50": raw_rho,
        "spearman_pauli_gain50": pauli_rho,
        "spearman_gram_gain50": gram_rho,
        "pauli_gram_selected_different": int(selected["pauli"] != selected["gram"]),
        "pauli_gram_rank_entries_changed": ranking_changes,
        "pauli_gram_max_rank_shift": max_rank_shift,
        "random_expected_gain": float(np.mean(candidate_gains)),
        "random_expected_oracle_regret": oracle_gain - float(np.mean(candidate_gains)),
        "random_selected_index": selected["random"],
        "pauli_selected_index": selected["pauli"],
        "gram_selected_index": selected["gram"],
        "oracle_index": selected["oracle"],
        "runtime_seconds": elapsed,
        "peak_memory_mb": peak_memory_mb(),
        "optimizer_message": checkpoint.optimizer_message,
    }
    traces = {
        f"{instance}__baseline": checkpoint.energies,
        f"{instance}__no_augmentation": no_energies,
        f"{instance}__checkpoint_theta": checkpoint.theta,
        f"{instance}__checkpoint_state": old_geometry.state,
        f"{instance}__ambient_state_gradient": old_geometry.ambient,
        f"{instance}__old_tangent_gram": (
            old_geometry.tangent_real.T @ old_geometry.tangent_real
        ),
    }
    for method in ("random", "pauli", "gram", "oracle"):
        candidate = method_candidates[method]
        assert candidate is not None and candidate.energies is not None
        traces[f"{instance}__{method}"] = candidate.energies
    return checkpoint_row, candidate_rows, method_rows, traces


def peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        value *= 1024.0
    return value / 1024.0**2


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_cache(
    instance: str,
    checkpoint: dict[str, object],
    candidates: list[dict[str, object]],
    methods: list[dict[str, object]],
    traces: dict[str, np.ndarray],
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{instance}.json").write_text(
        json.dumps(
            {"checkpoint": checkpoint, "candidates": candidates, "methods": methods},
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(CACHE_DIR / f"{instance}.npz", **traces)


def load_cache(
    instance: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    payload = json.loads((CACHE_DIR / f"{instance}.json").read_text(encoding="utf-8"))
    with np.load(CACHE_DIR / f"{instance}.npz") as archive:
        traces = {key: archive[key] for key in archive.files}
    return payload["checkpoint"], payload["candidates"], payload["methods"], traces


def aggregate_summary(method_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for n_label, n_value in [(str(n), n) for n in sorted({int(row["n"]) for row in method_rows})] + [("all", None)]:
        for method in METHODS:
            rows = [
                row
                for row in method_rows
                if row["method"] == method and (n_value is None or int(row["n"]) == n_value)
            ]
            if not rows:
                continue
            gains = np.asarray([float(row["gain_50"]) for row in rows])
            regrets = np.asarray([float(row["oracle_regret"]) for row in rows])
            outputs.append(
                {
                    "n": n_label,
                    "method": method,
                    "instances": len(rows),
                    "mean_gain_50": float(np.mean(gains)),
                    "median_gain_50": float(np.median(gains)),
                    "min_gain_50": float(np.min(gains)),
                    "max_gain_50": float(np.max(gains)),
                    "recovery_success_rate": float(
                        np.mean([int(row["recovery_success"]) for row in rows])
                    ),
                    "mean_oracle_regret": float(np.mean(regrets)),
                    "median_oracle_regret": float(np.median(regrets)),
                    "mean_final_energy_error": float(
                        np.mean([float(row["final_energy_error"]) for row in rows])
                    ),
                }
            )
    return outputs


def write_latex_table(summary_rows: list[dict[str, object]]) -> None:
    rows = [row for row in summary_rows if row["n"] != "all" and row["method"] != "oracle"]
    lines = [
        r"\begin{tabular}{clrrrr}",
        r"\toprule",
        r"$n$ & method & mean gain & success & mean regret & final error \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['n']} & {str(row['method']).replace('_', ' ')} & "
            f"{float(row['mean_gain_50']):.4g} & "
            f"{float(row['recovery_success_rate']):.2f} & "
            f"{float(row['mean_oracle_regret']):.4g} & "
            f"{float(row['mean_final_energy_error']):.4g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    SUMMARY_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_differences(
    method_rows: list[dict[str, object]],
    left: str,
    right: str,
) -> np.ndarray:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in method_rows:
        grouped[str(row["instance"])][str(row["method"])] = float(row["gain_50"])
    return np.asarray(
        [values[left] - values[right] for values in grouped.values() if left in values and right in values]
    )


def make_figure(
    checkpoints: list[dict[str, object]],
    candidates: list[dict[str, object]],
    methods: list[dict[str, object]],
    traces: dict[str, np.ndarray],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.1,
            "figure.dpi": 180,
        }
    )
    colors = {
        "no_augmentation": "#777777",
        "random": "#e69f00",
        "pauli": "#377eb8",
        "gram": "#c44e7d",
        "oracle": "#222222",
    }
    differing = [row for row in checkpoints if int(row["pauli_gram_selected_different"])]
    labels = {
        "no_augmentation": "no augmentation",
        "random": "random",
        "pauli": "candidate-only",
        "gram": "incremental",
        "oracle": "oracle",
    }
    representative_row = (differing or sorted(checkpoints, key=lambda item: (int(item["n"]), int(item["seed"]))))[0]
    representative = str(representative_row["instance"])
    ground = float(representative_row["ground_energy"])
    fig, axes = plt.subplots(2, 2, figsize=(8.25, 6.2), constrained_layout=True)
    steps = np.arange(POST_STEPS + 1)
    for method in METHODS:
        values = traces[f"{representative}__{method}"] - ground
        axes[0, 0].plot(
            steps,
            values,
            color=colors[method],
            linewidth=1.8 if method in ("gram", "oracle") else 1.25,
            linestyle="--" if method == "oracle" else "-",
            label=labels[method],
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Post-checkpoint optimization step")
    axes[0, 0].set_ylabel(r"Energy error $E-E_0$")
    axes[0, 0].set_title(
        "(a) Representative same-point recovery"
    )
    axes[0, 0].legend(frameon=False, loc="lower left")

    completed_sizes = sorted({int(row["n"]) for row in methods})
    plot_methods = ("no_augmentation", "random", "pauli", "gram")
    width = 0.18
    for method_index, method in enumerate(plot_methods):
        for n_index, n in enumerate(completed_sizes):
            values = np.asarray(
                [
                    float(row["gain_50"])
                    for row in methods
                    if row["method"] == method and int(row["n"]) == n
                ]
            )
            center = n_index + (method_index - 1.5) * width
            axes[0, 1].bar(
                center,
                float(np.mean(values)),
                width=0.9 * width,
                color=colors[method],
                alpha=0.72,
                label=labels[method] if n_index == 0 else None,
            )
            jitter = np.linspace(-0.04, 0.04, len(values))
            axes[0, 1].scatter(
                center + jitter,
                values,
                s=14,
                facecolor="white",
                edgecolor=colors[method],
                linewidth=0.8,
                zorder=3,
            )
    axes[0, 1].set_xticks(range(len(completed_sizes)), [f"n={n}" for n in completed_sizes])
    axes[0, 1].set_ylabel("50-step energy decrease")
    axes[0, 1].set_title("(b) Five fixed seeds per completed size")
    lower, upper = axes[0, 1].get_ylim()
    axes[0, 1].set_ylim(lower, 1.22 * upper)
    axes[0, 1].legend(frameon=False, ncol=2)

    rho_positions = np.arange(len(checkpoints))
    axes[1, 0].scatter(
        rho_positions - 0.09,
        [float(row["spearman_pauli_gain50"]) for row in checkpoints],
        color=colors["pauli"],
        s=24,
        label=labels["pauli"],
    )
    axes[1, 0].scatter(
        rho_positions + 0.09,
        [float(row["spearman_gram_gain50"]) for row in checkpoints],
        color=colors["gram"],
        marker="s",
        s=22,
        label=labels["gram"],
    )
    axes[1, 0].axhline(0.0, color="#888888", linewidth=0.7)
    axes[1, 0].set_ylim(-1.05, 1.12)
    axes[1, 0].set_xlabel("Instance (grouped by size)")
    axes[1, 0].set_ylabel(r"Candidate-level Spearman $\rho$")
    axes[1, 0].set_title("(c) Local score vs 50-step gain")
    axes[1, 0].legend(frameon=False, loc="lower left")
    group_start = 0
    for n in completed_sizes:
        count = sum(int(row["n"]) == n for row in checkpoints)
        axes[1, 0].text(
            group_start + (count - 1) / 2,
            1.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
        group_start += count
        if group_start < len(checkpoints):
            axes[1, 0].axvline(group_start - 0.5, color="#bbbbbb", linewidth=0.6)

    grouped_candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        grouped_candidates[str(row["instance"])].append(row)
    rank_x: list[float] = []
    rank_y: list[float] = []
    rank_color: list[int] = []
    for checkpoint in checkpoints:
        rows = grouped_candidates[str(checkpoint["instance"])]
        rank_x.extend(rankdata([float(row["pauli_score"]) for row in rows]))
        rank_y.extend(rankdata([float(row["gram_residual_score"]) for row in rows]))
        rank_color.extend([int(checkpoint["n"])] * len(rows))
    size_colors = {8: "#7b3294", 10: "#00897b", 12: "#c99700"}
    for n, color in size_colors.items():
        indices = [i for i, size in enumerate(rank_color) if size == n]
        axes[1, 1].scatter(
            [rank_x[i] for i in indices],
            [rank_y[i] for i in indices],
            color=color,
            label=f"$n={n}$",
            s=18,
            alpha=0.7,
            edgecolor="none",
        )
    upper = max(rank_x + rank_y) + 0.5
    axes[1, 1].plot([0.5, upper], [0.5, upper], color="#777777", linestyle=":", linewidth=0.9)
    axes[1, 1].set_xlim(0.5, upper)
    axes[1, 1].set_ylim(0.5, upper)
    axes[1, 1].set_xlabel("Within-pool candidate-only rank")
    axes[1, 1].set_ylabel("Within-pool incremental rank")
    changed = sum(int(row["pauli_gram_selected_different"]) for row in checkpoints)
    axes[1, 1].set_title(f"(d) Ranking changes; selectors differ in {changed}/{len(checkpoints)} pools")
    axes[1, 1].legend(loc="lower right", frameon=False)

    fig.savefig(FIGURE_PDF)
    fig.savefig(FIGURE_PNG, dpi=300)
    plt.close(fig)


def write_report(
    checkpoints: list[dict[str, object]],
    candidates: list[dict[str, object]],
    methods: list[dict[str, object]],
    stopped_n12: bool,
) -> None:
    gram_random = paired_differences(methods, "gram", "random")
    pauli_random = paired_differences(methods, "pauli", "random")
    gram_pauli = paired_differences(methods, "gram", "pauli")
    expected_random = {
        str(row["instance"]): float(row["random_expected_gain"])
        for row in checkpoints
    }
    selected_gains: dict[str, dict[str, float]] = defaultdict(dict)
    for row in methods:
        selected_gains[str(row["instance"])][str(row["method"])] = float(
            row["gain_50"]
        )
    gram_expected = np.asarray(
        [
            values["gram"] - expected_random[instance]
            for instance, values in selected_gains.items()
        ]
    )
    pauli_expected = np.asarray(
        [
            values["pauli"] - expected_random[instance]
            for instance, values in selected_gains.items()
        ]
    )
    selector_diff = sum(int(row["pauli_gram_selected_different"]) for row in checkpoints)
    gram_better_instances = int(np.sum(gram_pauli > 1e-7))
    connected_recovery = int(
        np.sum(
            [
                int(row["recovery_success"])
                for row in methods
                if row["method"] in ("pauli", "gram")
            ]
        )
    )
    total_selected = 2 * len(checkpoints)
    c_supported = selector_diff > 0 and gram_better_instances > 0
    all_candidate_obliqueness = [
        float(row["principal_angle_obliqueness_deg"]) for row in candidates
    ]
    all_candidate_commutators = [
        float(row["projector_commutator_fro"]) for row in candidates
    ]
    pauli_rhos = [float(row["spearman_pauli_gain50"]) for row in checkpoints]
    gram_rhos = [float(row["spearman_gram_gain50"]) for row in checkpoints]
    max_state = max(float(row["same_state_error_max"]) for row in checkpoints)
    max_energy = max(float(row["same_energy_error_max"]) for row in checkpoints)
    max_gradient = max(float(row["same_old_gradient_error_max"]) for row in checkpoints)
    completed = sorted({int(row["n"]) for row in checkpoints})
    text = json.dumps({
        "completed_sizes": completed,
        "instances": len(checkpoints),
        "stopped_n12": stopped_n12,
        "incremental_minus_random": gram_random.tolist(),
        "candidate_only_minus_random": pauli_random.tolist(),
        "incremental_minus_candidate_only": gram_pauli.tolist(),
        "incremental_minus_pool_mean": gram_expected.tolist(),
        "candidate_only_minus_pool_mean": pauli_expected.tolist(),
        "selector_differences": selector_diff,
        "incremental_better_instances": gram_better_instances,
        "recovery_successes": connected_recovery,
        "selected_method_instances": total_selected,
        "candidate_only_correlations_50": pauli_rhos,
        "incremental_correlations_50": gram_rhos,
        "maximum_same_state_error": max_state,
        "maximum_same_energy_error": max_energy,
        "maximum_same_old_gradient_error": max_gradient,
        "maximum_obliqueness_deg": max(all_candidate_obliqueness),
        "maximum_projector_commutator_fro": max(all_candidate_commutators),
    }, indent=2) + "\n"
    REPORT_FILE.write_text(text, encoding="utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connected-ansatz XXZ recovery pilot.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--seeds-per-size", type=int, default=SEEDS_PER_SIZE)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--max-n12-seed-seconds", type=float, default=900.0)
    parser.add_argument("--max-memory-mb", type=float, default=4096.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / ".matplotlib").mkdir(exist_ok=True)
    (HERE / ".cache").mkdir(exist_ok=True)
    started = time.time()
    checks = finite_difference_gradient_check()
    sizes = [4] if args.smoke else sorted(set(args.sizes))
    seeds_per_size = 1 if args.smoke else args.seeds_per_size
    pool_size = min(4, args.candidates) if args.smoke else min(20, args.candidates)
    all_checkpoints: list[dict[str, object]] = []
    all_candidates: list[dict[str, object]] = []
    all_methods: list[dict[str, object]] = []
    all_traces: dict[str, np.ndarray] = {}
    stopped_n12 = False
    completed_by_size: dict[int, int] = defaultdict(int)

    for n in sizes:
        if stopped_n12 and n == 12:
            break
        for seed_index in range(seeds_per_size):
            seed = BASE_SEED + 1000 * n + 37 * seed_index
            instance = f"xxz_n{n}_s{seed}"
            cache_ready = (
                (CACHE_DIR / f"{instance}.json").exists()
                and (CACHE_DIR / f"{instance}.npz").exists()
            )
            instance_started = time.time()
            if args.resume and cache_ready:
                checkpoint, candidate_rows, method_rows, traces = load_cache(instance)
                print(f"loaded {instance} from cache", flush=True)
            else:
                checkpoint, candidate_rows, method_rows, traces = run_instance(
                    n=n,
                    seed=seed,
                    pool_size=pool_size,
                    smoke=args.smoke,
                )
                save_cache(instance, checkpoint, candidate_rows, method_rows, traces)
                print(
                    f"finished {instance}: error={checkpoint['checkpoint_energy_error']:.5g}, "
                    f"gain20={checkpoint['checkpoint_gain_20']:.3e}, "
                    f"rho(P/G)={checkpoint['spearman_pauli_gain50']:.3f}/"
                    f"{checkpoint['spearman_gram_gain50']:.3f}, "
                    f"runtime={time.time()-instance_started:.1f}s",
                    flush=True,
                )
            all_checkpoints.append(checkpoint)
            all_candidates.extend(candidate_rows)
            all_methods.extend(method_rows)
            all_traces.update(traces)
            completed_by_size[n] += 1
            write_csv(CHECKPOINT_CSV, all_checkpoints)
            write_csv(CANDIDATE_CSV, all_candidates)
            write_csv(METHOD_CSV, all_methods)
            np.savez_compressed(TRACE_FILE, **all_traces)
            elapsed_seed = time.time() - instance_started
            if n == 12 and (
                elapsed_seed > args.max_n12_seed_seconds
                or peak_memory_mb() > args.max_memory_mb
            ):
                stopped_n12 = True
                print(
                    "n=12 stop condition triggered after one completed seed: "
                    f"runtime={elapsed_seed:.1f}s, peak_memory={peak_memory_mb():.1f}MB",
                    flush=True,
                )
                break

    if args.smoke:
        print("smoke test completed", flush=True)
        return
    summary_rows = aggregate_summary(all_methods)
    write_csv(SUMMARY_CSV, summary_rows)
    write_latex_table(summary_rows)
    make_figure(all_checkpoints, all_candidates, all_methods, all_traces)
    write_report(all_checkpoints, all_candidates, all_methods, stopped_n12)
    outputs = [
        CHECKPOINT_CSV,
        CANDIDATE_CSV,
        METHOD_CSV,
        SUMMARY_CSV,
        SUMMARY_TEX,
        TRACE_FILE,
        FIGURE_PDF,
        FIGURE_PNG,
        REPORT_FILE,
    ]
    manifest = {
        "experiment": "connected_brickwall_xxz_same_point_recovery_pilot",
        "task": {
            "hamiltonian": "sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1})",
            "boundary": "open",
            "delta": DELTA,
            "requested_sizes": list(args.sizes),
            "completed_per_size": dict(completed_by_size),
            "fixed_seeds_per_size": seeds_per_size,
        },
        "baseline": {
            "routing": "connected alternating even/odd nearest-neighbor brick wall",
            "routing_blocks": BASELINE_BLOCKS,
            "local_layer": "Rx-Ry-Rz on every qubit",
            "initial_state": "randomly perturbed Neel product state",
            "checkpoint_window": CHECKPOINT_WINDOW,
        },
        "intervention": {
            "insertion": "between the two brick-wall routing blocks",
            "identity_gadget": "C(B) L(phi) C(B)^dagger with phi=0",
            "candidate_pool_size": pool_size,
            "candidate_generation": "one shuffled directed traversal of every nearest-neighbor edge",
            "candidate_uses_ground_state": False,
            "parameters_added": "3n",
            "cnot_added": "2(n-1)",
            "post_steps": POST_STEPS,
            "selectors": list(METHODS),
        },
        "stopping": {
            "n12_triggered": stopped_n12,
            "max_n12_seed_seconds": args.max_n12_seed_seconds,
            "max_memory_mb": args.max_memory_mb,
        },
        "checks": checks,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "runtime_seconds": time.time() - started,
        "outputs": {
            path.name: {"sha256": hash_file(path), "bytes": path.stat().st_size}
            for path in outputs
        },
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {len(all_checkpoints)} instances in {time.time()-started:.1f}s; "
        f"n12_stopped={stopped_n12}",
        flush=True,
    )


if __name__ == "__main__":
    main()
