from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr

from VQC.routing_geometry.core import canonical_cuts, kappa, routing_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
SNAPSHOT_DIR = ROOT / "VQC" / "routing_geometry" / "outputs" / "multiblock_snapshots"
RESULTS_CSV = ROOT / "multiblock_stagnation_results.csv"
CANDIDATES_CSV = ROOT / "multiblock_stagnation_candidates.csv"
TRACE_FILE = ROOT / "VQC" / "routing_geometry" / "outputs" / "multiblock_stagnation_traces.npz"
MANIFEST_FILE = ROOT / "multiblock_stagnation_manifest.json"

BASE_SEED = 20260723
CNOTS_PER_ROUTING = 4
INTERVENTION_STEPS = 50
CHECKPOINTS = (1, 5, 20, 50)
METHODS = ("identity", "random", "osr_only", "pauli_only", "projection", "oracle")

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
PAULI = {"X": X, "Y": Y, "Z": Z}


@dataclass(frozen=True)
class Gate:
    kind: str
    q0: int
    q1: int = -1
    param: int = -1
    angle: float = 0.0
    matrix: np.ndarray | None = None


@dataclass
class AdamState:
    theta: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    step: int
    losses: np.ndarray


@dataclass
class Geometry:
    loss: float
    fidelity: float
    ambient_norm: float
    projected_norm: float
    projected_squared: float
    chi: float
    gram: np.ndarray
    tangent_real: np.ndarray
    ambient_real: np.ndarray


@dataclass
class Candidate:
    index: int
    name: str
    sequence: tuple[tuple[int, int], ...]
    matrix: np.ndarray
    key_kappa: int
    osr_score: float = 0.0
    pauli_score: float = 0.0
    full_score: float = 0.0
    delta: float = 0.0
    initial_loss_error: float = 0.0
    initial_state_error: float = 0.0
    initial_gradient_error: float = 0.0
    initial_ambient_error: float = 0.0
    losses: np.ndarray | None = None


def rotation(axis: str, angle: float) -> np.ndarray:
    pauli = PAULI[axis]
    return np.cos(angle / 2.0) * I2 - 1j * np.sin(angle / 2.0) * pauli


def _axis(gate: Gate) -> str:
    if not gate.kind.startswith("R"):
        raise ValueError(f"Not a rotation gate: {gate.kind}")
    return gate.kind[1:]


def one_qubit_matrix(gate: Gate, theta: np.ndarray) -> np.ndarray:
    if gate.kind == "fixed":
        if gate.matrix is None:
            raise ValueError("Fixed gate has no matrix.")
        return gate.matrix
    if gate.kind.startswith("R"):
        angle = theta[gate.param] if gate.param >= 0 else gate.angle
        return rotation(_axis(gate), angle)
    raise ValueError(f"Unsupported one-qubit gate: {gate.kind}")


def apply_one_qubit(values: np.ndarray, matrix: np.ndarray, qubit: int, n: int) -> np.ndarray:
    batched = values.ndim == 2
    batch = values.shape[1] if batched else 1
    tensor = values.reshape((2,) * n + ((batch,) if batched else ()))
    transformed = np.tensordot(matrix, tensor, axes=(1, qubit))
    transformed = np.moveaxis(transformed, 0, qubit)
    return transformed.reshape(values.shape)


def cnot_permutation(n: int, control: int, target: int) -> np.ndarray:
    indices = np.arange(2**n, dtype=np.int64)
    control_mask = 1 << (n - 1 - control)
    target_mask = 1 << (n - 1 - target)
    active = (indices & control_mask) != 0
    return indices ^ (active.astype(np.int64) * target_mask)


def apply_gate(values: np.ndarray, gate: Gate, theta: np.ndarray, n: int, dagger: bool = False) -> np.ndarray:
    if gate.kind == "CNOT":
        return values[cnot_permutation(n, gate.q0, gate.q1)]
    matrix = one_qubit_matrix(gate, theta)
    if dagger:
        matrix = matrix.conj().T
    return apply_one_qubit(values, matrix, gate.q0, n)


def zero_state(n: int) -> np.ndarray:
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    return state


def simulate(ops: list[Gate], theta: np.ndarray, n: int) -> np.ndarray:
    state = zero_state(n)
    for gate in ops:
        state = apply_gate(state, gate, theta, n)
    return state


def simulate_with_tangents(
    ops: list[Gate],
    theta: np.ndarray,
    n: int,
    parameter_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = zero_state(n)
    tangents = np.zeros((2**n, parameter_count), dtype=np.complex128)
    for gate in ops:
        before = state
        state = apply_gate(state, gate, theta, n)
        if parameter_count:
            tangents = apply_gate(tangents, gate, theta, n)
        if gate.param >= 0:
            derivative_matrix = (-0.5j * PAULI[_axis(gate)]) @ one_qubit_matrix(gate, theta)
            tangents[:, gate.param] += apply_one_qubit(before, derivative_matrix, gate.q0, n)
    return state, tangents


def loss_and_gradient(
    ops: list[Gate],
    theta: np.ndarray,
    target: np.ndarray,
    n: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    forward = [zero_state(n)]
    for gate in ops:
        forward.append(apply_gate(forward[-1], gate, theta, n))
    state = forward[-1]
    overlap = np.vdot(target, state)
    fidelity = float(abs(overlap) ** 2)
    loss = float(max(0.0, 1.0 - fidelity))
    gradient = np.zeros(theta.size, dtype=float)
    co_state = target.copy()
    for index in range(len(ops) - 1, -1, -1):
        gate = ops[index]
        if gate.param >= 0:
            derivative_matrix = (-0.5j * PAULI[_axis(gate)]) @ one_qubit_matrix(gate, theta)
            derivative_after_gate = apply_one_qubit(
                forward[index], derivative_matrix, gate.q0, n
            )
            derivative_overlap = np.vdot(co_state, derivative_after_gate)
            gradient[gate.param] += -2.0 * float(
                np.real(np.conj(overlap) * derivative_overlap)
            )
        co_state = apply_gate(co_state, gate, theta, n, dagger=True)
    return loss, gradient, state


def horizontalize(state: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    if tangents.size == 0:
        return tangents
    phases = state.conj() @ tangents
    return tangents - state[:, None] * phases[None, :]


def realify(vectors: np.ndarray) -> np.ndarray:
    return np.concatenate((vectors.real, vectors.imag), axis=0)


def projection_norm(tangent_real: np.ndarray, ambient_real: np.ndarray) -> float:
    if tangent_real.size == 0:
        return 0.0
    left_vectors, singular_values, _ = np.linalg.svd(
        tangent_real, full_matrices=False
    )
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0.0
    rank = int(np.sum(singular_values > 1e-10 * singular_values[0]))
    if rank == 0:
        return 0.0
    basis = left_vectors[:, :rank]
    return float(np.linalg.norm(basis.T @ ambient_real))


def geometry(
    ops: list[Gate],
    theta: np.ndarray,
    target: np.ndarray,
    n: int,
) -> Geometry:
    state, tangents = simulate_with_tangents(ops, theta, n, theta.size)
    overlap = np.vdot(target, state)
    fidelity = float(abs(overlap) ** 2)
    loss = float(max(0.0, 1.0 - fidelity))
    ambient = -2.0 * overlap * target + 2.0 * fidelity * state
    tangent_horizontal = horizontalize(state, tangents)
    tangent_real = realify(tangent_horizontal)
    ambient_real = realify(ambient[:, None]).reshape(-1)
    projected = projection_norm(tangent_real, ambient_real)
    ambient_norm = float(np.linalg.norm(ambient_real))
    gram = tangent_real.T @ tangent_real
    return Geometry(
        loss=loss,
        fidelity=fidelity,
        ambient_norm=ambient_norm,
        projected_norm=projected,
        projected_squared=projected**2,
        chi=projected / ambient_norm if ambient_norm > 1e-15 else 0.0,
        gram=gram,
        tangent_real=tangent_real,
        ambient_real=ambient_real,
    )


def add_local_layer(ops: list[Gate], n: int, parameter_offset: int) -> int:
    parameter = parameter_offset
    for qubit in range(n):
        for axis in ("X", "Y", "Z"):
            ops.append(Gate(f"R{axis}", qubit, param=parameter))
            parameter += 1
    return parameter


def add_fixed_local_layer(
    ops: list[Gate],
    n: int,
    angles: np.ndarray,
) -> None:
    for qubit in range(n):
        for axis_index, axis in enumerate(("X", "Y", "Z")):
            ops.append(Gate(f"R{axis}", qubit, angle=float(angles[qubit, axis_index])))


def add_cnot_sequence(ops: list[Gate], sequence: Iterable[tuple[int, int]]) -> None:
    for control, target in sequence:
        ops.append(Gate("CNOT", control, target))


def shifted_ops(ops: list[Gate], offset: int) -> list[Gate]:
    return [replace(gate, param=gate.param + offset) if gate.param >= 0 else gate for gate in ops]


def random_side_sequence(
    n: int,
    rng: np.random.Generator,
    length: int,
) -> tuple[tuple[int, int], ...]:
    half = n // 2
    sequence = []
    for index in range(length):
        side = index % 2
        qubits = np.arange(side * half, (side + 1) * half)
        control, target = rng.choice(qubits, size=2, replace=False)
        sequence.append((int(control), int(target)))
    return tuple(sequence)


def build_instance(
    n: int,
    blocks: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[Gate], np.ndarray, list[tuple[tuple[int, int], ...]], np.ndarray]:
    half = n // 2
    suffix_angles = rng.uniform(-0.73, 0.73, size=(blocks, n, 3))
    suffix_angles += np.sign(suffix_angles + 1e-12) * 0.071
    routings: list[tuple[tuple[int, int], ...]] = [
        tuple((qubit, half + qubit) for qubit in range(half))
    ]
    for _ in range(1, blocks):
        routings.append(random_side_sequence(n, rng, max(2, n // 2)))

    target_ops: list[Gate] = []
    for qubit in range(half):
        target_ops.append(Gate("fixed", qubit, matrix=H))
    add_cnot_sequence(target_ops, routings[0])
    add_fixed_local_layer(target_ops, n, suffix_angles[0])
    for layer in range(1, blocks):
        add_cnot_sequence(target_ops, routings[layer])
        add_fixed_local_layer(target_ops, n, suffix_angles[layer])
    target = simulate(target_ops, np.empty(0), n)

    old_ops: list[Gate] = []
    parameter = add_local_layer(old_ops, n, 0)
    parameter = add_local_layer(old_ops, n, parameter)
    for layer in range(1, blocks):
        add_cnot_sequence(old_ops, routings[layer])
        parameter = add_local_layer(old_ops, n, parameter)

    known_theta = np.zeros(parameter, dtype=float)
    for layer in range(blocks):
        start = (layer + 1) * 3 * n
        known_theta[start : start + 3 * n] = suffix_angles[layer].reshape(-1)
    return target, old_ops, known_theta, routings, suffix_angles


def adam_train(
    ops: list[Gate],
    initial_theta: np.ndarray,
    target: np.ndarray,
    n: int,
    steps: int,
    learning_rate: float,
    cosine_decay: bool = False,
) -> AdamState:
    theta = initial_theta.copy()
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    losses = []
    for step in range(1, steps + 1):
        loss, gradient, _ = loss_and_gradient(ops, theta, target, n)
        losses.append(loss)
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        rate = learning_rate
        if cosine_decay:
            rate *= 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * (step - 1) / steps))
        theta -= rate * first_hat / (np.sqrt(second_hat) + 1e-8)
    final_loss, _, _ = loss_and_gradient(ops, theta, target, n)
    losses.append(final_loss)
    return AdamState(theta, first, second, steps, np.asarray(losses))


def build_gadget_ops(
    sequence: tuple[tuple[int, int], ...],
    old_ops: list[Gate],
    n: int,
) -> list[Gate]:
    ops: list[Gate] = []
    add_cnot_sequence(ops, sequence)
    add_local_layer(ops, n, 0)
    add_cnot_sequence(ops, reversed(sequence))
    ops.extend(shifted_ops(old_ops, 3 * n))
    return ops


def identity_sequence(n: int) -> tuple[tuple[int, int], ...]:
    edge = (0, 1 if n > 1 else 0)
    return (edge, edge, edge, edge)


def random_candidate_pool(
    n: int,
    required_rank: int,
    pool_size: int,
    rng: np.random.Generator,
) -> list[Candidate]:
    key_cut = tuple(range(n // 2))
    candidates: list[Candidate] = []
    identity = identity_sequence(n)
    candidates.append(
        Candidate(
            index=0,
            name="identity",
            sequence=identity,
            matrix=routing_matrix(n, identity),
            key_kappa=0,
        )
    )
    seen = {np.eye(n, dtype=np.uint8).tobytes()}
    attempts = 0
    while len(candidates) < pool_size + 1 and attempts < 200000:
        attempts += 1
        sequence = []
        for _ in range(CNOTS_PER_ROUTING):
            control, target = rng.choice(n, size=2, replace=False)
            sequence.append((int(control), int(target)))
        sequence_tuple = tuple(sequence)
        matrix = routing_matrix(n, sequence_tuple)
        key = matrix.tobytes()
        key_rank = kappa(matrix, key_cut)
        if key in seen or key_rank >= required_rank:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                index=len(candidates),
                name=f"B{len(candidates):02d}",
                sequence=sequence_tuple,
                matrix=matrix,
                key_kappa=key_rank,
            )
        )
    if len(candidates) != pool_size + 1:
        raise RuntimeError(f"Could not construct {pool_size} fixed-cost candidates.")
    return candidates


def state_log_schmidt_rank(
    state: np.ndarray,
    cut: tuple[int, ...],
    n: int,
    tolerance: float = 1e-10,
) -> int:
    complement = tuple(qubit for qubit in range(n) if qubit not in cut)
    tensor = state.reshape((2,) * n)
    matrix = np.transpose(tensor, cut + complement).reshape(2 ** len(cut), -1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    rank = int(np.sum(singular_values > tolerance * singular_values[0]))
    return int(round(np.log2(max(rank, 1))))


def task_cut_demands(target: np.ndarray, n: int, maximum_cuts: int = 10) -> list[tuple[tuple[int, ...], int]]:
    scored = []
    key_cut = tuple(range(n // 2))
    for cut in canonical_cuts(n):
        demand = state_log_schmidt_rank(target, cut, n)
        balance = min(len(cut), n - len(cut))
        key_bonus = n if set(cut) == set(key_cut) else 0
        scored.append((demand, balance, key_bonus, cut))
    scored.sort(reverse=True)
    selected = [(cut, demand) for demand, _, _, cut in scored[:maximum_cuts]]
    if key_cut not in [cut for cut, _ in selected]:
        selected[-1] = (key_cut, state_log_schmidt_rank(target, key_cut, n))
    return selected


def score_candidates(
    candidates: list[Candidate],
    old_ops: list[Gate],
    old_theta: np.ndarray,
    old_geometry: Geometry,
    target: np.ndarray,
    n: int,
) -> tuple[list[Candidate], dict[int, tuple[list[Gate], np.ndarray, Geometry]]]:
    demands = task_cut_demands(target, n)
    cache: dict[int, tuple[list[Gate], np.ndarray, Geometry]] = {}
    old_count = old_theta.size
    for candidate in candidates:
        candidate.osr_score = float(
            sum(
                min(kappa(candidate.matrix, cut), demand) * (1.0 + 0.1 * demand)
                for cut, demand in demands
            )
        )
        ops = build_gadget_ops(candidate.sequence, old_ops, n)
        theta = np.concatenate((np.zeros(3 * n), old_theta))
        candidate_geometry = geometry(ops, theta, target, n)
        state = simulate(ops, theta, n)
        old_state = simulate(old_ops, old_theta, n)
        _, full_gradient, _ = loss_and_gradient(ops, theta, target, n)
        _, old_gradient, _ = loss_and_gradient(old_ops, old_theta, target, n)
        candidate.initial_state_error = float(np.linalg.norm(state - old_state))
        candidate.initial_loss_error = abs(candidate_geometry.loss - old_geometry.loss)
        candidate.initial_gradient_error = float(
            np.linalg.norm(full_gradient[3 * n :] - old_gradient)
        )
        candidate.initial_ambient_error = float(
            np.linalg.norm(candidate_geometry.ambient_real - old_geometry.ambient_real)
        )
        new_tangent = candidate_geometry.tangent_real[:, : 3 * n]
        candidate.pauli_score = projection_norm(
            new_tangent, candidate_geometry.ambient_real
        ) ** 2
        candidate.full_score = candidate_geometry.projected_squared
        candidate.delta = max(
            0.0, candidate.full_score - old_geometry.projected_squared
        )
        if (
            candidate.initial_state_error > 1e-11
            or candidate.initial_loss_error > 1e-12
            or candidate.initial_gradient_error > 1e-10
            or candidate.initial_ambient_error > 1e-11
            or candidate_geometry.gram.shape != (3 * n + old_count, 3 * n + old_count)
        ):
            raise AssertionError(
                "Identity initialization or equal-parameter check failed for "
                f"{candidate.name}."
            )
        cache[candidate.index] = (ops, theta, candidate_geometry)
    return candidates, cache


def finite_step_train_candidates(
    candidates: list[Candidate],
    cache: dict[int, tuple[list[Gate], np.ndarray, Geometry]],
    target: np.ndarray,
    n: int,
) -> None:
    for candidate in candidates:
        ops, theta, _ = cache[candidate.index]
        trained = adam_train(
            ops,
            theta,
            target,
            n,
            INTERVENTION_STEPS,
            learning_rate=0.035,
            cosine_decay=False,
        )
        candidate.losses = trained.losses


def selector_indices(candidates: list[Candidate], rng: np.random.Generator) -> dict[str, int]:
    nonidentity = candidates[1:]
    improvements = {
        candidate.index: float(candidate.losses[0] - candidate.losses[INTERVENTION_STEPS])
        for candidate in candidates
        if candidate.losses is not None
    }
    return {
        "identity": 0,
        "random": int(rng.choice([candidate.index for candidate in nonidentity])),
        "osr_only": max(nonidentity, key=lambda candidate: (candidate.osr_score, -candidate.index)).index,
        "pauli_only": max(nonidentity, key=lambda candidate: (candidate.pauli_score, -candidate.index)).index,
        "projection": max(nonidentity, key=lambda candidate: (candidate.delta, -candidate.index)).index,
        "oracle": max(candidates, key=lambda candidate: (improvements[candidate.index], -candidate.index)).index,
    }


def first_threshold_step(losses: np.ndarray, threshold_improvement: float) -> int:
    initial = float(losses[0])
    for step in range(1, len(losses)):
        if initial - float(losses[step]) >= threshold_improvement:
            return step
    return -1


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int = 4000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_instance(
    n: int,
    blocks: int,
    seed: int,
    pool_size: int,
    smoke: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    target, old_ops, known_theta, target_routings, suffix_angles = build_instance(
        n, blocks, rng
    )
    initial_theta = known_theta + rng.normal(0.0, 0.035, size=known_theta.size)
    old_steps = 80 if smoke else 450
    old_training = adam_train(
        old_ops,
        initial_theta,
        target,
        n,
        old_steps,
        learning_rate=0.025,
        cosine_decay=True,
    )
    known_loss, _, _ = loss_and_gradient(old_ops, known_theta, target, n)
    trained_loss = float(old_training.losses[-1])
    if trained_loss > known_loss + 2e-8:
        refinement = adam_train(
            old_ops,
            known_theta + rng.normal(0.0, 0.002, size=known_theta.size),
            target,
            n,
            200 if not smoke else 60,
            learning_rate=0.01,
            cosine_decay=True,
        )
        if refinement.losses[-1] < old_training.losses[-1]:
            old_training = refinement
    old_theta = old_training.theta
    old_geometry = geometry(old_ops, old_theta, target, n)
    old_gradient = loss_and_gradient(old_ops, old_theta, target, n)[1]
    key_cut = tuple(range(n // 2))
    target_rank = state_log_schmidt_rank(target, key_cut, n)
    if target_rank != n // 2:
        raise AssertionError("Target is not maximally entangled across the key cut.")
    expected_floor = 1.0 - 2.0 ** (-n // 2)
    if abs(old_geometry.loss - expected_floor) > (2e-5 if smoke else 2e-7):
        raise AssertionError(
            f"Old optimizer missed the controlled floor: {old_geometry.loss} vs {expected_floor}."
        )
    if old_geometry.ambient_norm < 1e-3 or old_geometry.chi > (5e-3 if smoke else 2e-4):
        raise AssertionError(
            f"Not a true constrained stagnation: ambient={old_geometry.ambient_norm}, "
            f"chi={old_geometry.chi}."
        )
    old_state = simulate(old_ops, old_theta, n)
    if np.linalg.norm(old_state - zero_state(n)) < 1e-2:
        raise AssertionError("The converged old physical state is too close to identity output.")

    candidates = random_candidate_pool(n, target_rank, pool_size, rng)
    candidates, cache = score_candidates(
        candidates, old_ops, old_theta, old_geometry, target, n
    )
    finite_step_train_candidates(candidates, cache, target, n)
    selected = selector_indices(candidates, rng)
    by_index = {candidate.index: candidate for candidate in candidates}
    oracle_candidate = by_index[selected["oracle"]]
    oracle_improvement = float(
        oracle_candidate.losses[0] - oracle_candidate.losses[INTERVENTION_STEPS]
    )
    threshold_improvement = 0.8 * oracle_improvement

    instance_id = f"n{n}_t{blocks}_s{seed}"
    method_rows = []
    for method in METHODS:
        candidate = by_index[selected[method]]
        losses = candidate.losses
        assert losses is not None
        row: dict[str, object] = {
            "instance": instance_id,
            "seed": seed,
            "n": n,
            "t": blocks,
            "method": method,
            "candidate_index": candidate.index,
            "candidate_name": candidate.name,
            "sequence": json.dumps(candidate.sequence, separators=(",", ":")),
            "parameters_added": 3 * n,
            "routing_cnot_count": CNOTS_PER_ROUTING,
            "gadget_cnot_count": 2 * CNOTS_PER_ROUTING,
            "candidate_pool_size": pool_size,
            "old_loss": old_geometry.loss,
            "old_ambient_norm": old_geometry.ambient_norm,
            "old_projected_norm": old_geometry.projected_norm,
            "old_chi": old_geometry.chi,
            "key_kappa": candidate.key_kappa,
            "osr_score": candidate.osr_score,
            "pauli_score": candidate.pauli_score,
            "full_projected_score": candidate.full_score,
            "delta_B": candidate.delta,
            "final_loss": float(losses[-1]),
            "success_90pct_oracle": int(
                float(losses[0] - losses[-1]) >= 0.9 * oracle_improvement - 1e-12
            ),
            "steps_to_80pct_oracle": first_threshold_step(
                losses, threshold_improvement
            ),
            "improvement_per_cnot_50": float(losses[0] - losses[-1])
            / (2 * CNOTS_PER_ROUTING),
        }
        for checkpoint in CHECKPOINTS:
            row[f"loss_after_{checkpoint}"] = float(losses[checkpoint])
            row[f"improvement_after_{checkpoint}"] = float(
                losses[0] - losses[checkpoint]
            )
        method_rows.append(row)

    candidate_rows = []
    selected_reverse: dict[int, list[str]] = {}
    for method, index in selected.items():
        selected_reverse.setdefault(index, []).append(method)
    for candidate in candidates:
        losses = candidate.losses
        assert losses is not None
        candidate_rows.append(
            {
                "instance": instance_id,
                "seed": seed,
                "n": n,
                "t": blocks,
                "candidate_index": candidate.index,
                "candidate_name": candidate.name,
                "selected_by": ";".join(selected_reverse.get(candidate.index, [])),
                "sequence": json.dumps(candidate.sequence, separators=(",", ":")),
                "matrix": json.dumps(candidate.matrix.tolist(), separators=(",", ":")),
                "key_kappa": candidate.key_kappa,
                "osr_score": candidate.osr_score,
                "pauli_score": candidate.pauli_score,
                "full_projected_score": candidate.full_score,
                "delta_B": candidate.delta,
                "initial_state_error": candidate.initial_state_error,
                "initial_loss_error": candidate.initial_loss_error,
                "initial_gradient_error": candidate.initial_gradient_error,
                "initial_ambient_error": candidate.initial_ambient_error,
                "loss_after_1": float(losses[1]),
                "loss_after_5": float(losses[5]),
                "loss_after_20": float(losses[20]),
                "loss_after_50": float(losses[50]),
                "improvement_after_1": float(losses[0] - losses[1]),
                "improvement_after_5": float(losses[0] - losses[5]),
                "improvement_after_20": float(losses[0] - losses[20]),
                "improvement_after_50": float(losses[0] - losses[50]),
            }
        )

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SNAPSHOT_DIR / f"{instance_id}.npz",
        target=target,
        old_state=old_state,
        old_theta=old_theta,
        old_gradient=old_gradient,
        old_gram=old_geometry.gram,
        old_tangent_real=old_geometry.tangent_real,
        old_ambient_real=old_geometry.ambient_real,
        optimizer_first_moment=old_training.first_moment,
        optimizer_second_moment=old_training.second_moment,
        optimizer_step=np.array(old_training.step),
        old_training_losses=old_training.losses,
        target_routings=np.array(
            [json.dumps(sequence) for sequence in target_routings], dtype=str
        ),
        suffix_angles=suffix_angles,
    )
    trace_data = {
        f"{instance_id}__{candidate.name}": candidate.losses
        for candidate in candidates
        if candidate.losses is not None
    }
    summary = {
        "instance": instance_id,
        "seed": seed,
        "n": n,
        "t": blocks,
        "target_log2_schmidt_rank": target_rank,
        "expected_loss_floor": expected_floor,
        "old_loss": old_geometry.loss,
        "old_ambient_norm": old_geometry.ambient_norm,
        "old_projected_norm": old_geometry.projected_norm,
        "old_chi": old_geometry.chi,
        "old_gradient_norm": float(np.linalg.norm(old_gradient)),
        "old_state_distance_from_zero": float(np.linalg.norm(old_state - zero_state(n))),
        "oracle_improvement_50": oracle_improvement,
        "selectors": selected,
        "maximum_same_point_state_error": max(
            candidate.initial_state_error for candidate in candidates
        ),
        "maximum_same_point_loss_error": max(
            candidate.initial_loss_error for candidate in candidates
        ),
        "maximum_same_point_old_gradient_error": max(
            candidate.initial_gradient_error for candidate in candidates
        ),
        "maximum_same_point_ambient_error": max(
            candidate.initial_ambient_error for candidate in candidates
        ),
    }
    return method_rows, candidate_rows, summary, trace_data


def correlation_summary(candidate_rows: list[dict[str, object]]) -> dict[str, object]:
    nonidentity = [row for row in candidate_rows if int(row["candidate_index"]) != 0]
    result: dict[str, object] = {}
    for checkpoint in CHECKPOINTS:
        improvement = np.array(
            [float(row[f"improvement_after_{checkpoint}"]) for row in nonidentity]
        )
        for score_name in ("delta_B", "pauli_score", "osr_score"):
            score = np.array([float(row[score_name]) for row in nonidentity])
            if np.ptp(score) < 1e-15 or np.ptp(improvement) < 1e-15:
                result[f"{score_name}_pearson_{checkpoint}"] = [float("nan"), float("nan")]
                result[f"{score_name}_spearman_{checkpoint}"] = [float("nan"), float("nan")]
                continue
            pearson = pearsonr(score, improvement)
            spearman = spearmanr(score, improvement)
            result[f"{score_name}_pearson_{checkpoint}"] = [
                float(pearson.statistic),
                float(pearson.pvalue),
            ]
            result[f"{score_name}_spearman_{checkpoint}"] = [
                float(spearman.statistic),
                float(spearman.pvalue),
            ]
    return result


def aggregate_methods(
    method_rows: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    aggregate = {}
    for method in METHODS:
        rows = [row for row in method_rows if row["method"] == method]
        improvements = np.array(
            [float(row["improvement_after_50"]) for row in rows]
        )
        success = np.array([float(row["success_90pct_oracle"]) for row in rows])
        improvement_ci = bootstrap_ci(improvements, rng)
        success_ci = bootstrap_ci(success, rng)
        aggregate[method] = {
            "instances": len(rows),
            "mean_improvement_50": float(np.mean(improvements)),
            "bootstrap95_improvement_50": list(improvement_ci),
            "success_rate_90pct_oracle": float(np.mean(success)),
            "bootstrap95_success_rate": list(success_ci),
            "median_steps_to_80pct_oracle": float(
                np.median(
                    [
                        int(row["steps_to_80pct_oracle"])
                        for row in rows
                        if int(row["steps_to_80pct_oracle"]) >= 0
                    ]
                )
            )
            if any(int(row["steps_to_80pct_oracle"]) >= 0 for row in rows)
            else -1.0,
        }
    return aggregate


def make_figure(
    method_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    traces: dict[str, np.ndarray],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    colors = {
        "identity": "#7f7f7f",
        "random": "#d98c10",
        "osr_only": "#2a9d8f",
        "pauli_only": "#5b8ff9",
        "projection": "#c23b73",
        "oracle": "#222222",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)

    representative = sorted({str(row["instance"]) for row in method_rows})[0]
    representative_rows = [
        row for row in method_rows if str(row["instance"]) == representative
    ]
    old_projected = float(representative_rows[0]["old_projected_norm"])
    jump_methods = ("identity", "random", "osr_only", "pauli_only", "projection")
    jump_rows = {
        str(row["method"]): row
        for row in representative_rows
        if str(row["method"]) in jump_methods
    }
    jump_values = [old_projected] + [
        np.sqrt(float(jump_rows[method]["full_projected_score"]))
        for method in jump_methods
    ]
    jump_labels = ["old"] + ["I", "random", "OSR", "Pauli", r"$\Delta_B$"]
    jump_colors = ["#222222"] + [colors[method] for method in jump_methods]
    display_floor = 1e-13
    axes[0, 0].bar(
        np.arange(len(jump_values)),
        np.maximum(np.asarray(jump_values), display_floor) - display_floor,
        bottom=display_floor,
        color=jump_colors,
        width=0.68,
        alpha=0.82,
    )
    axes[0, 0].set_xticks(np.arange(len(jump_values)))
    axes[0, 0].set_xticklabels(jump_labels, rotation=20, ha="right")
    axes[0, 0].set_title("(a) Instantaneous tangent recovery")
    axes[0, 0].set_ylabel(r"$\|\Pi_{\mathcal{T}}G\|$")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(display_floor, 1.0)
    axes[0, 0].axhline(
        max(old_projected, display_floor),
        color="#222222",
        linewidth=0.8,
        linestyle=":",
    )

    for row in representative_rows:
        method = str(row["method"])
        candidate_name = str(row["candidate_name"])
        key = f"{representative}__{candidate_name}"
        axes[0, 1].plot(
            np.arange(INTERVENTION_STEPS + 1),
            traces[key],
            label=method.replace("_", " "),
            color=colors[method],
            linewidth=1.8 if method in ("projection", "oracle") else 1.25,
            linestyle="--" if method == "oracle" else "-",
        )
    axes[0, 1].set_title("(b) Same-point intervention")
    axes[0, 1].set_xlabel("Optimization step")
    axes[0, 1].set_ylabel("Infidelity")
    axes[0, 1].legend(frameon=False, ncol=2)

    nonidentity = [row for row in candidate_rows if int(row["candidate_index"]) != 0]
    x_values = np.array([float(row["delta_B"]) for row in nonidentity])
    y_values = np.array([float(row["improvement_after_20"]) for row in nonidentity])
    sizes = np.array([int(row["n"]) for row in nonidentity])
    for n in sorted(set(sizes)):
        mask = sizes == n
        axes[1, 0].scatter(
            x_values[mask],
            y_values[mask],
            s=18,
            alpha=0.68,
            label=f"n={n}",
        )
    axes[1, 0].set_title(r"(c) Tangent score predicts descent")
    axes[1, 0].set_xlabel(r"$\Delta_B$ at the fixed physical point")
    axes[1, 0].set_ylabel("Loss decrease after 20 steps")
    axes[1, 0].legend(frameon=False)
    if np.ptp(x_values) > 0 and np.ptp(y_values) > 0:
        rho = spearmanr(x_values, y_values).statistic
        axes[1, 0].text(
            0.04,
            0.94,
            rf"Spearman $\rho={rho:.3f}$",
            transform=axes[1, 0].transAxes,
            va="top",
        )

    ns = sorted({int(row["n"]) for row in method_rows})
    width = 0.12
    for method_index, method in enumerate(METHODS):
        rates = []
        for n in ns:
            values = [
                float(row["success_90pct_oracle"])
                for row in method_rows
                if row["method"] == method and int(row["n"]) == n
            ]
            rates.append(float(np.mean(values)))
        offset = (method_index - (len(METHODS) - 1) / 2) * width
        axes[1, 1].bar(
            np.arange(len(ns)) + offset,
            rates,
            width=width,
            color=colors[method],
            label=method.replace("_", " "),
        )
    axes[1, 1].set_xticks(np.arange(len(ns)))
    axes[1, 1].set_xticklabels([f"n={n}" for n in ns])
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("(d) Near-oracle selection rate")
    axes[1, 1].set_ylabel(r"Fraction reaching $\geq90\%$ oracle gain")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "fig_multiblock_stagnation.pdf")
    fig.savefig(FIGURE_DIR / "fig_multiblock_stagnation.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled multiblock same-point routing intervention."
    )
    parser.add_argument("--smoke", action="store_true", help="Run one n=4 diagnostic instance.")
    parser.add_argument("--seeds", type=int, default=4, help="Seeds per (n,t) configuration.")
    parser.add_argument("--pool-size", type=int, default=12, help="Nonidentity candidates per instance.")
    args = parser.parse_args()

    start_time = time.time()
    configurations = ((4, 2),) if args.smoke else ((4, 2), (6, 3), (8, 4))
    seeds_per_configuration = 1 if args.smoke else args.seeds
    pool_size = min(args.pool_size, 5) if args.smoke else args.pool_size
    all_method_rows: list[dict[str, object]] = []
    all_candidate_rows: list[dict[str, object]] = []
    instance_summaries = []
    all_traces: dict[str, np.ndarray] = {}

    for n, blocks in configurations:
        for seed_index in range(seeds_per_configuration):
            seed = BASE_SEED + 1000 * n + 100 * blocks + seed_index
            print(f"running n={n}, t={blocks}, seed={seed}, pool={pool_size}", flush=True)
            method_rows, candidate_rows, summary, traces = run_instance(
                n, blocks, seed, pool_size, args.smoke
            )
            all_method_rows.extend(method_rows)
            all_candidate_rows.extend(candidate_rows)
            instance_summaries.append(summary)
            all_traces.update(traces)
            print(
                "  old loss={:.9f}, ambient={:.4e}, chi={:.4e}, "
                "oracle gain={:.4e}".format(
                    summary["old_loss"],
                    summary["old_ambient_norm"],
                    summary["old_chi"],
                    summary["oracle_improvement_50"],
                ),
                flush=True,
            )

    output_prefix = "smoke_" if args.smoke else ""
    results_path = ROOT / f"{output_prefix}{RESULTS_CSV.name}"
    candidates_path = ROOT / f"{output_prefix}{CANDIDATES_CSV.name}"
    traces_path = TRACE_FILE.with_name(f"{output_prefix}{TRACE_FILE.name}")
    manifest_path = ROOT / f"{output_prefix}{MANIFEST_FILE.name}"
    write_csv(results_path, all_method_rows)
    write_csv(candidates_path, all_candidate_rows)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(traces_path, **all_traces)
    if not args.smoke:
        make_figure(all_method_rows, all_candidate_rows, all_traces)

    correlation = correlation_summary(all_candidate_rows)
    aggregate = aggregate_methods(all_method_rows, BASE_SEED + 77)
    manifest = {
        "experiment": "multiblock same-point routing intervention",
        "status": "smoke" if args.smoke else "full",
        "base_seed": BASE_SEED,
        "configurations": [list(configuration) for configuration in configurations],
        "seeds_per_configuration": seeds_per_configuration,
        "nonidentity_candidates_per_instance": pool_size,
        "routing_cnot_count": CNOTS_PER_ROUTING,
        "gadget_cnot_count": 2 * CNOTS_PER_ROUTING,
        "intervention_steps": INTERVENTION_STEPS,
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.035,
            "beta1": 0.9,
            "beta2": 0.999,
            "backtracking": False,
        },
        "success_definition": "50-step loss decrease >= 90% of the per-instance oracle decrease",
        "threshold_definition": "first step reaching 80% of the per-instance oracle 50-step decrease",
        "candidate_constraint": "four physical CNOTs and key-cut kappa strictly below target log2 Schmidt rank",
        "method_scope": {
            "identity": "logically identity routing padded to the same physical CNOT count",
            "random": "uniform draw from the same fixed candidate pool",
            "osr_only": "maximizes task-cut OSR demand coverage",
            "pauli_only": "maximizes candidate-new-direction projection without old-space redundancy correction",
            "projection": "maximizes the increase in full Gram-corrected projected gradient",
            "oracle": "best observed 50-step candidate; retrospective upper bound",
        },
        "instances": instance_summaries,
        "aggregate": aggregate,
        "correlations": correlation,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "elapsed_seconds": time.time() - start_time,
        "files": {
            "results_csv": str(results_path.relative_to(ROOT)),
            "candidates_csv": str(candidates_path.relative_to(ROOT)),
            "traces_npz": str(traces_path.relative_to(ROOT)),
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            **(
                {
                    "figure_pdf": "figures/fig_multiblock_stagnation.pdf",
                    "figure_png": "figures/fig_multiblock_stagnation.png",
                }
                if not args.smoke
                else {}
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    hash_paths = [results_path, candidates_path, traces_path, Path(__file__).resolve()]
    if not args.smoke:
        hash_paths.extend(
            (
                FIGURE_DIR / "fig_multiblock_stagnation.pdf",
                FIGURE_DIR / "fig_multiblock_stagnation.png",
            )
        )
    manifest["sha256"] = {
        str(path.relative_to(ROOT)): hash_file(path) for path in hash_paths
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {results_path.name}, {candidates_path.name}, {manifest_path.name}")


if __name__ == "__main__":
    main()
