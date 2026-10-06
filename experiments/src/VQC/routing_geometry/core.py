from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = (X, Y, Z)


def gf2_rank(matrix: np.ndarray) -> int:
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivot = next((row for row in range(rank, rows) if a[row, col]), None)
        if pivot is None:
            continue
        a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
    return rank


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    n = matrix.shape[0]
    augmented = np.concatenate((matrix.copy(), np.eye(n, dtype=np.uint8)), axis=1)
    for col in range(n):
        pivot = next((row for row in range(col, n) if augmented[row, col]), None)
        if pivot is None:
            raise ValueError("Matrix is singular over GF(2).")
        augmented[[col, pivot]] = augmented[[pivot, col]]
        for row in range(n):
            if row != col and augmented[row, col]:
                augmented[row] ^= augmented[col]
    return augmented[:, n:]


def cnot_binary_matrix(n: int, control: int, target: int) -> np.ndarray:
    matrix = np.eye(n, dtype=np.uint8)
    matrix[target, control] ^= 1
    return matrix


def routing_matrix(n: int, gates: Iterable[tuple[int, int]]) -> np.ndarray:
    matrix = np.eye(n, dtype=np.uint8)
    for control, target in gates:
        matrix = (cnot_binary_matrix(n, control, target) @ matrix) % 2
    return matrix


def bits_from_index(index: int, n: int) -> np.ndarray:
    return np.array([(index >> (n - 1 - q)) & 1 for q in range(n)], dtype=np.uint8)


def index_from_bits(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def permutation_unitary(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    n = matrix.shape[0]
    unitary = np.zeros((2**n, 2**n), dtype=np.complex128)
    for column in range(2**n):
        output = (matrix @ bits_from_index(column, n)) % 2
        unitary[index_from_bits(output), column] = 1.0
    return unitary


def canonical_cuts(n: int) -> tuple[tuple[int, ...], ...]:
    cuts: list[tuple[int, ...]] = []
    full_mask = (1 << n) - 1
    for mask in range(1, full_mask):
        complement = full_mask ^ mask
        if mask > complement:
            continue
        cuts.append(tuple(q for q in range(n) if (mask >> q) & 1))
    return tuple(cuts)


def kappa(matrix: np.ndarray, cut: tuple[int, ...]) -> int:
    complement = tuple(q for q in range(matrix.shape[0]) if q not in cut)
    forward = matrix[np.ix_(cut, complement)]
    backward = matrix[np.ix_(complement, cut)]
    return gf2_rank(forward) + gf2_rank(backward)


def profile(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(kappa(matrix, cut) for cut in canonical_cuts(matrix.shape[0]))


def kron_all(factors: Iterable[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0]], dtype=np.complex128)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def pauli_on_qubits(n: int, assignments: dict[int, str]) -> np.ndarray:
    lookup = {"I": I2, "X": X, "Y": Y, "Z": Z}
    return kron_all(lookup[assignments.get(q, "I")] for q in range(n))


def local_pauli_generators(n: int) -> tuple[np.ndarray, ...]:
    generators = []
    for q in range(n):
        for pauli in ("X", "Y", "Z"):
            generators.append(pauli_on_qubits(n, {q: pauli}))
    return tuple(generators)


def routing_dictionary(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    circuit = permutation_unitary(matrix)
    return tuple(circuit.conj().T @ pauli @ circuit for pauli in local_pauli_generators(matrix.shape[0]))


def hs_inner(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.trace(left.conj().T @ right).real / left.shape[0])


def projection_fraction(target: np.ndarray, generators: tuple[np.ndarray, ...]) -> float:
    gram = np.array([[hs_inner(a, b) for b in generators] for a in generators])
    overlaps = np.array([hs_inner(generator, target) for generator in generators])
    numerator = float(overlaps @ np.linalg.pinv(gram, rcond=1e-12) @ overlaps)
    denominator = hs_inner(target, target)
    return numerator / denominator if denominator > 0 else 0.0


def rotation(pauli: np.ndarray, angle: float) -> np.ndarray:
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * pauli


def local_unitary(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    factors = []
    for x_angle, y_angle, z_angle in theta.reshape(-1, 3):
        factors.append(
            rotation(Z, z_angle) @ rotation(Y, y_angle) @ rotation(X, x_angle)
        )
    return kron_all(factors)


def gadget_unitary(matrix: np.ndarray, theta: np.ndarray) -> np.ndarray:
    circuit = permutation_unitary(matrix)
    return circuit.conj().T @ local_unitary(theta) @ circuit


def unitary_infidelity(target: np.ndarray, unitary: np.ndarray) -> float:
    dimension = target.shape[0]
    overlap = np.trace(target.conj().T @ unitary) / dimension
    return float(max(0.0, 1.0 - abs(overlap) ** 2))


def loss_gradient_metric(
    matrix: np.ndarray,
    target: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    unitary = gadget_unitary(matrix, theta)
    loss = unitary_infidelity(target, unitary)
    derivatives = []
    gradient = np.zeros(theta.size, dtype=float)
    flat = theta.reshape(-1)
    for index in range(theta.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[index] += eps
        minus[index] -= eps
        unitary_plus = gadget_unitary(matrix, plus.reshape(theta.shape))
        unitary_minus = gadget_unitary(matrix, minus.reshape(theta.shape))
        gradient[index] = (
            unitary_infidelity(target, unitary_plus)
            - unitary_infidelity(target, unitary_minus)
        ) / (2 * eps)
        derivatives.append((unitary_plus - unitary_minus) / (2 * eps))
    metric = np.empty((theta.size, theta.size), dtype=float)
    dimension = unitary.shape[0]
    for row, derivative_row in enumerate(derivatives):
        for col in range(row + 1):
            value = float(
                np.trace(derivatives[col].conj().T @ derivative_row).real / dimension
            )
            metric[row, col] = metric[col, row] = value
    return loss, gradient, metric


@dataclass
class TrainingTrace:
    losses: np.ndarray
    projected_scores: np.ndarray
    theta: np.ndarray


def train_gadget(
    matrix: np.ndarray,
    target: np.ndarray,
    steps: int,
    learning_rate: float = 0.5,
    tolerance: float = 1e-12,
) -> TrainingTrace:
    n = matrix.shape[0]
    theta = np.zeros((n, 3), dtype=float)
    losses: list[float] = []
    scores: list[float] = []
    for _ in range(steps):
        loss, gradient, metric = loss_gradient_metric(matrix, target, theta)
        metric_inverse = np.linalg.pinv(metric, rcond=1e-10)
        score = float(max(0.0, gradient @ metric_inverse @ gradient))
        losses.append(loss)
        scores.append(score)
        if score < tolerance:
            theta = theta.copy()
            continue
        direction = metric_inverse @ gradient
        step_size = learning_rate
        accepted = False
        while step_size >= 1e-6:
            candidate = theta.reshape(-1) - step_size * direction
            candidate = candidate.reshape(theta.shape)
            candidate_loss = unitary_infidelity(target, gadget_unitary(matrix, candidate))
            if candidate_loss <= loss - 1e-4 * step_size * score:
                theta = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            theta = theta.copy()
    final_loss, final_gradient, final_metric = loss_gradient_metric(matrix, target, theta)
    final_inverse = np.linalg.pinv(final_metric, rcond=1e-10)
    losses.append(final_loss)
    scores.append(float(max(0.0, final_gradient @ final_inverse @ final_gradient)))
    return TrainingTrace(np.array(losses), np.array(scores), theta)


def matrix_key(matrix: np.ndarray) -> bytes:
    return bytes(np.asarray(matrix, dtype=np.uint8).reshape(-1).tolist())


def gl4_cnot_distances() -> dict[bytes, int]:
    n = 4
    generators = [
        cnot_binary_matrix(n, control, target)
        for control in range(n)
        for target in range(n)
        if control != target
    ]
    identity = np.eye(n, dtype=np.uint8)
    distances = {matrix_key(identity): 0}
    queue: deque[np.ndarray] = deque([identity])
    while queue:
        matrix = queue.popleft()
        next_distance = distances[matrix_key(matrix)] + 1
        for generator in generators:
            candidate = (generator @ matrix) % 2
            key = matrix_key(candidate)
            if key not in distances:
                distances[key] = next_distance
                queue.append(candidate)
    if len(distances) != 20160:
        raise AssertionError(f"Expected |GL(4,2)|=20160, found {len(distances)}")
    return distances


def matrix_from_key(key: bytes, n: int = 4) -> np.ndarray:
    return np.frombuffer(key, dtype=np.uint8).reshape(n, n).copy()


def apply_operator_to_state(operator: np.ndarray, state: np.ndarray) -> np.ndarray:
    return operator @ state


def zero_state(n: int) -> np.ndarray:
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    return state


def bell_target(m: int) -> np.ndarray:
    n = 2 * m
    state = np.zeros(2**n, dtype=np.complex128)
    for value in range(2**m):
        bits = bits_from_index(value, m)
        state[index_from_bits(np.concatenate((bits, bits)))] = 2 ** (-m / 2)
    return state


def bell_constructor_state(m: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 2 * m
    input_state = zero_state(n)
    h_gate = (X + Z) / np.sqrt(2)
    input_layer = kron_all(h_gate if q < k else I2 for q in range(n))
    routed_input = input_layer @ input_state
    matrix = routing_matrix(n, ((q, m + q) for q in range(k)))
    circuit = permutation_unitary(matrix)
    return circuit @ routed_input, routed_input, circuit



