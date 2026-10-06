from __future__ import annotations

import unittest

import numpy as np

from VQC.routing_geometry.core import (
    I2,
    X,
    cnot_binary_matrix,
    kappa,
    kron_all,
    local_unitary,
    permutation_unitary,
    rotation,
)


def realignment(unitary: np.ndarray, cut: tuple[int, ...]) -> np.ndarray:
    n = int(round(np.log2(unitary.shape[0])))
    complement = tuple(qubit for qubit in range(n) if qubit not in cut)
    tensor = unitary.reshape((2,) * (2 * n))
    axes = (
        cut
        + tuple(n + qubit for qubit in cut)
        + complement
        + tuple(n + qubit for qubit in complement)
    )
    return tensor.transpose(axes).reshape(
        4 ** len(cut), 4 ** len(complement)
    )


def osr(unitary: np.ndarray, cut: tuple[int, ...]) -> int:
    singular_values = np.linalg.svd(realignment(unitary, cut), compute_uv=False)
    return int(np.count_nonzero(singular_values > 1e-10))


class MultiblockCutBudgetTests(unittest.TestCase):
    def test_random_multiblock_circuits_obey_cumulative_bound(self) -> None:
        rng = np.random.default_rng(20260723)
        for n in (3, 4):
            directed_cnots = [
                (control, target)
                for control in range(n)
                for target in range(n)
                if control != target
            ]
            cuts = tuple((qubit,) for qubit in range(n))
            for _ in range(20):
                matrices = []
                unitary = local_unitary(rng.uniform(-1.0, 1.0, size=(n, 3)))
                for _block in range(3):
                    control, target = directed_cnots[
                        int(rng.integers(len(directed_cnots)))
                    ]
                    matrix = cnot_binary_matrix(n, control, target)
                    matrices.append(matrix)
                    unitary = permutation_unitary(matrix) @ unitary
                    unitary = (
                        local_unitary(rng.uniform(-1.0, 1.0, size=(n, 3)))
                        @ unitary
                    )
                for cut in cuts:
                    cumulative_budget = sum(kappa(matrix, cut) for matrix in matrices)
                    dimension_cap = 4 ** min(len(cut), n - len(cut))
                    self.assertLessEqual(
                        osr(unitary, cut),
                        min(2**cumulative_budget, dimension_cap),
                    )

    def test_independent_cross_cut_cnots_saturate_bound(self) -> None:
        n = 4
        cut = (0, 1)
        first = cnot_binary_matrix(n, 0, 2)
        second = cnot_binary_matrix(n, 1, 3)
        unitary = permutation_unitary(second) @ permutation_unitary(first)
        cumulative_budget = kappa(first, cut) + kappa(second, cut)
        self.assertEqual(cumulative_budget, 2)
        self.assertEqual(osr(unitary, cut), 2**cumulative_budget)

    def test_cancellation_can_make_bound_loose(self) -> None:
        n = 3
        cut = (0,)
        matrix = cnot_binary_matrix(n, 0, 1)
        unitary = permutation_unitary(matrix) @ permutation_unitary(matrix)
        self.assertEqual(kappa(matrix, cut) + kappa(matrix, cut), 2)
        self.assertTrue(np.allclose(unitary, np.eye(2**n)))
        self.assertEqual(osr(unitary, cut), 1)

    def test_three_block_example_has_osr_three(self) -> None:
        n = 3
        cnot_12 = permutation_unitary(cnot_binary_matrix(n, 0, 1))
        cnot_13 = permutation_unitary(cnot_binary_matrix(n, 0, 2))
        local = kron_all([rotation(X, np.pi / 3), I2, I2])
        unitary = cnot_12 @ local @ cnot_13 @ local @ cnot_12
        singular_values = np.linalg.svd(
            realignment(unitary, (0,)),
            compute_uv=False,
        )
        self.assertEqual(osr(unitary, (0,)), 3)
        self.assertTrue(
            np.allclose(
                singular_values,
                np.array([2.0, np.sqrt(3.0), 1.0, 0.0]),
                atol=1e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
