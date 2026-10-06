from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm

from VQC.routing_geometry.core import X, bits_from_index, gadget_unitary, gf2_inverse, index_from_bits, pauli_on_qubits, permutation_unitary, profile, projection_fraction, routing_dictionary


A = np.array(
    [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    dtype=np.uint8,
)
B_PRIME = np.array(
    [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
    dtype=np.uint8,
)


class CoreConventionTests(unittest.TestCase):
    def test_permutation_unitary_matches_binary_action(self) -> None:
        circuit = permutation_unitary(B_PRIME)
        for column in range(16):
            output = (B_PRIME @ bits_from_index(column, 4)) % 2
            self.assertEqual(np.argmax(abs(circuit[:, column])), index_from_bits(output))

    def test_inverse_and_profile_witness(self) -> None:
        identity = np.eye(4, dtype=np.uint8)
        self.assertTrue(np.array_equal((A @ gf2_inverse(A)) % 2, identity))
        self.assertEqual(profile(A), profile(B_PRIME))
        self.assertEqual(profile(A), (2, 2, 4, 2, 4, 0, 2))

    def test_pauli_direction_reversal(self) -> None:
        x4 = pauli_on_qubits(4, {3: "X"})
        x1x4 = pauli_on_qubits(4, {0: "X", 3: "X"})
        self.assertAlmostEqual(projection_fraction(x4, routing_dictionary(A)), 1.0)
        self.assertAlmostEqual(projection_fraction(x4, routing_dictionary(B_PRIME)), 0.0)
        self.assertAlmostEqual(projection_fraction(x1x4, routing_dictionary(A)), 0.0)
        self.assertAlmostEqual(projection_fraction(x1x4, routing_dictionary(B_PRIME)), 1.0)

    def test_identity_initialized_gadget(self) -> None:
        theta = np.zeros((4, 3))
        self.assertTrue(np.allclose(gadget_unitary(A, theta), np.eye(16)))
        target = expm(-0.4j * pauli_on_qubits(4, {3: "X"}))
        self.assertEqual(target.shape, (16, 16))

    


if __name__ == "__main__":
    unittest.main()

