import unittest

import numpy as np

import multiblock_stagnation_intervention as experiment


class MultiblockInterventionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 4
        self.rng = np.random.default_rng(731)
        (
            self.target,
            self.old_ops,
            self.known_theta,
            _,
            _,
        ) = experiment.build_instance(self.n, 2, self.rng)

    def test_analytic_gradient_matches_central_difference(self) -> None:
        theta = self.known_theta + self.rng.normal(0.0, 0.08, self.known_theta.size)
        loss, gradient, _ = experiment.loss_and_gradient(
            self.old_ops, theta, self.target, self.n
        )
        finite_difference = np.zeros_like(theta)
        epsilon = 1e-6
        for index in range(theta.size):
            plus = theta.copy()
            minus = theta.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_loss = experiment.loss_and_gradient(
                self.old_ops, plus, self.target, self.n
            )[0]
            minus_loss = experiment.loss_and_gradient(
                self.old_ops, minus, self.target, self.n
            )[0]
            finite_difference[index] = (plus_loss - minus_loss) / (2 * epsilon)
        self.assertGreater(loss, 0.0)
        np.testing.assert_allclose(gradient, finite_difference, atol=2e-8, rtol=2e-7)

    def test_controlled_old_point_is_true_constrained_stationarity(self) -> None:
        old_geometry = experiment.geometry(
            self.old_ops, self.known_theta, self.target, self.n
        )
        self.assertAlmostEqual(old_geometry.loss, 0.75, places=12)
        self.assertGreater(old_geometry.ambient_norm, 0.8)
        self.assertLess(old_geometry.projected_norm, 1e-10)
        self.assertLess(old_geometry.chi, 1e-10)

    def test_identity_initialized_gadget_preserves_physical_point(self) -> None:
        old_geometry = experiment.geometry(
            self.old_ops, self.known_theta, self.target, self.n
        )
        candidates = experiment.random_candidate_pool(
            self.n, required_rank=2, pool_size=3, rng=self.rng
        )
        candidates, _ = experiment.score_candidates(
            candidates,
            self.old_ops,
            self.known_theta,
            old_geometry,
            self.target,
            self.n,
        )
        for candidate in candidates:
            self.assertEqual(candidate.initial_state_error, 0.0)
            self.assertEqual(candidate.initial_loss_error, 0.0)
            self.assertEqual(candidate.initial_gradient_error, 0.0)
            self.assertEqual(candidate.initial_ambient_error, 0.0)
            self.assertLess(candidate.key_kappa, 2)


if __name__ == "__main__":
    unittest.main()
