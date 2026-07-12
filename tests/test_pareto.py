import unittest

import numpy as np

from experiments.pareto_landscape import objectives, pareto_mask, scalarized_indices


class ParetoTests(unittest.TestCase):
    def test_objective_shape(self):
        self.assertEqual(objectives(np.array([-1.0, 0.0, 1.0])).shape, (3, 2))

    def test_known_frontier(self):
        values = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 4.0]])
        np.testing.assert_array_equal(pareto_mask(values), [True, True, True, False])

    def test_scalarization_returns_candidate_per_weight(self):
        values = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        result = scalarized_indices(values, np.array([0.0, 0.5, 1.0]))
        self.assertEqual(result.shape, (3,))


if __name__ == "__main__":
    unittest.main()

