"""Scientific controls: distribution preservation, perturbation size and scoring."""

import unittest
from types import SimpleNamespace

import numpy as np

from embedding.order_evaluation import (ORDERS, make_variants, summary_vectors,
    summary_distance, raw_distance, cosine_distance, triplet_scores,
    select_complete_nonoverlapping)


class OrderEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(7).normal(size=(3, 7, 64)).astype(np.float32)

    def test_joint_permutation_preserves_values_and_summary_exactly(self):
        variants, permutations = make_variants(self.x)
        original = self.x.copy()
        for name in ORDERS:
            for row in range(len(self.x)):
                np.testing.assert_array_equal(variants[name][row], self.x[row][:, permutations[name][row]])
            np.testing.assert_array_equal(np.sort(variants[name], axis=2), np.sort(self.x, axis=2))
            np.testing.assert_array_equal(summary_vectors(variants[name]), summary_vectors(self.x))
            np.testing.assert_array_equal(summary_distance(summary_vectors(self.x), summary_vectors(variants[name])), 0)
        np.testing.assert_array_equal(self.x, original)

    def test_noise_rms_and_constant_channels(self):
        self.x[:, 0] = 4
        variants, _ = make_variants(self.x)
        for level in (.02, .05, .10):
            difference = variants[f"noise{level:.2f}"] - self.x
            np.testing.assert_allclose(np.sqrt(np.mean(difference ** 2, axis=2)),
                                       level * self.x.std(axis=2), atol=2e-7)
            np.testing.assert_array_equal(difference[:, 0], 0)
        other, _ = make_variants(self.x)
        for name in variants:
            np.testing.assert_array_equal(variants[name], other[name])

    def test_existing_summary_definition(self):
        actual = summary_vectors(self.x).reshape(-1, 7, 2)
        expected = self.x.astype(float).mean(axis=2)
        expected[:, 1] = np.median(abs(self.x[:, 1]), axis=1)
        np.testing.assert_allclose(actual[:, :, 0], expected)
        np.testing.assert_allclose(actual[:, :, 1],
            np.percentile(self.x, 95, axis=2) - np.percentile(self.x, 5, axis=2))

    def test_scoring_ties_and_undefined_cosine(self):
        scores, ties = triplet_scores([.1, .3, .2, np.nan], [.3, .1, .2, .1])
        np.testing.assert_array_equal(scores[:3], [1, 0, .5])
        self.assertTrue(np.isnan(scores[3]))
        np.testing.assert_array_equal(ties, [False, False, True, False])
        d = cosine_distance([[1, 0], [0, 0]], [[0, 1], [1, 0]])
        self.assertEqual(d[0], 1)
        self.assertTrue(np.isnan(d[1]))
        np.testing.assert_array_equal(raw_distance(self.x, self.x), 0)

    def test_selection_excludes_gaps_and_overlaps(self):
        dataset = SimpleNamespace(items=[(0, 0), (0, 32), (0, 64), (1, 0), (1, 32)], window_size=64)
        valid = np.ones((5, 7, 64), bool)
        valid[3, 0, 10] = False
        np.testing.assert_array_equal(select_complete_nonoverlapping(dataset, valid), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
