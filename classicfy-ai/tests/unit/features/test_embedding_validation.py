"""Numerical guarantees for the analysis, including unbiased retrieval ties."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from validate_embedding import retained_indices, retrieval_metrics, summarize, weighted_distances


class EmbeddingValidationTest(unittest.TestCase):
    def test_identical_candidates_do_not_get_arbitrary_perfect_retrieval(self):
        metrics = retrieval_metrics(np.zeros((3, 3)), np.arange(3))
        np.testing.assert_allclose(metrics["top1"], 1 / 3)
        np.testing.assert_allclose(metrics["mrr"], (1 + 1/2 + 1/3) / 3)
        np.testing.assert_array_equal(metrics["margin"], 0)

    def test_rank_and_partial_ties(self):
        metrics = retrieval_metrics(np.array([[.2, .2, .7], [.1, .3, .3]]), np.array([1, 2]))
        np.testing.assert_allclose(metrics["top1"], [.5, 0])
        np.testing.assert_allclose(metrics["mrr"], [.75, (1/2 + 1/3)/2])
        self.assertEqual(metrics["margin"][0], 0)
        self.assertLess(metrics["margin"][1], 0)

    def test_pedal_has_same_block_weight_despite_six_coordinates(self):
        gallery = np.zeros((1, 14))
        queries = np.zeros((2, 14))
        queries[0, :2] = 3
        queries[1, 8:] = 3
        np.testing.assert_allclose(weighted_distances(queries, gallery, "all"), np.sqrt(9/5))

    def test_omitted_feature_cannot_change_distance(self):
        queries, gallery = np.zeros((1, 14)), np.zeros((1, 14))
        queries[:, 4:6] = 10
        self.assertGreater(weighted_distances(queries, gallery, "all")[0, 0], 0)
        self.assertEqual(weighted_distances(queries, gallery, "without_Dynamics")[0, 0], 0)

    def test_zero_dropout_recovers_exact_original_summary_without_mutation(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(7, 100))
        copy = values.copy()
        kept = retained_indices(100, 0, "clean", rng)
        np.testing.assert_allclose(summarize(values[:, kept]), summarize(values), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(values, copy)
        self.assertEqual(weighted_distances(summarize(values)[None], summarize(values)[None], "all")[0, 0], 0)

    def test_dropout_counts_reproducibility_and_contiguous_gap(self):
        for pattern in ("random", "contiguous"):
            a = retained_indices(100, .2, pattern, np.random.default_rng(7))
            b = retained_indices(100, .2, pattern, np.random.default_rng(7))
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(a), 80)
            self.assertEqual(len(set(a)), 80)
            if pattern == "contiguous":
                missing = np.setdiff1d(np.arange(100), a)
                np.testing.assert_array_equal(np.diff(missing), 1)

    def test_summary_requires_finite_shared_beats(self):
        for array in (np.zeros((6, 10)), np.zeros((7, 1)), np.full((7, 10), np.nan)):
            with self.assertRaises(ValueError):
                summarize(array)


if __name__ == "__main__":
    unittest.main()
