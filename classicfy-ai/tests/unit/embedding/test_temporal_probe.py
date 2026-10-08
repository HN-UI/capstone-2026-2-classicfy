"""Check target semantics, work separation and train-only weighted linear probes."""

import unittest

import numpy as np

from embedding.temporal_probe import (quarter_targets, sample_weights, work_folds,
    fit_ridge, predict_ridge, fit_pca, project_pca, evaluate_channels)


class TemporalProbeTest(unittest.TestCase):
    def test_targets_remove_level_and_retain_quarter_order(self):
        x = np.tile(np.repeat([1., 2., 3., 4.], 16), (2, 7, 1))
        target = quarter_targets(x)
        np.testing.assert_allclose(target[0, 0], [-1.5, -.5, .5, 1.5])
        np.testing.assert_allclose(target.mean(axis=2), 0)
        np.testing.assert_allclose(quarter_targets(x + 100), target)
        np.testing.assert_allclose(quarter_targets(x[:, :, ::-1]), target[:, :, ::-1])
        np.testing.assert_array_equal(quarter_targets(np.ones((1, 7, 64))), 0)

    def test_equal_work_performance_weights_ignore_duplicate_windows(self):
        weights = sample_weights(["A", "A", "A", "B"], ["a", "a", "b", "c"])
        np.testing.assert_allclose(weights, [.125, .125, .25, .5])
        duplicated = sample_weights(["A", "A", "A", "A", "A", "B"], ["a", "a", "a", "a", "b", "c"])
        self.assertAlmostEqual(weights[:2].sum(), duplicated[:4].sum())

    def test_folds_never_split_a_work_and_are_reproducible(self):
        pieces = np.repeat([f"work{i}" for i in range(11)], 3)
        folds, mapping = work_folds(pieces)
        for work in set(pieces):
            self.assertEqual(len(set(folds[pieces == work])), 1)
        np.testing.assert_array_equal(work_folds(pieces)[0], folds)
        self.assertEqual(set(mapping.values()), set(range(5)))

    def test_linear_probe_recovers_known_mapping_and_keeps_training_statistics(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(100, 6))
        coefficients = rng.normal(size=(6, 28))
        targets = (x @ coefficients).reshape(-1, 7, 4)
        model = fit_ridge(x[:80], targets[:80], np.ones(80) / 80, 1e-10)
        np.testing.assert_allclose(predict_ridge(model, x[80:]), targets[80:], atol=2e-8)
        expected_mean = x[:80].mean(axis=0)
        np.testing.assert_allclose(model["mean"], expected_mean)
        original = model["mean"].copy()
        predict_ridge(model, x[80:] + 1000)
        np.testing.assert_array_equal(model["mean"], original)

    def test_pca_uses_training_only_and_has_requested_dimensions(self):
        x = np.random.default_rng(5).normal(size=(100, 20))
        pca = fit_pca(x[:80], np.ones(80) / 80, 8)
        np.testing.assert_allclose(pca["mean"], x[:80].mean(axis=0))
        np.testing.assert_allclose(pca["components"].T @ pca["components"], np.eye(8), atol=1e-12)
        self.assertEqual(project_pca(pca, x[80:]).shape, (20, 8))

    def test_flat_reference_and_perfect_scores(self):
        target = quarter_targets(np.random.default_rng(8).normal(size=(5, 7, 64)))
        weights = np.ones(5) / 5
        flat = evaluate_channels(target, np.zeros_like(target), weights)
        perfect = evaluate_channels(target, target, weights)
        np.testing.assert_allclose(flat["score_percent"], 0)
        np.testing.assert_allclose(perfect["score_percent"], 100)
        np.testing.assert_allclose(perfect["centered_cosine"], 1)


if __name__ == "__main__":
    unittest.main()
