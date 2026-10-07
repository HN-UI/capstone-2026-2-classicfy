"""Flow checks must not mistake a fitted window mean for temporal shape."""

import unittest

import numpy as np
import torch

from embedding.evaluation import baseline_predictions, shuffle_visible, temporal_metrics


class TemporalEvaluationTest(unittest.TestCase):
    def test_centering_removes_window_mean_shortcut(self):
        target = np.array([[0., 1, 2, 3], [10., 11, 12, 13]])
        constant = np.array([[1.5] * 4, [11.5] * 4])
        selected = np.ones_like(target, bool)
        result = temporal_metrics(target, constant, selected)
        self.assertGreater(np.corrcoef(target.ravel(), constant.ravel())[0, 1], .9)
        self.assertIsNone(result["shape_correlation"])
        self.assertEqual(result["shape_amplitude_ratio"], 0)
        self.assertEqual(result["slope_mse"], 1)
        self.assertEqual(result["direction_accuracy"], 0)

    def test_shape_and_value_errors_have_different_meaning(self):
        target = np.array([[0., 1, 3, 2]])
        shifted = target + 10
        result = temporal_metrics(target, shifted, np.ones_like(target, bool))
        self.assertEqual(result["mse"], 100)
        self.assertAlmostEqual(result["shape_correlation"], 1)
        self.assertEqual(result["slope_mse"], 0)
        self.assertEqual(result["direction_accuracy"], 1)
        self.assertEqual(result["hidden_adjacent_pairs"], 3)

    def test_no_compressed_gap_or_visible_boundary_in_flow_pairs(self):
        target = np.array([[0., 10, 20, 30, 40]])
        selected = np.array([[True, False, True, True, False]])
        predicted = np.array([[1., 999, 20, 30, 999]])
        result = temporal_metrics(target, predicted, selected)
        self.assertEqual(result["hidden_adjacent_pairs"], 1)
        self.assertEqual(result["slope_mse"], 0)

    def test_baselines_cannot_read_hidden_truth_or_invalid_values(self):
        target = np.array([[[0., 100, 2, 999]]], dtype=np.float32)
        valid = np.array([[[True, True, True, False]]])
        hidden = np.array([[[False, True, False, False]]])
        initial = baseline_predictions(target, valid, hidden)
        target[hidden] = -1000
        changed = baseline_predictions(target, valid, hidden)
        for method in initial:
            np.testing.assert_array_equal(initial[method], changed[method])
        self.assertEqual(initial["visible_mean"][0, 0, 1], 1)
        self.assertEqual(initial["linear_interpolation"][0, 0, 1], 1)

    def test_visible_shuffle_preserves_joint_feature_vectors_and_hidden_values(self):
        values = torch.arange(8, dtype=torch.float32)[None, None].repeat(1, 7, 1)
        values += 100 * torch.arange(7)[None, :, None]
        valid = torch.ones_like(values, dtype=torch.bool)
        valid[:, :, 7] = False
        hidden = torch.zeros_like(valid)
        hidden[:, :, 2:4] = True
        original = values.clone()
        first = shuffle_visible(values, valid, hidden, generator=torch.Generator().manual_seed(5))
        second = shuffle_visible(values, valid, hidden, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(values, original)
        torch.testing.assert_close(first[hidden | ~valid], original[hidden | ~valid])
        positions = torch.tensor([0, 1, 4, 5, 6])
        np.testing.assert_array_equal(np.sort(first[0, 0, positions]), np.sort(original[0, 0, positions]))
        torch.testing.assert_close(first[0] - first[0, :1], original[0] - original[0, :1])


if __name__ == "__main__":
    unittest.main()
