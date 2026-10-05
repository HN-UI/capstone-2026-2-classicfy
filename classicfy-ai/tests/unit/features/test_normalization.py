from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from features import (
    BeatSequence, PieceFeatureInput, fit_residual_scale, separate_piece_feature,
    standardize_piece_feature, standardize_sequence,
)


def sequence(values, mask=None):
    return BeatSequence(values, np.ones(len(values), bool) if mask is None else mask)


def separated(values, masks=None):
    return list(separate_piece_feature([
        PieceFeatureInput(str(i), "piece", np.arange(len(row) + 1),
                          sequence(row, masks[i] if masks else None))
        for i, row in enumerate(values)
    ]).values())


class ResidualScaleTest(unittest.TestCase):
    def test_mad_and_iqr_use_normal_consistency_factors(self):
        scale = fit_residual_scale([sequence([-2., -1., 0., 1., 2.])])
        self.assertAlmostEqual(scale.value, 1.4826)
        self.assertAlmostEqual(scale.iqr, 2 / 1.3489795003921634)
        self.assertAlmostEqual(scale.std, np.sqrt(2))
        self.assertEqual(scale.valid_count, 5)
        self.assertEqual(scale.method, "mad")

    def test_mask_and_nonfinite_values_are_excluded(self):
        scale = fit_residual_scale([sequence([-2, 0, 2, 999, np.nan, np.inf], [1, 1, 1, 0, 1, 1])])
        self.assertAlmostEqual(scale.value, 2 * 1.4826)
        self.assertEqual(scale.valid_count, 3)

    def test_mad_falls_back_to_iqr(self):
        scale = fit_residual_scale([sequence([0, 0, 0, 2])])
        self.assertEqual(scale.method, "iqr")
        self.assertEqual(scale.mad, 0)
        self.assertAlmostEqual(scale.value, .5 / 1.3489795003921634)

    def test_sparse_distribution_falls_back_to_std(self):
        for method in ("mad", "iqr", "std"):
            scale = fit_residual_scale([sequence([0] * 8 + [2])], method=method)
            self.assertEqual(scale.method, "std")
            self.assertAlmostEqual(scale.value, np.sqrt(32) / 9)

    def test_tiny_roundoff_is_not_a_denominator(self):
        scale = fit_residual_scale([sequence([0, 0, 1e-16, 1e-16, 1])])
        self.assertEqual(scale.method, "std")
        self.assertGreater(scale.value, .1)

    def test_degenerate_spread_uses_unit_scale_without_removing_validity(self):
        for values in ([0, 0], [3, 3]):
            scale = fit_residual_scale([sequence(values)])
            result = standardize_sequence(sequence(values), scale)
            self.assertEqual(scale.method, "unit")
            self.assertEqual(scale.value, 1)
            np.testing.assert_array_equal(result.values, values)
            self.assertTrue(result.mask.all())

    def test_empty_fit_has_no_scale_and_masks_transform(self):
        for sequences in ([], [sequence([1, np.nan], [0, 1])]):
            scale = fit_residual_scale(sequences)
            self.assertEqual(scale.method, "empty")
            self.assertTrue(np.isnan(scale.value))
            output = standardize_sequence(sequence([0, 1]), scale)
            self.assertTrue(np.isnan(output.values).all())
            self.assertFalse(output.mask.any())

    def test_transform_divides_without_recentering_or_clipping(self):
        scale = fit_residual_scale([sequence([1, 2, 3])])
        output = standardize_sequence(sequence([10, 1e6, -5, np.nan], [1, 1, 0, 1]), scale)
        np.testing.assert_allclose(output.values, [10 / 1.4826, 1e6 / 1.4826, np.nan, np.nan])
        np.testing.assert_array_equal(output.mask, [1, 1, 0, 0])

    def test_fitted_scale_can_be_reused_without_refitting(self):
        scale = fit_residual_scale([sequence([-1, 0, 1])], method="std")
        self.assertAlmostEqual(standardize_sequence(sequence([100]), scale).values[0], 100 / np.sqrt(2 / 3))
        self.assertEqual(scale.valid_count, 3)

    def test_unknown_method_and_overflow_raise_errors(self):
        with self.assertRaises(ValueError):
            fit_residual_scale([], method="invalid")
        with self.assertRaisesRegex(ValueError, "overflow"):
            fit_residual_scale([sequence([-1e308, 1e308])])


class PieceStandardizationTest(unittest.TestCase):
    def test_feature_specific_defaults_and_shared_piece_scale(self):
        inputs = separated([[0, 2, 4], [2, 4, 6]])
        for channel, expected, denominator in (
            ("dynamics", "mad", 1.4826), ("articulation", "mad", 1.4826),
            ("pedal_depth", "std", 1), ("pedal_down_ratio", "std", 1), ("pedal_changes", "std", 1),
        ):
            results = standardize_piece_feature(inputs, feature_name=channel)
            for key, result in results.items():
                self.assertEqual(result.scale.requested_method, expected)
                self.assertAlmostEqual(result.scale.value, denominator)
                self.assertEqual(result.scale.valid_count, 6)
                np.testing.assert_allclose(result.standardized.values, np.array([-1, -1, -1]) / denominator if key == "0" else np.array([1, 1, 1]) / denominator)
                np.testing.assert_allclose(result.relative.values, result.raw.values - result.common.values)
            a, b = results.values()
            np.testing.assert_allclose(a.standardized.values - b.standardized.values,
                                       (a.raw.values - b.raw.values) / denominator)

    def test_method_override_and_no_double_common_subtraction(self):
        inputs = separated([[10, 20], [12, 24]])
        output = standardize_piece_feature(inputs, feature_name="dynamics", method="std")["0"]
        np.testing.assert_allclose(output.relative.values, [-1, -2])
        np.testing.assert_allclose(output.standardized.values, np.array([-1, -2]) / np.sqrt(2.5))
        np.testing.assert_array_equal(output.common.values, [11, 22])

    def test_masks_and_support_preserved_and_inputs_immutable(self):
        inputs = separated([[0, 999, np.nan], [2, 4, 6], [4, 6, 8]], [[1, 0, 1], [1, 1, 1], [1, 1, 1]])
        original = inputs[0]
        result = standardize_piece_feature(inputs, feature_name="dynamics")["0"]
        for field in ("raw", "common", "relative"):
            np.testing.assert_array_equal(getattr(result, field).values, getattr(original, field).values)
            np.testing.assert_array_equal(getattr(result, field).mask, getattr(original, field).mask)
            self.assertFalse(np.shares_memory(getattr(result, field).values, getattr(original, field).values))
        np.testing.assert_array_equal(result.standardized.mask, [1, 0, 0])
        np.testing.assert_array_equal(result.common_support, [3, 2, 2])
        for array in (result.standardized.values, result.standardized.mask, result.common_support):
            with self.assertRaises(ValueError):
                array[0] = 99
        with self.assertRaises(FrozenInstanceError):
            result.feature_name = "other"

    def test_identical_performances_remain_valid_zero(self):
        for channel in ("dynamics", "pedal_changes"):
            results = standardize_piece_feature(separated([[0, 0], [0, 0]]), feature_name=channel)
            for result in results.values():
                self.assertEqual(result.scale.method, "unit")
                self.assertTrue(result.standardized.mask.all())
                np.testing.assert_array_equal(result.standardized.values, [0, 0])

    def test_insufficient_support_remains_missing(self):
        results = standardize_piece_feature(separated([[1, 2], [3, 4]], [[1, 0], [0, 1]]), feature_name="dynamics")
        for result in results.values():
            self.assertEqual(result.scale.method, "empty")
            self.assertFalse(result.standardized.mask.any())
            self.assertTrue(np.isnan(result.standardized.values).all())

    def test_incompatible_piece_grid_or_common_pattern_rejected(self):
        features = separated([[0, 1], [1, 2]])
        for altered in (
            replace(features[1], piece_key="other"),
            replace(features[1], score_beats=np.array([0, 2, 3])),
            replace(features[1], common=sequence([9, 9])),
            replace(features[1], common_support=np.array([3, 3])),
        ):
            with self.assertRaises(ValueError):
                standardize_piece_feature([features[0], altered], feature_name="dynamics")
        for items in (features[:1], [features[0], features[0]]):
            with self.assertRaises(ValueError):
                standardize_piece_feature(items, feature_name="dynamics")
        with self.assertRaises(ValueError):
            standardize_piece_feature(features, feature_name="tempo")
        with self.assertRaises(ValueError):
            standardize_piece_feature(features, feature_name="dynamics", method="")


if __name__ == "__main__":
    unittest.main()
