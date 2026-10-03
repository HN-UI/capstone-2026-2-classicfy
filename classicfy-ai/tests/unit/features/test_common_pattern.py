from dataclasses import FrozenInstanceError
import unittest
import warnings

import numpy as np

from features import BeatSequence, PieceFeatureInput, separate_piece_feature


def make_input(key, values, mask=None, *, piece="piece", beats=None, beat_types=None):
    return PieceFeatureInput(
        performance_key=key,
        piece_key=piece,
        score_beats=np.arange(len(values) + 1) if beats is None else beats,
        sequence=BeatSequence(values, [True] * len(values) if mask is None else mask),
        score_beat_types=beat_types,
    )


class SeparatePieceFeatureTest(unittest.TestCase):
    def test_odd_and_even_medians_and_relative_relationship(self) -> None:
        for rows, expected in (
            ([[1, 9], [3, 5]], [2, 7]),
            ([[1, 9], [3, 5], [20, 7]], [3, 7]),
        ):
            with self.subTest(performances=len(rows)):
                results = separate_piece_feature([
                    make_input(str(i), row) for i, row in enumerate(rows)
                ])
                for key, result in results.items():
                    np.testing.assert_allclose(result.raw.values, rows[int(key)])
                    np.testing.assert_allclose(result.common.values, expected)
                    np.testing.assert_allclose(
                        result.relative.values, result.raw.values - result.common.values
                    )
                    np.testing.assert_array_equal(result.common_support, [len(rows)] * 2)
                    self.assertTrue(result.relative.mask.all())

    def test_unequal_performances_keep_their_pairwise_differences(self) -> None:
        results = separate_piece_feature([
            make_input("a", [.1, .9]), make_input("b", [.5, .3]), make_input("c", [.8, .6]),
        ])
        a, b = results["a"], results["b"]
        np.testing.assert_allclose(
            a.relative.values - b.relative.values, a.raw.values - b.raw.values
        )
        self.assertFalse(np.allclose(a.relative.values, 0))
        self.assertLess(a.relative.values[0], 0)
        self.assertGreater(a.relative.values[1], 0)

    def test_identical_values_produce_zero_relative_including_valid_zeros(self) -> None:
        for result in separate_piece_feature([
            make_input("a", [0, .4, 1]), make_input("b", [0, .4, 1]),
        ]).values():
            np.testing.assert_array_equal(result.relative.values, [0, 0, 0])
            np.testing.assert_array_equal(result.common_support, [2, 2, 2])
            self.assertTrue(result.relative.mask.all())

    def test_masks_nan_and_infinities_are_excluded_without_overwriting_raw(self) -> None:
        result = separate_piece_feature([
            make_input("a", [100, np.nan, np.inf, 1], [False, True, True, True]),
            make_input("b", [2, 2, 4, -np.inf]),
            make_input("c", [4, 6, 8, 3]),
        ])["a"]
        np.testing.assert_allclose(result.common.values, [3, 4, 6, 2])
        np.testing.assert_array_equal(result.common_support, [2, 2, 2, 2])
        np.testing.assert_array_equal(result.raw.mask, [False, True, True, True])
        np.testing.assert_array_equal(result.relative.mask, [False, False, False, True])
        np.testing.assert_allclose(result.relative.values, [np.nan, np.nan, np.nan, -1])
        self.assertEqual(result.raw.values[0], 100)
        self.assertTrue(np.isposinf(result.raw.values[2]))

    def test_minimum_support_masks_common_and_relative_only(self) -> None:
        result = separate_piece_feature([
            make_input("a", [1, 2, np.nan]),
            make_input("b", [3, 4, np.nan], [True, False, False]),
            make_input("c", [5, 6, np.nan], [True, False, False]),
        ], minimum_support=3)["a"]
        np.testing.assert_array_equal(result.common_support, [3, 1, 0])
        np.testing.assert_array_equal(result.common.mask, [True, False, False])
        np.testing.assert_allclose(result.common.values, [3, np.nan, np.nan])
        np.testing.assert_allclose(result.relative.values, [-2, np.nan, np.nan])
        self.assertEqual(result.raw.values[1], 2)
        self.assertTrue(result.raw.mask[1])
        self.assertEqual(result.minimum_support, 3)

    def test_all_missing_and_threshold_above_group_size_are_fully_masked(self) -> None:
        for inputs, minimum in (
            ([make_input("a", [np.nan]), make_input("b", [np.nan])], 2),
            ([make_input("a", [1]), make_input("b", [2])], 3),
        ):
            with self.subTest(minimum=minimum), warnings.catch_warnings():
                warnings.simplefilter("error")
                result = separate_piece_feature(inputs, minimum_support=minimum)["a"]
                self.assertTrue(np.isnan(result.common.values).all())
                self.assertTrue(np.isnan(result.relative.values).all())
                self.assertFalse(result.common.mask.any())
                self.assertFalse(result.relative.mask.any())

    def test_zero_width_score_intervals_do_not_contribute_to_support(self) -> None:
        result = separate_piece_feature([
            make_input("a", [1, 99, 3], beats=[0, 1, 1, 2]),
            make_input("b", [3, 99, 5], beats=[0, 1, 1, 2]),
        ])["a"]
        np.testing.assert_array_equal(result.common_support, [2, 0, 2])
        np.testing.assert_allclose(result.common.values, [2, np.nan, 4])
        np.testing.assert_array_equal(result.relative.mask, [True, False, True])
        self.assertTrue(result.raw.mask[1])
        self.assertEqual(result.raw.values[1], 99)

    def test_valid_zero_counts_toward_median_and_support(self) -> None:
        result = separate_piece_feature([
            make_input("a", [0]), make_input("b", [2]), make_input("c", [4]),
        ])["a"]
        self.assertEqual(result.common_support[0], 3)
        self.assertEqual(result.common.values[0], 2)
        self.assertEqual(result.relative.values[0], -2)
        self.assertTrue(result.relative.mask[0])

    def test_different_pieces_grids_lengths_and_duplicate_keys_are_rejected(self) -> None:
        first = make_input("a", [1, 2])
        for other, error in (
            (make_input("b", [1, 2], piece="piece_extra_repeat"), "same piece"),
            (make_input("b", [1, 2], beats=[0, .5, 2]), "same score beat positions"),
            (make_input("b", [1]), "same interval count"),
            (make_input("a", [1, 2]), "Duplicate performance key"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                separate_piece_feature([first, other])

    def test_score_beat_types_must_match_when_provided(self) -> None:
        first = make_input("a", [1], beat_types=["db", "b"])
        for beat_types in (["db", "bR"], None):
            with self.subTest(types=beat_types), self.assertRaisesRegex(ValueError, "same score beat types"):
                separate_piece_feature([first, make_input("b", [1], beat_types=beat_types)])

    def test_score_grid_roundoff_is_accepted(self) -> None:
        result = separate_piece_feature([
            make_input("a", [1], beats=[0, 1]),
            make_input("b", [3], beats=[0, 1 + 1e-10]),
        ])["a"]
        self.assertEqual(result.common.values[0], 2)

    def test_empty_singleton_and_invalid_support_are_rejected(self) -> None:
        for inputs in ([], [make_input("a", [1])]):
            with self.assertRaisesRegex(ValueError, "At least two performances"):
                separate_piece_feature(inputs)
        for minimum in (True, 1, 0, -1, 2.5, "2"):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, "minimum_support"):
                separate_piece_feature(
                    [make_input("a", [1]), make_input("b", [2])], minimum_support=minimum
                )

    def test_snapshots_are_readonly_and_do_not_share_input_storage(self) -> None:
        source = BeatSequence(np.array([1, 3]), np.array([True, False]))
        beats = np.array([0., 1., 2.])
        types = ["db", "b", "b"]
        first = PieceFeatureInput("a", "piece", beats, source, types)
        second = make_input("b", [5, 7], beat_types=types)
        result = separate_piece_feature([first, second])["a"]
        np.testing.assert_array_equal(source.values, [1, 3])
        np.testing.assert_array_equal(source.mask, [True, False])
        beats[:] = -1
        types[0] = "bR"
        # 기존 BeatSequence는 setflags로 수정 가능해도 새 결과의 독립 복사본에는 영향이 없다.
        source.values.setflags(write=True)
        source.values[:] = 99
        np.testing.assert_array_equal(first.sequence.values, [1, 3])
        np.testing.assert_array_equal(result.raw.values, [1, 3])
        np.testing.assert_array_equal(result.score_beats, [0, 1, 2])
        self.assertEqual(result.score_beat_types, ("db", "b", "b"))
        self.assertEqual(result.raw.values.dtype.kind, "i")
        for seq in (first.sequence, result.raw, result.common, result.relative):
            for array in (seq.values, seq.mask):
                self.assertFalse(array.flags.writeable)
                with self.assertRaises(ValueError):
                    array[0] = 0
        with self.assertRaises(ValueError):
            result.common_support[0] = 0
        with self.assertRaises(FrozenInstanceError):
            result.piece_key = "other"


class PieceFeatureInputTest(unittest.TestCase):
    def test_invalid_score_grid_and_sequence_length_are_rejected(self) -> None:
        for beats in ([0], [1, 0], [0, np.nan], [0, np.inf], [[0, 1]], ["0", "1"]):
            with self.subTest(beats=beats), self.assertRaises(ValueError):
                make_input("a", [1], beats=beats)
        with self.assertRaisesRegex(ValueError, "length len"):
            make_input("a", [1], beats=[0, 1, 2])

    def test_non_numeric_values_invalid_types_and_empty_identifiers_are_rejected(self) -> None:
        for values in (["1"], [1j], [True]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "real numbers"):
                make_input("a", values)
        for beat_types in (["db"], ["db", "bad"]):
            with self.assertRaisesRegex(ValueError, "score_beat_types"):
                make_input("a", [1], beat_types=beat_types)
        for key, piece in (("", "piece"), ("a", " "), (None, "piece")):
            with self.subTest(key=key, piece=piece), self.assertRaises(ValueError):
                make_input(key, [1], piece=piece)


if __name__ == "__main__":
    unittest.main()
