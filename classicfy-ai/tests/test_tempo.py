import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from features import TempoInput, extract_piece_tempo_features
from preprocessing import AsapSample


class ExtractPieceTempoFeaturesTest(unittest.TestCase):
    @staticmethod
    def _input(
        key: str,
        performance_beats: list[float],
        beat_types: list[str] | None = None,
    ) -> TempoInput:
        types = beat_types or ["db", "b", "b", "db"]
        return TempoInput(
            performance_key=key,
            score_beats=[0.0, 0.5, 1.0, 1.5],
            performance_beats=performance_beats,
            score_beat_types=["db", "b", "b", "db"],
            performance_beat_types=types,
        )

    @staticmethod
    def _sample(aligned: bool = True) -> AsapSample:
        return AsapSample(
            performance_key="piece/performance.mid",
            composer="Composer",
            title="Piece",
            score_path=Path("piece/score.mid"),
            performance_path=Path("piece/performance.mid"),
            aligned=aligned,
            score_beats=[0.0, 0.5, 1.0],
            performance_beats=[0.0, 0.6, 1.2],
            score_beat_types=["db", "b", "b"],
            performance_beat_types=["db", "b", "b"],
            score_downbeats=[0.0],
            performance_downbeats=[0.0],
            score_time_signatures={},
            performance_time_signatures={},
        )

    def test_builds_input_from_aligned_asap_sample(self) -> None:
        sample = self._sample()

        tempo_input = TempoInput.from_asap_sample(sample)

        self.assertEqual(tempo_input.performance_key, sample.performance_key)
        self.assertEqual(tempo_input.score_beats, sample.score_beats)
        self.assertEqual(tempo_input.performance_beats, sample.performance_beats)
        self.assertIsNot(tempo_input.score_beats, sample.score_beats)

    def test_rejects_unaligned_asap_sample(self) -> None:
        sample = replace(self._sample(), aligned=False)

        with self.assertRaisesRegex(ValueError, "not aligned"):
            TempoInput.from_asap_sample(sample)

    def test_extracts_score_common_and_individual_tempo(self) -> None:
        features = extract_piece_tempo_features(
            [
                self._input("fast.mid", [0.0, 0.4, 0.8, 1.2]),
                self._input("middle.mid", [0.0, 0.5, 1.0, 1.5]),
                self._input("slow.mid", [0.0, 1.0, 2.0, 3.0]),
            ]
        )

        fast = features["fast.mid"]
        middle = features["middle.mid"]
        slow = features["slow.mid"]

        for interval in fast.intervals:
            self.assertAlmostEqual(interval.score_relative_tempo, math.log2(1.25))
        np.testing.assert_allclose(middle.common_tempo_sequence.values, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            middle.individual_tempo_sequence.values, [0.0, 0.0, 0.0]
        )
        self.assertGreater(fast.overall_individual_tempo, 0.0)
        self.assertLess(slow.overall_individual_tempo, 0.0)

    def test_preserves_intervals_touching_br_as_special(self) -> None:
        features = extract_piece_tempo_features(
            [
                self._input(
                    "first.mid",
                    [0.0, 0.5, 1.0, 1.5],
                    ["db", "b", "bR", "db"],
                ),
                self._input(
                    "second.mid",
                    [0.0, 0.6, 1.2, 1.8],
                    ["db", "b", "bR", "db"],
                ),
            ]
        )

        first = features["first.mid"]
        self.assertEqual(first.intervals[0].status, "regular")
        self.assertEqual(first.intervals[1].status, "special")
        self.assertEqual(first.intervals[1].status_reason, "contains_bR")
        self.assertFalse(first.intervals[1].mask)
        self.assertIsNone(first.intervals[1].score_relative_tempo)
        self.assertEqual(first.common_tempo_sequence.mask.tolist(), [True, False, False])
        self.assertEqual(
            first.individual_tempo_sequence.mask.tolist(), [True, False, False]
        )

    def test_uses_available_regular_performances_for_common_tempo(self) -> None:
        features = extract_piece_tempo_features(
            [
                self._input("regular.mid", [0.0, 0.5, 1.0, 1.5]),
                self._input(
                    "special.mid",
                    [0.0, 0.5, 1.0, 1.5],
                    ["db", "bR", "b", "db"],
                ),
            ]
        )

        regular = features["regular.mid"]
        self.assertEqual(regular.common_tempo_sequence.mask.tolist(), [False, False, True])
        np.testing.assert_allclose(
            regular.common_tempo_sequence.values,
            [np.nan, np.nan, 0.0],
            equal_nan=True,
        )
        self.assertEqual(regular.common_tempo_support.tolist(), [1, 1, 2])
        self.assertEqual(
            regular.individual_tempo_sequence.mask.tolist(), [False, False, True]
        )
        self.assertEqual(
            features["special.mid"].individual_tempo_sequence.mask.tolist(),
            [False, False, True],
        )

    def test_marks_extreme_peer_and_score_deviation_as_suspicious(self) -> None:
        features = extract_piece_tempo_features(
            [
                self._input("first.mid", [0.0, 0.5, 1.0, 1.5]),
                self._input("second.mid", [0.0, 0.6, 1.1, 1.6]),
                self._input("third.mid", [0.0, 0.55, 1.05, 1.55]),
                self._input("extreme.mid", [0.0, 0.0005, 0.5005, 1.0005]),
            ]
        )

        interval = features["extreme.mid"].intervals[0]
        self.assertEqual(interval.status, "suspicious")
        self.assertEqual(interval.status_reason, "extreme_score_and_peer_deviation")
        self.assertFalse(interval.mask)
        self.assertIsNotNone(interval.score_relative_tempo)
        self.assertGreater(interval.peer_log2_deviation, 3.0)
        self.assertFalse(features["extreme.mid"].individual_tempo_sequence.mask[0])
        self.assertTrue(
            np.isnan(features["extreme.mid"].individual_tempo_sequence.values[0])
        )

    def test_keeps_shared_large_score_deviation_as_regular(self) -> None:
        features = extract_piece_tempo_features(
            [
                self._input("first.mid", [0.0, 0.05, 0.10, 0.15]),
                self._input("second.mid", [0.0, 0.05, 0.10, 0.15]),
                self._input("third.mid", [0.0, 0.05, 0.10, 0.15]),
                self._input("fourth.mid", [0.0, 0.05, 0.10, 0.15]),
            ]
        )

        self.assertTrue(
            all(
                interval.status == "regular"
                for interval in features["first.mid"].intervals
            )
        )

    def test_masks_zero_duration_interval_without_shifting_sequence(self) -> None:
        features = extract_piece_tempo_features(
            [
                self._input("duplicate.mid", [0.0, 0.5, 0.5, 1.5]),
                self._input("regular.mid", [0.0, 0.6, 1.1, 1.6]),
            ]
        )

        duplicate = features["duplicate.mid"]
        interval = duplicate.intervals[1]
        self.assertFalse(interval.mask)
        self.assertEqual(interval.status, "invalid")
        self.assertEqual(interval.status_reason, "zero_duration")
        self.assertIsNone(interval.score_relative_tempo)
        self.assertEqual(
            duplicate.individual_tempo_sequence.mask.tolist(), [True, False, True]
        )

    def test_requires_multiple_performances_of_the_same_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least two performances"):
            extract_piece_tempo_features(
                [self._input("only.mid", [0.0, 0.5, 1.0, 1.5])]
            )

        different_score = TempoInput(
            performance_key="different.mid",
            score_beats=[0.0, 0.4, 1.0, 1.5],
            performance_beats=[0.0, 0.5, 1.0, 1.5],
            score_beat_types=["db", "b", "b", "db"],
            performance_beat_types=["db", "b", "b", "db"],
        )
        with self.assertRaisesRegex(ValueError, "same score beat positions"):
            extract_piece_tempo_features(
                [
                    self._input("first.mid", [0.0, 0.5, 1.0, 1.5]),
                    different_score,
                ]
            )

    def test_rejects_invalid_beat_data(self) -> None:
        invalid_inputs = [
            self._input("decreasing.mid", [0.0, 0.5, 0.4, 1.5]),
            self._input("nan.mid", [0.0, 0.5, math.nan, 1.5]),
            self._input(
                "short-types.mid", [0.0, 0.5, 1.0, 1.5], ["db", "b"]
            ),
        ]

        for invalid_input in invalid_inputs:
            with self.subTest(key=invalid_input.performance_key):
                with self.assertRaises(ValueError):
                    extract_piece_tempo_features(
                        [
                            self._input("valid.mid", [0.0, 0.5, 1.0, 1.5]),
                            invalid_input,
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
