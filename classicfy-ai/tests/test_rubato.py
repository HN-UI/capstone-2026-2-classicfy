import math
import unittest

from features import (
    TempoInput,
    extract_piece_rubato_features,
    extract_piece_tempo_features,
)


class ExtractPieceRubatoFeaturesTest(unittest.TestCase):
    @staticmethod
    def _input(key: str, intervals: list[float]) -> TempoInput:
        performance_beats = [0.0]
        for interval in intervals:
            performance_beats.append(performance_beats[-1] + interval)
        return TempoInput(
            performance_key=key,
            score_beats=[0.0, 0.5, 1.0, 1.5],
            performance_beats=performance_beats,
            score_beat_types=["db", "b", "b", "db"],
            performance_beat_types=["db", "b", "b", "db"],
        )

    def test_separates_absolute_common_and_relative_rubato(self) -> None:
        tempo = extract_piece_tempo_features(
            [
                self._input("variable.mid", [0.4, 0.5, 0.6]),
                self._input("middle.mid", [0.5, 0.5, 0.5]),
                self._input("opposite.mid", [0.6, 0.5, 0.4]),
            ]
        )

        rubato = extract_piece_rubato_features(tempo)["variable.mid"]

        self.assertGreater(rubato.absolute_rubato_sequence[0], 0.0)
        self.assertAlmostEqual(rubato.absolute_rubato_sequence[1], 0.0)
        self.assertLess(rubato.absolute_rubato_sequence[2], 0.0)
        for value in rubato.common_rubato_sequence:
            self.assertAlmostEqual(value, 0.0)
        self.assertEqual(
            rubato.relative_rubato_sequence,
            rubato.absolute_rubato_sequence,
        )
        self.assertAlmostEqual(
            rubato.absolute_rubato_amount,
            math.log2(1.2),
        )

    def test_shared_local_timing_becomes_common_not_relative_rubato(self) -> None:
        tempo = extract_piece_tempo_features(
            [
                self._input("first.mid", [0.4, 0.5, 0.6]),
                self._input("second.mid", [0.4, 0.5, 0.6]),
                self._input("third.mid", [0.4, 0.5, 0.6]),
            ]
        )

        rubato = extract_piece_rubato_features(tempo)["first.mid"]

        self.assertGreater(rubato.absolute_rubato_amount, 0.0)
        self.assertGreater(rubato.common_rubato_sequence[0], 0.0)
        self.assertLess(rubato.common_rubato_sequence[2], 0.0)
        for value in rubato.relative_rubato_sequence:
            self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(rubato.relative_rubato_amount, 0.0)

    def test_constant_relative_speed_has_no_rubato(self) -> None:
        tempo = extract_piece_tempo_features(
            [
                self._input("fast.mid", [0.4, 0.4, 0.4]),
                self._input("middle.mid", [0.5, 0.5, 0.5]),
                self._input("slow.mid", [0.6, 0.6, 0.6]),
            ]
        )

        rubato = extract_piece_rubato_features(tempo)["fast.mid"]

        for value in rubato.absolute_rubato_sequence:
            self.assertAlmostEqual(value, 0.0)
        for value in rubato.relative_rubato_sequence:
            self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(rubato.absolute_rubato_amount, 0.0)
        self.assertAlmostEqual(rubato.relative_rubato_amount, 0.0)

    def test_preserves_br_mask(self) -> None:
        tempo = extract_piece_tempo_features(
            [
                TempoInput(
                    performance_key="first.mid",
                    score_beats=[0.0, 0.5, 1.0, 1.5],
                    performance_beats=[0.0, 0.5, 1.0, 1.5],
                    score_beat_types=["db", "b", "bR", "db"],
                    performance_beat_types=["db", "b", "bR", "db"],
                ),
                TempoInput(
                    performance_key="second.mid",
                    score_beats=[0.0, 0.5, 1.0, 1.5],
                    performance_beats=[0.0, 0.6, 1.2, 1.8],
                    score_beat_types=["db", "b", "bR", "db"],
                    performance_beat_types=["db", "b", "bR", "db"],
                ),
            ]
        )

        rubato = extract_piece_rubato_features(tempo)["first.mid"]

        self.assertIsNotNone(rubato.absolute_rubato_sequence[0])
        self.assertEqual(rubato.absolute_rubato_sequence[1:], [None, None])
        self.assertEqual(rubato.relative_rubato_sequence[1:], [None, None])

    def test_preserves_suspicious_mask(self) -> None:
        tempo = extract_piece_tempo_features(
            [
                self._input("first.mid", [0.5, 0.5, 0.5]),
                self._input("second.mid", [0.6, 0.5, 0.5]),
                self._input("third.mid", [0.55, 0.5, 0.5]),
                self._input("extreme.mid", [0.0005, 0.5, 0.5]),
            ]
        )

        rubato = extract_piece_rubato_features(tempo)["extreme.mid"]

        self.assertEqual(tempo["extreme.mid"].intervals[0].status, "suspicious")
        self.assertIsNone(rubato.absolute_rubato_sequence[0])
        self.assertIsNone(rubato.relative_rubato_sequence[0])

    def test_requires_multiple_matching_tempo_features(self) -> None:
        tempo = extract_piece_tempo_features(
            [
                self._input("first.mid", [0.5, 0.5, 0.5]),
                self._input("second.mid", [0.6, 0.6, 0.6]),
            ]
        )

        with self.assertRaisesRegex(ValueError, "At least two tempo features"):
            extract_piece_rubato_features({"first.mid": tempo["first.mid"]})

        with self.assertRaisesRegex(ValueError, "key mismatch"):
            extract_piece_rubato_features(
                {
                    "wrong.mid": tempo["first.mid"],
                    "second.mid": tempo["second.mid"],
                }
            )


if __name__ == "__main__":
    unittest.main()
