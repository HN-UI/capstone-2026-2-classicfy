import math
import unittest

import numpy as np

from features import (
    TempoInput,
    extract_dynamics,
    extract_pedaling,
    extract_piece_rubato_features,
    extract_piece_tempo_features,
    summarize_dynamics,
    summarize_pedaling,
    summarize_rubato,
    summarize_tempo,
)
from preprocessing import MidiData, NoteEvent, PedalEvent


class FeatureCharacterizationTest(unittest.TestCase):
    def test_representative_feature_values_and_summaries(self) -> None:
        performance = MidiData(
            notes=[
                NoteEvent(60, 0.2, 0.4, 64, 0, 0),
                NoteEvent(64, 0.7, 0.9, 127, 0, 0),
                NoteEvent(67, 1.2, 1.4, 32, 0, 0),
            ],
            pedals=[
                PedalEvent(0.0, 127, 0),
                PedalEvent(0.5, 0, 0),
                PedalEvent(1.5, 127, 0),
                PedalEvent(2.5, 0, 0),
            ],
            duration=3.0,
        )
        beats = [0.0, 1.0, 2.0, 3.0]

        dynamics = extract_dynamics(performance, beats)
        pedaling = extract_pedaling(performance, beats)

        np.testing.assert_allclose(
            dynamics.sequence.values,
            [95.5 / 127, 32 / 127, np.nan],
            equal_nan=True,
        )
        self.assertEqual(dynamics.sequence.mask.tolist(), [True, True, False])
        self.assertEqual(dynamics.onset_counts.tolist(), [2, 1, 0])
        np.testing.assert_allclose(pedaling.depth.values, [0.5, 0.5, 0.5])
        np.testing.assert_allclose(pedaling.down_ratio.values, [0.5, 0.5, 0.5])
        self.assertEqual(pedaling.changes.values.tolist(), [2, 1, 1])

        dynamics_summary = summarize_dynamics(dynamics)
        pedaling_summary = summarize_pedaling(pedaling)
        self.assertAlmostEqual(
            dynamics_summary["dynamics_mean"], (95.5 / 127 + 32 / 127) / 2
        )
        self.assertAlmostEqual(pedaling_summary["pedal_depth_mean"], 0.5)
        self.assertAlmostEqual(pedaling_summary["pedal_usage"], 0.5)
        self.assertAlmostEqual(pedaling_summary["pedal_change_rate"], 4 / 3)

        beat_types = ["db", "b", "b", "db"]
        tempo = extract_piece_tempo_features(
            [
                TempoInput(
                    "expressive.mid",
                    beats,
                    [0.0, 0.5, 1.5, 3.0],
                    beat_types,
                    beat_types,
                ),
                TempoInput(
                    "reference.mid",
                    beats,
                    beats,
                    beat_types,
                    beat_types,
                ),
            ]
        )
        expressive_tempo = tempo["expressive.mid"]
        expected_score_relative = [1.0, 0.0, math.log2(2 / 3)]
        expected_common = np.array(expected_score_relative) / 2
        np.testing.assert_allclose(
            expressive_tempo.common_tempo_sequence.values, expected_common
        )
        np.testing.assert_allclose(
            expressive_tempo.individual_tempo_sequence.values, expected_common
        )
        self.assertEqual(
            summarize_tempo(expressive_tempo),
            {
                "overall_score_relative_tempo": 0.0,
                "overall_individual_tempo": 0.0,
            },
        )

        expressive_rubato = extract_piece_rubato_features(tempo)["expressive.mid"]
        np.testing.assert_allclose(
            expressive_rubato.absolute_rubato_sequence.values,
            expected_score_relative,
        )
        np.testing.assert_allclose(
            expressive_rubato.relative_rubato_sequence.values,
            expected_common,
        )
        rubato_summary = summarize_rubato(expressive_rubato)
        self.assertAlmostEqual(
            rubato_summary["absolute_rubato_amount"], abs(math.log2(2 / 3))
        )
        self.assertAlmostEqual(
            rubato_summary["relative_rubato_amount"], abs(math.log2(2 / 3)) / 2
        )


if __name__ == "__main__":
    unittest.main()
