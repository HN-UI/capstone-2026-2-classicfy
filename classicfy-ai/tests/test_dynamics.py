import unittest

import numpy as np

from preprocessing import MidiData, NoteEvent
from preprocessing.features.dynamics import (
    MAX_VELOCITY,
    DynamicsFeature,
    extract_dynamics,
    summarize_dynamics,
)


def make_performance(notes: list[tuple[float, int]]) -> MidiData:
    events = [
        NoteEvent(pitch=60, start=start, end=start + 0.1, velocity=velocity,
                  instrument_idx=0, program=0)
        for start, velocity in notes
    ]
    return MidiData(notes=events, pedals=[], duration=10.0)


class ExtractDynamicsTest(unittest.TestCase):
    def test_window_value_is_mean_velocity_of_onsets(self) -> None:
        performance = make_performance([(0.0, 40), (0.5, 80), (1.0, 100)])

        feature = extract_dynamics(performance, [0.0, 1.0, 2.0])

        self.assertEqual(len(feature.values), 2)
        self.assertAlmostEqual(feature.values[0], 60 / MAX_VELOCITY)
        self.assertAlmostEqual(feature.values[1], 100 / MAX_VELOCITY)
        self.assertEqual(feature.onset_counts.tolist(), [2, 1])

    def test_window_without_onset_is_masked_and_nan(self) -> None:
        performance = make_performance([(0.0, 64), (2.5, 64)])

        feature = extract_dynamics(performance, [0.0, 1.0, 2.0, 3.0])

        self.assertEqual(feature.mask.tolist(), [True, False, True])
        self.assertTrue(np.isnan(feature.values[1]))
        self.assertEqual(feature.onset_counts.tolist(), [1, 0, 1])

    def test_notes_outside_beat_range_are_ignored(self) -> None:
        performance = make_performance([(-1.0, 10), (0.5, 64), (5.0, 127)])

        feature = extract_dynamics(performance, [0.0, 1.0])

        self.assertEqual(feature.onset_counts.tolist(), [1])
        self.assertAlmostEqual(feature.values[0], 64 / MAX_VELOCITY)

    def test_performance_without_notes_gives_all_masked_windows(self) -> None:
        feature = extract_dynamics(make_performance([]), [0.0, 1.0, 2.0])

        self.assertEqual(feature.mask.tolist(), [False, False])

    def test_values_stay_in_unit_range(self) -> None:
        performance = make_performance([(0.0, 1), (1.0, 127)])

        feature = extract_dynamics(performance, [0.0, 1.0, 2.0])

        self.assertTrue(np.all(feature.values >= 0))
        self.assertTrue(np.all(feature.values <= 1))


class SummarizeDynamicsTest(unittest.TestCase):
    def test_summary_ignores_masked_windows(self) -> None:
        feature = DynamicsFeature(
            values=np.array([0.2, np.nan, 0.6]),
            mask=np.array([True, False, True]),
            onset_counts=np.array([1, 0, 1]),
        )

        summary = summarize_dynamics(feature)

        self.assertAlmostEqual(summary["dynamics_mean"], 0.4)
        # 5·95 백분위는 0.22와 0.58이다.
        self.assertAlmostEqual(summary["dynamics_range"], 0.36)

    def test_summary_of_all_masked_feature_is_nan(self) -> None:
        feature = DynamicsFeature(
            values=np.array([np.nan]), mask=np.array([False]), onset_counts=np.array([0])
        )

        summary = summarize_dynamics(feature)

        self.assertTrue(np.isnan(summary["dynamics_mean"]))
        self.assertTrue(np.isnan(summary["dynamics_range"]))


if __name__ == "__main__":
    unittest.main()
