import unittest

import numpy as np

from preprocessing import MidiData, PedalEvent
from preprocessing.features.pedaling import (
    MAX_PEDAL_VALUE,
    PedalingFeature,
    extract_pedaling,
    summarize_pedaling,
)


def make_performance(pedals: list[tuple[float, int]]) -> MidiData:
    events = [PedalEvent(time=time, value=value, instrument_idx=0) for time, value in pedals]
    return MidiData(notes=[], pedals=events, duration=10.0)


class ExtractPedalingTest(unittest.TestCase):
    def test_depth_and_down_ratio_are_time_weighted(self) -> None:
        performance = make_performance([(0.5, 127), (1.5, 0)])

        feature = extract_pedaling(performance, [0.0, 1.0, 2.0])

        np.testing.assert_allclose(feature.depth, [0.5, 0.5])
        np.testing.assert_allclose(feature.down_ratio, [0.5, 0.5])
        self.assertEqual(feature.changes.tolist(), [1, 1])

    def test_pedal_is_up_before_first_event(self) -> None:
        performance = make_performance([(1.0, 127)])

        feature = extract_pedaling(performance, [0.0, 1.0, 2.0])

        np.testing.assert_allclose(feature.depth, [0.0, 1.0])
        self.assertEqual(feature.changes.tolist(), [0, 1])

    def test_pedal_held_from_before_first_beat_counts_inside_window(self) -> None:
        performance = make_performance([(-1.0, 127), (5.0, 0)])

        feature = extract_pedaling(performance, [0.0, 1.0])

        np.testing.assert_allclose(feature.depth, [1.0])
        np.testing.assert_allclose(feature.down_ratio, [1.0])
        self.assertEqual(feature.changes.tolist(), [0])

    def test_half_pedal_keeps_depth_and_uses_threshold_for_down(self) -> None:
        for value, expected_down in ((63, 0.0), (64, 1.0)):
            with self.subTest(value=value):
                feature = extract_pedaling(make_performance([(0.0, value)]), [0.0, 1.0])

                np.testing.assert_allclose(feature.depth, [value / MAX_PEDAL_VALUE])
                np.testing.assert_allclose(feature.down_ratio, [expected_down])

    def test_changes_count_only_threshold_crossings(self) -> None:
        performance = make_performance([(0.2, 100), (0.4, 127), (0.6, 90), (0.8, 10)])

        feature = extract_pedaling(performance, [0.0, 1.0])

        self.assertEqual(feature.changes.tolist(), [2])

    def test_performance_without_pedal_events_is_all_zero_and_valid(self) -> None:
        feature = extract_pedaling(make_performance([]), [0.0, 1.0, 2.0])

        np.testing.assert_allclose(feature.depth, [0.0, 0.0])
        self.assertEqual(feature.changes.tolist(), [0, 0])
        self.assertEqual(feature.mask.tolist(), [True, True])

    def test_zero_width_window_is_masked(self) -> None:
        performance = make_performance([(0.0, 127)])

        feature = extract_pedaling(performance, [0.0, 1.0, 1.0, 2.0])

        self.assertEqual(feature.mask.tolist(), [True, False, True])
        np.testing.assert_allclose(feature.depth[[0, 2]], [1.0, 1.0])

    def test_matches_dense_sampling_on_random_pedal_signal(self) -> None:
        rng = np.random.default_rng(0)
        times = np.sort(rng.uniform(-1.0, 11.0, size=60))
        values = rng.integers(0, MAX_PEDAL_VALUE + 1, size=60)
        beats = np.sort(rng.uniform(0.0, 10.0, size=15))
        performance = make_performance(list(zip(times.tolist(), values.tolist())))

        feature = extract_pedaling(performance, beats)

        # 1e-4초 간격으로 신호를 직접 샘플링해 구간 평균을 구한 값과 비교한다.
        grid = np.arange(beats[0], beats[-1], 1e-4)
        last = np.searchsorted(times, grid, side="right") - 1
        signal = np.where(last >= 0, values[np.maximum(last, 0)], 0) / MAX_PEDAL_VALUE
        window = np.searchsorted(beats, grid, side="right") - 1
        expected = np.array([signal[window == i].mean() for i in range(len(beats) - 1)])
        np.testing.assert_allclose(feature.depth, expected, atol=2e-3)


class SummarizePedalingTest(unittest.TestCase):
    def test_summary_averages_over_valid_windows(self) -> None:
        feature = PedalingFeature(
            depth=np.array([0.2, 0.9, 0.4]),
            down_ratio=np.array([0.0, 1.0, 0.5]),
            changes=np.array([0, 2, 1]),
            mask=np.array([True, False, True]),
        )

        summary = summarize_pedaling(feature)

        self.assertAlmostEqual(summary["pedal_depth_mean"], 0.3)
        self.assertAlmostEqual(summary["pedal_usage"], 0.25)
        self.assertAlmostEqual(summary["pedal_change_rate"], 0.5)

    def test_summary_without_valid_windows_is_nan(self) -> None:
        feature = PedalingFeature(
            depth=np.zeros(1), down_ratio=np.zeros(1),
            changes=np.zeros(1, dtype=int), mask=np.array([False]),
        )

        summary = summarize_pedaling(feature)

        self.assertTrue(all(np.isnan(value) for value in summary.values()))


if __name__ == "__main__":
    unittest.main()
