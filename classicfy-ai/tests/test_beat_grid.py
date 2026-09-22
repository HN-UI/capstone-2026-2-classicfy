import unittest

import numpy as np

from preprocessing.features.beat_grid import as_beat_array, assign_windows


class BeatGridTest(unittest.TestCase):
    def test_assign_windows_uses_half_open_intervals(self) -> None:
        beats = [1.0, 2.0, 4.0]

        windows = assign_windows([1.0, 1.99, 2.0, 3.99], beats)

        self.assertEqual(windows.tolist(), [0, 0, 1, 1])

    def test_assign_windows_excludes_times_outside_first_and_last_beat(self) -> None:
        beats = [1.0, 2.0, 4.0]

        windows = assign_windows([0.5, 4.0, 5.0], beats)

        self.assertEqual(windows.tolist(), [-1, -1, -1])

    def test_assign_windows_skips_zero_width_window(self) -> None:
        windows = assign_windows([2.0], [1.0, 2.0, 2.0, 3.0])

        self.assertEqual(windows.tolist(), [2])

    def test_as_beat_array_rejects_unusable_beats(self) -> None:
        for beats in ([1.0], [2.0, 1.0], [0.0, float("nan")], [[0.0, 1.0]]):
            with self.subTest(beats=beats):
                with self.assertRaises(ValueError):
                    as_beat_array(beats)

    def test_as_beat_array_returns_float_array(self) -> None:
        array = as_beat_array([0, 1, 2])

        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.dtype, float)


if __name__ == "__main__":
    unittest.main()
