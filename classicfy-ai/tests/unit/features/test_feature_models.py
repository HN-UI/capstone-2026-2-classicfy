import unittest

import numpy as np

from features import BeatSequence


class BeatSequenceTest(unittest.TestCase):
    def test_copies_values_and_mask_as_readonly_arrays(self) -> None:
        values = np.array([0.2, 0.4])
        mask = np.array([True, False])

        sequence = BeatSequence(values=values, mask=mask)
        values[0] = 1.0
        mask[0] = False

        self.assertEqual(sequence.values.tolist(), [0.2, 0.4])
        self.assertEqual(sequence.mask.tolist(), [True, False])
        self.assertFalse(sequence.values.flags.writeable)
        self.assertFalse(sequence.mask.flags.writeable)
        with self.assertRaises(ValueError):
            sequence.values[0] = 1.0

    def test_rejects_mismatched_or_multidimensional_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            BeatSequence(values=np.zeros(2), mask=np.ones(1, dtype=bool))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            BeatSequence(values=np.zeros((1, 2)), mask=np.ones(2, dtype=bool))

    def test_converts_optional_values_to_nan_and_mask(self) -> None:
        sequence = BeatSequence.from_optional([0.1, None, -0.2])

        np.testing.assert_allclose(sequence.values, [0.1, np.nan, -0.2], equal_nan=True)
        self.assertEqual(sequence.mask.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
