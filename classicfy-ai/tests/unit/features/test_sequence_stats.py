import unittest

from features.sequence_stats import median_sequence_with_support


class MedianSequenceWithSupportTest(unittest.TestCase):
    def test_calculates_position_medians_and_support(self) -> None:
        common, support = median_sequence_with_support(
            [
                [1.0, None, 3.0],
                [2.0, 4.0, None],
                [100.0, 6.0, 9.0],
            ]
        )

        self.assertEqual(common, [2.0, 5.0, 6.0])
        self.assertEqual(support, [3, 2, 2])

    def test_respects_minimum_support(self) -> None:
        common, support = median_sequence_with_support(
            [[1.0, None], [3.0, 2.0]], minimum_support=2
        )

        self.assertEqual(common, [2.0, None])
        self.assertEqual(support, [2, 1])

    def test_rejects_missing_or_mismatched_sequences(self) -> None:
        with self.assertRaises(ValueError):
            median_sequence_with_support([])
        with self.assertRaises(ValueError):
            median_sequence_with_support([[1.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            median_sequence_with_support([[1.0]], minimum_support=0)


if __name__ == "__main__":
    unittest.main()
