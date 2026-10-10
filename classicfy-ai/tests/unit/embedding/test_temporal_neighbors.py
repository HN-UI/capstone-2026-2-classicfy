"""Cross-work observations must preserve gaps and avoid trivial/tie-biased controls."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from embedding.data import PerformanceSequence
from observe_temporal_neighbors import block_distances, compare_candidate, select_fixed_queries, step_summary


class TemporalNeighborsTest(unittest.TestCase):
    def test_step_summary_does_not_bridge_missing_beats(self):
        values = np.tile([1., 3., 999., 7., 10.], (7, 1))
        valid = np.tile([True, True, False, True, True], (7, 1))
        sequence = PerformanceSequence("key", "work", "grid", values, valid)
        rms, support = step_summary(sequence)
        np.testing.assert_allclose(rms, np.sqrt((2**2 + 3**2) / 2))
        np.testing.assert_array_equal(support, [2] * 7)

    def test_three_pedal_channels_do_not_outweigh_one_feature(self):
        values = np.zeros((3, 7))
        values[1, 0] = 1
        values[2, 4:] = 1
        distance = block_distances(values)
        self.assertAlmostEqual(distance[0, 1], np.sqrt(1 / 5))
        self.assertAlmostEqual(distance[0, 2], distance[0, 1])

    def test_peer_comparison_excludes_candidate_and_gives_ties_half_credit(self):
        distances = np.zeros((5, 5))
        distances[0] = [0, .5, .5, 1., 0]
        result = compare_candidate(distances, ["A", "B", "B", "B", "A"], 0, 1)
        self.assertEqual(result["candidate_work_peers"], 2)
        self.assertEqual(result["cross_work_candidates"], 3)
        self.assertEqual(result["candidate_work_peer_mean_distance"], .75)
        self.assertEqual(result["closer_than_peer_share"], .75)
        with self.assertRaises(ValueError):
            compare_candidate(distances, ["A", "B", "B", "B", "A"], 0, 4)

    def test_examples_depend_on_fixed_work_and_key_order(self):
        rows = [{"key": f"{work}/{key}", "piece": work, "split": "test"}
                for work in ("Z", "A", "M", "B") for key in ("3", "1", "2")]
        rows.append({"key": "train/1", "piece": "0", "split": "train"})
        chosen = select_fixed_queries(rows)
        self.assertEqual([rows[i]["key"] for i in chosen], ["A/2", "M/2", "Z/2"])


if __name__ == "__main__":
    unittest.main()
