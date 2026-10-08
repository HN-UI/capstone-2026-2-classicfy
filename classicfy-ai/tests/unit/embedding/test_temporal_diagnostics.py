"""Protect gap diagnostics from invalid joins, outcome selection, and bad weights."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyze_temporal_failures import collect_gaps, classify, runs, summarize
from embedding.data import PerformanceSequence, WindowDataset
from embedding.evaluation import baseline_predictions


class TemporalDiagnosticsTest(unittest.TestCase):
    def test_actual_runs_merge_blocks_but_do_not_join_gaps(self):
        self.assertEqual(runs([False, True, True, True, True, True, True, False, True]), [(1, 7), (8, 9)])

    def test_invalid_anchor_is_not_observed_context(self):
        values = np.tile(np.arange(8, dtype=np.float32), (7, 1))
        valid = np.ones_like(values, dtype=bool)
        valid[:, 3] = False
        sequence = PerformanceSequence("example.mid", "work", "grid", values, valid)
        dataset = WindowDataset([sequence], window_size=8, stride=4)
        target, validity = values[None], valid[None]
        hidden = np.zeros_like(validity)
        hidden[:, :, 1:3] = True
        hidden[:, :, 5:7] = True
        predictions = baseline_predictions(target, validity, hidden)
        predictions.pop("zero")
        predictions["cnn"] = target.copy()
        gaps = collect_gaps("train", dataset, (target, validity, hidden, predictions))
        self.assertTrue((~gaps[gaps.start == 1].two_sided).all())
        self.assertTrue(gaps[gaps.start == 5].two_sided.all())
        self.assertTrue((gaps.pairs == 1).all())
        self.assertTrue((gaps.cnn_sse == 0).all())
        self.assertTrue((gaps.cnn_direction_hits == 1).all())

    def test_shape_definitions_are_exclusive_and_handle_flat_pedal(self):
        thresholds = {"range_q25": 0, "step_q75": 1}
        self.assertEqual(classify(0, 0, 1, thresholds), "small")
        self.assertEqual(classify(2, .8, 1, thresholds), "gradual")
        self.assertEqual(classify(3, 2, 1, thresholds), "abrupt")
        self.assertEqual(classify(2, .8, .2, thresholds), "mixed")

    def test_metrics_pool_targets_instead_of_averaging_unequal_gaps(self):
        rows = []
        for targets, sse, pairs in ((1, 9., 0), (3, 3., 2)):
            row = dict(split="train", channel="tempo", piece="piece", key="key",
                       targets=targets, pairs=pairs, moving_pairs=pairs, actual_step_ss=float(pairs))
            for method in ("cnn", "visible_mean", "linear_interpolation"):
                row.update({f"{method}_sse": sse, f"{method}_step_sse": 0.,
                            f"{method}_step_ss": float(pairs), f"{method}_direction_hits": pairs})
            rows.append(row)
        result = summarize(pd.DataFrame(rows), ["split", "channel"])
        self.assertTrue((result.mse == 3.).all())
        self.assertTrue((result.direction_percent == 100.).all())
        self.assertTrue((result.step_amplitude_percent == 100.).all())


if __name__ == "__main__":
    unittest.main()
