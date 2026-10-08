"""Beat topology, leakage, masking, bottleneck learning and saved-model inference."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from embedding.data import (PerformanceSequence, WindowDataset, make_hidden_mask,
                            restore_sequences, split_by_piece)
from embedding.model import ModelConfig, TemporalAutoencoder, error_totals, feature_loss
from embedding.training import (TrainingConfig, cross_work_neighbors, extract_embeddings,
                                fixed_masks, load_model, train_model)


def sequence(key, piece, count=12):
    curve = np.linspace(-1, 1, count, dtype=np.float32)
    values = np.stack([curve * (1 + i / 7) for i in range(7)])
    return PerformanceSequence(key, piece, "grid", values, np.ones_like(values, bool))


class TemporalEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_restore_gaps_trailing_missing_and_work_variants(self):
        groups = [{"piece": "work::no_repeat", "grid": "g", "keys": ["a"],
                   "shared_indices": np.array([0, 2, 4]), "values": np.ones((1, 7, 3))}]
        audits = [{"piece": "work::no_repeat", "grid": "g", "total_intervals": 7, "accepted": True}]
        restored = restore_sequences(groups, audits)[0]
        self.assertEqual(restored.piece, "work")
        self.assertEqual(restored.values.shape, (7, 7))
        np.testing.assert_array_equal(restored.valid[0], [True, False, True, False, True, False, False])
        np.testing.assert_array_equal(restored.values[0], [1, 0, 1, 0, 1, 0, 0])

    def test_split_is_work_disjoint_seeded_and_order_independent(self):
        sequences = [sequence(str(i), f"work{i // 2}") for i in range(12)]
        a = split_by_piece(sequences, seed=12)
        self.assertEqual(a, split_by_piece(list(reversed(sequences)), seed=12))
        self.assertEqual(set(a.values()), {"train", "validation", "test"})
        for i in range(0, len(sequences), 2):
            self.assertEqual(a[sequences[i].piece], a[sequences[i + 1].piece])

    def test_windows_preserve_missing_values_padding_and_tail(self):
        sample = sequence("a", "work", count=6)
        sample.valid[:, 2] = False
        sample.values[:, 2] = np.nan
        dataset = WindowDataset([sample], window_size=8, stride=4)
        values, valid, _ = dataset[0]
        self.assertTrue(torch.isfinite(values).all())
        self.assertFalse(valid[:, 2].any())
        self.assertFalse(valid[:, 6:].any())
        self.assertEqual(len(dataset), 1)
        long = WindowDataset([sequence("b", "work", 19)], window_size=8, stride=4)
        self.assertEqual([start for _, start in long.items], [0, 4, 8, 11])
        self.assertEqual(WindowDataset([sequence("c", "work", 3)], window_size=8, stride=4).skipped_keys, ["c"])

    def test_hidden_mask_has_no_invalid_targets_and_keeps_context(self):
        valid = torch.ones(2, 7, 12, dtype=torch.bool)
        valid[:, :, [2, 3, 11]] = False
        original = valid.clone()
        a = make_hidden_mask(valid, fraction=.4, generator=torch.Generator().manual_seed(2))
        b = make_hidden_mask(valid, fraction=.4, generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(valid, original)
        self.assertFalse((a & ~valid).any())
        self.assertTrue(((valid & ~a).sum(dim=(1, 2)) > 0).all())
        self.assertTrue((a[:, 0].sum(dim=1) == 4).all())

    def test_hidden_truth_cannot_leak_to_prediction_or_embedding(self):
        torch.manual_seed(1)
        model = TemporalAutoencoder(ModelConfig(window_size=8, hidden_size=8, embedding_size=4, blocks=2, dropout=0))
        model.eval()
        values = torch.randn(2, 7, 8)
        valid = torch.ones_like(values, dtype=torch.bool)
        hidden = make_hidden_mask(valid, generator=torch.Generator().manual_seed(3))
        before = model(values, valid, hidden)
        changed = values.clone()
        changed[hidden] = 10000
        after = model(changed, valid, hidden)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b)

    def test_five_feature_blocks_have_equal_loss_weight(self):
        target = torch.zeros(2, 7, 8)
        prediction = target.clone()
        prediction[0, 0] = 2
        prediction[1, 4:] = 2
        losses = [feature_loss(*error_totals(prediction[i:i + 1], target[i:i + 1],
                          torch.ones_like(target[i:i + 1], dtype=torch.bool))) for i in range(2)]
        torch.testing.assert_close(losses[0], losses[1])
        torch.testing.assert_close(losses[0], torch.tensor(4 / 5))
        with self.assertRaises(ValueError):
            error_totals(prediction, target, torch.zeros_like(target, dtype=torch.bool))

    def test_optimizer_updates_encoder_and_improves_fixed_mask_reconstruction(self):
        torch.manual_seed(7)
        model = TemporalAutoencoder(ModelConfig(8, 8, 4, 2, 0))
        values = torch.stack([torch.from_numpy(sequence("a", "work", 8).values)] * 4)
        valid = torch.ones_like(values, dtype=torch.bool)
        hidden = make_hidden_mask(valid, generator=torch.Generator().manual_seed(7))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        initial_weights = model.input_projection.weight.detach().clone()
        initial = float(feature_loss(*error_totals(model(values, valid, hidden)[0], values, hidden)).detach())
        for _ in range(20):
            optimizer.zero_grad()
            loss = feature_loss(*error_totals(model(values, valid, hidden)[0], values, hidden))
            loss.backward()
            optimizer.step()
        final = float(feature_loss(*error_totals(model(values, valid, hidden)[0], values, hidden)).detach())
        self.assertLess(final, initial)
        self.assertFalse(torch.equal(initial_weights, model.input_projection.weight))

    def test_training_saves_reloadable_best_and_unmasked_embeddings(self):
        train = WindowDataset([sequence("a", "train")], window_size=8, stride=4)
        validation = WindowDataset([sequence("b", "validation")], window_size=8, stride=4)
        config = TrainingConfig(epochs=2, batch_size=2, seed=9)
        torch.testing.assert_close(fixed_masks(validation, config), fixed_masks(validation, config))
        with tempfile.TemporaryDirectory() as temporary:
            model, result = train_model(train, validation, temporary, model_config=ModelConfig(8, 8, 4, 2, 0), config=config)
            restored, checkpoint = load_model(Path(temporary) / "best.pt")
            self.assertEqual(checkpoint["epoch"], result["best_epoch"])
            first = extract_embeddings(model, train)[1]
            second = extract_embeddings(restored, train)[1]
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (1, 8))
            with (Path(temporary) / "history.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            self.assertTrue((Path(temporary) / "reconstruction.npz").exists())
        with self.assertRaises(ValueError):
            train_model(train, train, "/tmp/unused-temporal", model_config=ModelConfig(8, 8, 4, 2, 0), config=config)

    def test_neighbors_exclude_all_same_work_and_reject_zero_vectors(self):
        rows = [{"key": "a", "piece": "one"}, {"key": "b", "piece": "one"}, {"key": "c", "piece": "two"}]
        vectors = np.array([[1., 0], [1., .1], [1., .2]])
        neighbors = cross_work_neighbors(rows, vectors)
        self.assertEqual(next(row["candidate"] for row in neighbors if row["query"] == "a"), "c")
        self.assertTrue(all(row["query_piece"] != row["candidate_piece"] for row in neighbors))
        with self.assertRaises(ValueError):
            cross_work_neighbors(rows, np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
