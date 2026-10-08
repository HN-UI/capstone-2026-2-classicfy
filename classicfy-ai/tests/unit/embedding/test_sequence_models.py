"""Safety and learning checks for both alternative masked sequence encoders."""

from pathlib import Path
import tempfile
import unittest

import torch

from embedding.model import ModelConfig, TemporalAutoencoder, error_totals, feature_loss
from embedding.sequence_models import (SequenceAutoencoder, SequenceConfig,
                                       load_sequence_model, save_sequence_model)


class SequenceModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def models(self):
        config = ModelConfig(window_size=8, hidden_size=8, embedding_size=4, blocks=2, dropout=0)
        for architecture in ("bilstm", "transformer"):
            torch.manual_seed(42)
            yield architecture, SequenceAutoencoder(config, SequenceConfig(architecture, heads=2, feedforward_size=16))

    def test_no_hidden_or_missing_truth_leak_with_internal_gaps_and_padding(self):
        for name, model in self.models():
            with self.subTest(name=name):
                model.eval()
                values = torch.randn(2, 7, 8)
                valid = torch.ones_like(values, dtype=torch.bool)
                valid[:, :, [2, 7]] = False
                hidden = torch.zeros_like(valid)
                hidden[:, :, 4:6] = True
                original = model(values, valid, hidden)
                changed = values.clone()
                changed[hidden | ~valid] = float("nan")
                after = model(changed, valid, hidden)
                self.assertEqual(original[0].shape, values.shape)
                self.assertEqual(original[1].shape, (2, 4))
                for before, later in zip(original, after):
                    self.assertTrue(torch.isfinite(later).all())
                    torch.testing.assert_close(before, later, rtol=0, atol=0)

    def test_shared_heads_match_seeded_cnn(self):
        for name, model in self.models():
            torch.manual_seed(42)
            cnn = TemporalAutoencoder(model.config)
            for part in ("bottleneck", "decoder"):
                for expected, actual in zip(getattr(cnn, part).parameters(), getattr(model, part).parameters()):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_bottleneck_and_encoder_learn_and_reload_exactly(self):
        for name, model in self.models():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                values = torch.randn(3, 7, 8)
                valid = torch.ones_like(values, dtype=torch.bool)
                hidden = torch.zeros_like(valid)
                hidden[:, :, 3:5] = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
                before = model.bottleneck[1].weight.detach().clone()
                for _ in range(12):
                    optimizer.zero_grad()
                    prediction, embedding = model(values, valid, hidden)
                    feature_loss(*error_totals(prediction, values, hidden)).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters()))
                    optimizer.step()
                self.assertFalse(torch.equal(before, model.bottleneck[1].weight))
                model.eval()
                path = Path(tmp) / "model.pt"
                save_sequence_model(path, model, epoch=12)
                loaded, checkpoint = load_sequence_model(path)
                self.assertEqual(checkpoint["epoch"], 12)
                for left, right in zip(model(values, valid, hidden), loaded(values, valid, hidden)):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_invalid_masks_and_empty_windows_rejected(self):
        for name, model in self.models():
            x = torch.zeros(1, 7, 8)
            valid = torch.zeros_like(x, dtype=torch.bool)
            with self.assertRaises(ValueError):
                model(x, valid)
            with self.assertRaises(ValueError):
                model(x, valid, ~valid)


if __name__ == "__main__":
    unittest.main()
