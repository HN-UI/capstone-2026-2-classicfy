"""The ablation must isolate encoder width and supervise only hidden pairs."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from train_temporal_ablation import EncoderExperiment, delta_totals, training_loss
from embedding.model import ModelConfig, TemporalAutoencoder, error_totals, feature_loss


class TemporalAblationTest(unittest.TestCase):
    def test_baseline_is_identical_and_wider_encoder_keeps_same_decoder(self):
        config=ModelConfig(window_size=8)
        torch.manual_seed(10)
        original=TemporalAutoencoder(config)
        torch.manual_seed(10)
        baseline=EncoderExperiment(config,64)
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,baseline.state_dict()[name],rtol=0,atol=0)
        torch.manual_seed(10)
        wide=EncoderExperiment(config,128)
        for name,value in baseline.decoder.state_dict().items():
            torch.testing.assert_close(value,wide.decoder.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(baseline.bottleneck[1].out_features,64)
        self.assertEqual(wide.bottleneck[1].out_features,64)
        self.assertGreater(sum(p.numel() for p in wide.parameters()),sum(p.numel() for p in baseline.parameters()))

    def test_delta_loss_ignores_visible_boundaries_and_disconnected_gaps(self):
        target=torch.tensor([[[0.,1,3,99,20,22]]]).repeat(1,7,1)
        predicted=torch.tensor([[[999.,2,3,-999,10,12]]]).repeat(1,7,1)
        hidden=torch.tensor([[[False,True,True,False,True,True]]]).repeat(1,7,1)
        sums,counts=delta_totals(predicted,target,hidden)
        torch.testing.assert_close(counts,torch.full((7,),2,dtype=torch.int64))
        # First internal pair: (3-2)-(3-1) = -1; second pair is exact.
        torch.testing.assert_close(sums,torch.ones(7))
        changed=target.clone()
        changed[~hidden]=100000
        torch.testing.assert_close(delta_totals(predicted,changed,hidden)[0],sums)

    def test_objective_is_exact_baseline_at_zero_weight_and_has_finite_gradients(self):
        prediction=torch.randn(2,7,8,requires_grad=True)
        target=torch.randn_like(prediction)
        hidden=torch.zeros_like(prediction,dtype=torch.bool)
        hidden[:,:,2:5]=True
        reference=feature_loss(*error_totals(prediction,target,hidden))
        torch.testing.assert_close(training_loss(prediction,target,hidden,0),reference,rtol=0,atol=0)
        loss=training_loss(prediction,target,hidden,.5)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[~hidden]==0).all())

    def test_single_hidden_beat_does_not_create_fake_delta_targets(self):
        prediction=torch.randn(1,7,8,requires_grad=True)
        target=torch.randn_like(prediction)
        hidden=torch.zeros_like(prediction,dtype=torch.bool)
        hidden[:,:,3]=True
        self.assertEqual(int(delta_totals(prediction,target,hidden)[1].sum()),0)
        loss=training_loss(prediction,target,hidden,.5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__=="__main__":
    unittest.main()
