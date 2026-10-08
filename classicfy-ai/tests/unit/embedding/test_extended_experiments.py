"""Encoder initialization, paired schedules, patience and exact resume checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from embedding.extended_experiments import build_model, make_schedule, train, load, atomic_save
from embedding.model import ModelConfig, error_totals, feature_loss
from embedding.training import TrainingConfig, loader


def score(model,dataset,masks,config):
    model.eval()
    sums,counts=torch.zeros(7,dtype=torch.float64),torch.zeros(7,dtype=torch.int64)
    with torch.no_grad():
        for x,v,ix in loader(dataset,config):
            errors,support=error_totals(model(x,v,masks[ix])[0],x,masks[ix])
            sums+=errors.double();counts+=support
    return {"value_mse":float(feature_loss(sums,counts)),"delta_mse":0.}


class ExtendedExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_larger_bottlenecks_preserve_encoder_and_output_layer(self):
        config=ModelConfig(window_size=8,hidden_size=8,embedding_size=64,blocks=2)
        models=[build_model(a,42,config) for a in ("bilstm64","bilstm128","bilstm256")]
        x=torch.zeros(2,7,8);v=torch.ones_like(x,dtype=torch.bool)
        for model,dimensions in zip(models,(64,128,256)):
            model.eval()
            self.assertEqual(model(x,v)[1].shape,(2,dimensions))
            self.assertEqual(model.decoder[0].out_features,16)
            for name,p in models[0].named_parameters():
                if name.startswith(("encoder.","input_projection.","decoder.2.")):
                    torch.testing.assert_close(p,dict(model.named_parameters())[name],rtol=0,atol=0)

    def fixture(self,directory):
        torch.manual_seed(1)
        x=torch.randn(4,7,8);v=torch.ones_like(x,dtype=torch.bool)
        v[:,:,2]=False;x[:,:,2]=0
        dataset=TensorDataset(x,v,torch.arange(4))
        masks=torch.zeros_like(v);masks[:,:,4:6]=True
        config=TrainingConfig(epochs=20,batch_size=2,patience=30,seed=31)
        schedule=make_schedule(x,v,config,directory/"schedule.npz")
        self.assertFalse((schedule[1] & ~v.numpy()[schedule[0]]).any())
        return x,v,dataset,masks,config,schedule

    def test_resume_preserves_optimizer_dropout_and_exact_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            x,v,dataset,masks,config,schedule=self.fixture(directory)
            mc=ModelConfig(window_size=8,hidden_size=8,embedding_size=64,blocks=2)
            args=("bilstm64",31,x,v,dataset,masks,schedule,config,mc)
            uninterrupted=train(*args,directory/"full",score_function=score)
            def interrupt(path,value):
                atomic_save(path,value)
                if Path(path).name=="last.pt" and value["epoch"]==3:
                    raise RuntimeError("simulated interruption")
            with patch("embedding.extended_experiments.atomic_save",side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,"simulated"):
                    train(*args,directory/"resume",score_function=score)
            resumed=train(*args,directory/"resume",score_function=score)
            self.assertEqual(uninterrupted["best_validation_mse"],resumed["best_validation_mse"])
            full,_=load(directory/"full/final.pt");later,_=load(directory/"resume/final.pt")
            for a,b in zip(full.parameters(),later.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_minimum_epoch_and_patience_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            x,v,dataset,masks,config,schedule=self.fixture(directory)
            config=replace(config,patience=2)
            def constant(*args):return {"value_mse":1.,"delta_mse":0.}
            result=train("bilstm64",31,x,v,dataset,masks,schedule,config,
                ModelConfig(window_size=8,hidden_size=8,embedding_size=64,blocks=2),directory/"run",score_function=constant)
            self.assertEqual(result["completed_epochs"],20)
            self.assertEqual(result["stop_reason"],"patience")
            self.assertEqual(result["best_epoch"],0)


if __name__=="__main__":unittest.main()
