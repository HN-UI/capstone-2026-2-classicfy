"""Optional FP32 inference of the unchanged public GlueNote network graph."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import numpy as np
import torch


class OpenVINOConfidence(torch.nn.Module):
    def __init__(self, original, ir, device):
        super().__init__()
        import openvino as ov
        manifest=json.loads(ir.with_suffix('.json').read_text(encoding='utf-8'))
        checkpoint=importlib.metadata.distribution('parangonar').locate_file('parangonar/assets/thegluenote_small_checkpoint.pt')
        if manifest['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('Converted graph belongs to a different checkpoint')
        self.position_number=original.position_number
        self.device=torch.device('cpu')
        self.compiled=ov.Core().compile_model(str(ir),device,
            {'INFERENCE_PRECISION_HINT':'f32','PERFORMANCE_HINT':'LATENCY'})
    def forward(self,tokens,return_confidence_matrix=True):
        if not return_confidence_matrix: raise ValueError('Confidence-only inference wrapper')
        result=self.compiled([tokens.detach().cpu().numpy()])[self.compiled.output(0)]
        return torch.from_numpy(np.array(result,copy=True))


def accelerate(engine,backend):
    ir=Path(__file__).resolve().parents[4]/'ASAP_dataset/alignment_models/thegluenote_small_f32.xml'
    engine.model=OpenVINOConfidence(engine.model,ir,'GPU' if backend=='openvino_gpu' else 'CPU')
    return engine
