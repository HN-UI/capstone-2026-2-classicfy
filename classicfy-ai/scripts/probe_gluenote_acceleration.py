"""Verify FLOAT32 graph conversion; does not change the active benchmark."""
from pathlib import Path
import sys
import json
import time
import warnings
import hashlib
import importlib.metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import openvino as ov
from preprocessing.research_alignment import matcher


class Confidence(torch.nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,tokens):
        return self.model(tokens,return_confidence_matrix=True)


def main():
    torch.set_num_threads(2)
    engine=matcher('gluenote'); model=engine.model
    wrapper=Confidence(model).eval()
    torch.manual_seed(43)
    tokens=torch.randint(0,model.token_number,(1,model.position_number*2*4),dtype=torch.int64)
    root=Path('../ASAP_dataset/alignment_models').resolve(); root.mkdir(parents=True,exist_ok=True)
    target=root/'thegluenote_small_f32.xml'
    if not target.exists():
        with torch.inference_mode(),warnings.catch_warnings():
            warnings.simplefilter('ignore')
            converted=ov.convert_model(wrapper,example_input=tokens)
        ov.save_model(converted,str(target),compress_to_fp16=False)
    core=ov.Core(); rows=[]
    checkpoint=importlib.metadata.distribution('parangonar').locate_file('parangonar/assets/thegluenote_small_checkpoint.pt')
    target.with_suffix('.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        precision='f32',compress_to_fp16=False,openvino=ov.__version__),indent=2),encoding='utf-8')
    for device in ['CPU','GPU']:
        try:
            compiled=core.compile_model(str(target),device,{'INFERENCE_PRECISION_HINT':'f32','PERFORMANCE_HINT':'LATENCY'})
            for trial in range(4):
                x=torch.randint(0,model.token_number,tokens.shape,dtype=torch.int64)
                with torch.inference_mode():
                    start=time.perf_counter(); expected=wrapper(x).cpu().numpy(); original_seconds=time.perf_counter()-start
                start=time.perf_counter(); actual=compiled([x.numpy()])[compiled.output(0)]; seconds=time.perf_counter()-start
                rows.append(dict(device=device,trial=trial,torch_seconds=original_seconds,converted_seconds=seconds,
                    max_absolute_difference=float(np.max(np.abs(actual-expected))),
                    mean_absolute_difference=float(np.mean(np.abs(actual-expected))),
                    output_shape=list(actual.shape)))
        except Exception as e: rows.append(dict(device=device,error=str(e)))
    out=Path('classicfy-ai/analysis/gluenote_acceleration_probe.json')
    out.write_text(json.dumps(dict(openvino=ov.__version__,devices=core.available_devices,rows=rows),indent=2),encoding='utf-8')
    print(out.read_text(encoding='utf-8'),flush=True)


if __name__=='__main__': main()
