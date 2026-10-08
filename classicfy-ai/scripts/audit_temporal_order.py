"""Independently recalculate saved perturbation distances, scores and aggregation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.order_evaluation import make_variants


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ai=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ai/"analysis/temporal_order_evaluation")
    parser.add_argument("--runs",type=Path,default=ai.parent.parent/"datasets/temporal_order_evaluation_run01")
    args=parser.parse_args()
    protocol=json.loads((args.out/"protocol.json").read_text())
    assert protocol==json.loads((args.runs/"protocol.json").read_text())
    generation=json.loads((args.out/"report_generation.json").read_text())
    assert generation["plot_source_sha256"]==sha(ai/"scripts/plot_temporal_order_evaluation.py")
    assert generation["evaluation_protocol_sha256"]==sha(args.out/"protocol.json")
    for source,digest in protocol["sources"].items():
        if source!="scripts/plot_temporal_order_evaluation.py":
            assert sha(ai/source)==digest, source
    for checkpoint in protocol["checkpoints"]:
        assert sha(checkpoint["path"])==checkpoint["sha256"]
    summary=pd.read_csv(args.out/"summary.csv")
    frame=pd.read_csv(args.runs/"window_metrics.csv")
    selected=pd.read_csv(args.out/"selected_windows.csv")
    weights=np.array([1/5]*4+[1/15]*3)
    checked=0
    for split in ("validation","test"):
        selected_split=selected[selected.split==split]
        inputs=np.load(args.runs/f"{split}_inputs.npz",allow_pickle=False)
        x=inputs["original"]
        assert hashlib.sha256(x.tobytes()).hexdigest()==protocol["input_sha256"][split]
        regenerated,_=make_variants(x,seed=protocol["perturbation_seed"]+(split=="test"))
        for name,values in regenerated.items():
            np.testing.assert_array_equal(values,inputs[name])
        for order in protocol["orders"]:
            p=inputs[f"permutation_{order}"]
            np.testing.assert_array_equal(np.sort(p,axis=1),np.tile(np.arange(64),(len(x),1)))
            np.testing.assert_array_equal(inputs[order],np.take_along_axis(x,p[:,None,:],axis=2))
            np.testing.assert_array_equal(np.sort(inputs[order],axis=2),np.sort(x,axis=2))
        for _,performance in selected_split.groupby("key"):
            starts=np.sort(performance.start)
            assert np.all(np.diff(starts)>=64)
        method_rows=summary[summary.split==split][["arm","stage","seed"]].drop_duplicates()
        for method in method_rows.itertuples(index=False):
            if method.arm in ("raw","summary14"):
                reps={k:inputs[k].astype(float) for k in ("original",*[f"noise{n:.2f}" for n in protocol["noise_levels"]],*protocol["orders"])}
                if method.arm=="summary14":
                    def original_summary(v):
                        mean=v.mean(axis=2);mean[:,1]=np.median(abs(v[:,1]),axis=1)
                        width=np.percentile(v,95,axis=2)-np.percentile(v,5,axis=2)
                        return np.stack((mean,width),axis=2)
                    reps={k:original_summary(v) for k,v in reps.items()}
                    def distance(l,r):
                        return np.sqrt(np.sum(np.mean((l-r)**2,axis=2)*weights,axis=1))
                else:
                    def distance(l,r):
                        return np.sqrt(np.sum(np.mean((l-r)**2,axis=2)*weights,axis=1))
            else:
                saved=np.load(args.runs/f"{method.arm}_{method.seed}_{method.stage}_{split}.npz",allow_pickle=False)
                reps={k:saved[k].astype(float) for k in saved.files}
                def distance(l,r):
                    ln=np.linalg.norm(l,axis=1);rn=np.linalg.norm(r,axis=1)
                    assert np.all(ln>1e-12) and np.all(rn>1e-12)
                    return 1-np.clip(np.einsum("ij,ij->i",l,r)/(ln*rn),-1,1)
            for order in protocol["orders"]:
                far=distance(reps["original"],reps[order])
                raw_far=np.sqrt(np.sum(np.mean((x.astype(float)-inputs[order])**2,axis=2)*weights,axis=1))
                for noise in protocol["noise_levels"]:
                    near=distance(reps["original"],reps[f"noise{noise:.2f}"])
                    rows=frame[(frame.split==split)&(frame.arm==method.arm)&(frame.stage==method.stage)&(frame.seed==method.seed)&(frame.order==order)&(frame.noise==noise)]
                    np.testing.assert_array_equal(rows.window.to_numpy(),selected_split.window.to_numpy())
                    np.testing.assert_allclose(rows.d_noise,near,atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(rows.d_order,far,atol=1e-12,rtol=1e-10)
                    tied=np.abs(near-far)<=protocol["tie_absolute_tolerance"]
                    expected=np.where(tied,.5,(near<far).astype(float))
                    expected[raw_far<=1e-10]=np.nan
                    np.testing.assert_allclose(rows.score,expected,atol=0,rtol=0,equal_nan=True)
                    checked+=len(rows)
    keys=["split","arm","stage","seed","order","noise"]
    good=frame[frame.score.notna()]
    perf=good.groupby(keys+["piece","key"]).score.mean()
    works=perf.groupby(level=keys+["piece"]).mean()
    expected_summary=works.groupby(level=keys).mean().sort_index()*100
    actual_summary=summary.set_index(keys).score_percent.sort_index()
    pd.testing.assert_index_equal(expected_summary.index,actual_summary.index)
    np.testing.assert_allclose(expected_summary,actual_summary,rtol=0,atol=1e-10)
    manifest=json.loads((args.out/"figure_manifest.json").read_text())
    for artifact in manifest:
        assert sha(args.out/artifact["file"])==artifact["sha256"]
    assert json.loads((args.out/"layout_audit.json").read_text())["status"]=="passed"
    result=dict(status="passed",checked_window_rows=checked,expected_rows=len(frame),
        summary_rows=len(summary),checkpoint_hashes_checked=len(protocol['checkpoints']),
        figure_hashes_checked=len(manifest),
        checks=["Frozen evaluation protocol and source/checkpoint integrity",
            "Exact reproducible variants and joint permutations preserving distributions",
            "No overlapping selected windows within performances",
            "All saved distances and triplet scores recalculated from stored inputs/embeddings",
            "Independent performance/work-macro aggregation matches summary",
            "Figure hashes and canvas-bound layout audit"],
        source_sha256=sha(Path(__file__)), protocol_sha256=sha(args.out/"protocol.json"))
    assert checked==len(frame)
    (args.out/"independent_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
