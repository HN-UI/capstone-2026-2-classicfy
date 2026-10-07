"""Independent ASAP beat annotations audit the new automatic alignment heuristic."""
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import warnings
import numpy as np
import pandas as pd
import pretty_midi
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preprocessing import ASAPLoader,load_midi
from preprocessing.atepp_alignment import load_score_midi,align_score_performance


def check(sample):
    row=dict(key=sample.performance_key,composer=sample.composer)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _,grid,est,quality=align_score_performance(load_score_midi(sample.score_path),load_midi(sample.performance_path))
        row.update(quality)
        score=pretty_midi.PrettyMIDI(str(sample.score_path))
        quarter=np.array([score.time_to_tick(t)/score.resolution for t in sample.score_beats])
        real=np.array(sample.performance_beats)
        # Exclude repeated/special annotations and locations outside fitted anchors.
        mask=(quarter>=grid[0])&(quarter<=grid[-1])&np.array([b!='bR' for b in sample.score_beat_types])
        error=np.abs(np.interp(quarter[mask],grid,est)-real[mask])
        widths=np.diff(real); positive=widths[widths>0]
        row.update(annotation_points=int(mask.sum()),beat_error_median_seconds=float(np.median(error)),
                   beat_error_p90_seconds=float(np.percentile(error,90)),
                   beat_error_p90_relative=float(np.percentile(error,90)/np.median(positive)))
    except Exception as e: row.update(error=f'{type(e).__name__}: {e}')
    return row


def main():
    root=Path('../ASAP_dataset/asap-dataset').resolve()
    samples=list(ASAPLoader(root).iter_samples(aligned_only=True))
    # Deterministic samples spread across works and composers; at most one per score.
    unique={}
    for s in samples: unique.setdefault(str(s.score_path),s)
    samples=list(unique.values()); rng=np.random.default_rng(42)
    samples=[samples[i] for i in rng.choice(len(samples),min(40,len(samples)),replace=False)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(check,samples): rows.append(row)
    out=Path('classicfy-ai/analysis/atepp'); out.mkdir(parents=True,exist_ok=True)
    d=pd.DataFrame(rows); d.to_csv(out/'alignment_calibration_asap.csv',index=False,encoding='utf-8-sig')
    print(d[['accepted','beat_error_p90_relative']].groupby('accepted').agg(['count','median','max']).to_string(),flush=True)


if __name__=='__main__': main()
