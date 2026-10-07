"""Blind MIDI-to-MIDI comparison against ASAP beats and (n)ASAP note references.

Run from repository root. Append-only results support resuming the complete corpus.
Reference match files are read AFTER prediction and never passed to the aligners.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import warnings
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pretty_midi
from preprocessing import ASAPLoader, load_midi
from preprocessing.note_alignment import load_match
from preprocessing.atepp_alignment import load_score_midi
from preprocessing.research_alignment import align_research, tempo_grid, MODELS


def identity_map(reference, target, time_ref, time_target, tolerance):
    """Map file identities by pitch/onset alone, independent of cross-file note pairs.

    Ambiguous equal-pitch onsets and many-to-one correspondences are excluded.
    """
    by_pitch = {}
    for t in target: by_pitch.setdefault(t.pitch, []).append(t)
    indexed={}
    for pitch,items in by_pitch.items():
        items=sorted(items,key=time_target)
        indexed[pitch]=(np.array([time_target(t) for t in items]),items)
    found = {}
    for r in reference:
        times,items=indexed.get(r.pitch,(np.empty(0),[]))
        left,right=np.searchsorted(times,[time_ref(r)-tolerance,time_ref(r)+tolerance])
        candidates=items[left:right]
        if len(candidates) == 1: found[r.note_id] = candidates[0].note_id
    counts = {}
    for value in found.values(): counts[value] = counts.get(value, 0)+1
    return {k:v for k,v in found.items() if counts[v] == 1}


def note_reference_metrics(prediction, score, midi, path, robust):
    from preprocessing.note_alignment import PerformedNote
    ref = load_match(path)
    rs = {s.note_id:s for s,p in ref.matches}
    rs.update({s.note_id:s for s in ref.deletions})
    rp = {p.note_id:p for s,p in ref.matches}
    rp.update({p.note_id:p for p in ref.insertions})
    # (n)ASAP has meter-dependent beat coordinates / pickup offsets. Reconcile
    # score coordinates using SCORE notes only; never fit to performance anchors.
    best = (-1, None, None, None)
    for scale in (1., .5, 2., 1.5, 2/3, .25, 4.):
        for offset in {0., min(s.onset_beats for s in score)-scale*min(s.onset_beats for s in rs.values())}:
            mapping = identity_map(rs.values(), score, lambda s:scale*s.onset_beats+offset,
                                   lambda s:s.onset_beats, .002)
            if len(mapping)>best[0]: best=(len(mapping),mapping,scale,offset)
    sm, scale, offset = best[1:]
    ps = [PerformedNote(str(i),n.pitch,n.start,n.end,n.velocity) for i,n in enumerate(midi.notes)]
    pm = identity_map(rp.values(), ps, lambda p:p.onset, lambda p:p.onset, .005)
    gt = {(sm[s.note_id],pm[p.note_id]) for s,p in ref.matches if s.note_id in sm and p.note_id in pm}
    score_domain,perf_domain=set(sm.values()),set(pm.values())
    predicted = {(s.note_id,p.note_id) for s,p in prediction.matches
                 if s.note_id in score_domain and p.note_id in perf_domain}
    tp = len(gt & predicted)
    precision = tp/len(predicted) if predicted else 0.
    recall = tp/len(gt) if gt else 0.
    true_perf_by_score=dict(gt); seconds_by_perf={p.note_id:p.onset for p in ps}
    onset_errors=[abs(seconds_by_perf[pi]-seconds_by_perf[true_perf_by_score[si]])
                  for si,pi in predicted if si in true_perf_by_score]
    result=dict(reference_robust=robust, reference_score_mapping_fraction=len(sm)/len(rs),
                reference_performance_mapping_fraction=len(pm)/len(rp),
                reference_pair_mapping_fraction=len(gt)/max(len(ref.matches),1),
                reference_score_scale=scale,reference_score_offset=offset,
                note_evaluation_eligible=bool(robust and len(sm)/len(rs)>=.95 and len(pm)/len(rp)>=.98 and gt),
                note_reference_pairs=len(gt),note_predicted_pairs=len(predicted),note_true_positive=tp,
                note_precision=precision,note_recall=recall,
                note_f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    if onset_errors:
        result.update(note_onset_error_mae_seconds=float(np.mean(onset_errors)),
                      note_onset_error_p90_seconds=float(np.percentile(onset_errors,90)))
    return result


def evaluate(task):
    sample, model, cache = task
    row = dict(key=sample.performance_key, work=str(sample.score_path), composer=sample.composer, model=model)
    row.update(runtime_workers=int(os.environ.get('CLASSICFY_WORKERS','4')),
               runtime_torch_threads=int(os.environ.get('CLASSICFY_TORCH_THREADS','2')))
    row['inference_backend']=os.environ.get('CLASSICFY_GLUE_BACKEND','torch') if model=='gluenote' else 'not_neural'
    start=time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            score=load_score_midi(sample.score_path); midi=load_midi(sample.performance_path)
            alignment,grid,estimated,metrics=align_research(score,midi,model)
        row.update(metrics)
        positions,times,_,_=tempo_grid(alignment)
        raw_score=pretty_midi.PrettyMIDI(str(sample.score_path))
        q=np.array([raw_score.time_to_tick(t)/raw_score.resolution for t in sample.score_beats])
        real=np.asarray(sample.performance_beats)
        valid=np.array([b!='bR' for b in sample.score_beat_types])
        local=np.r_[np.diff(real),np.median(np.diff(real))]
        valid &= np.isfinite(q)&np.isfinite(real)&(local>0)
        q,real,local=q[valid],real[valid],local[valid]
        covered=(q>=positions[0])&(q<=positions[-1])
        err=np.abs(np.interp(q[covered],positions,times)-real[covered])
        relative=err/local[covered]
        row.update(status='ok',annotation_points=len(q),covered_points=int(covered.sum()),
                   beat_coverage=float(covered.mean()),
                   beat_mae_seconds=float(err.mean()),beat_median_seconds=float(np.median(err)),
                   beat_p90_seconds=float(np.percentile(err,90)),beat_p95_seconds=float(np.percentile(err,95)),
                   beat_p90_relative=float(np.percentile(relative,90)),
                   beat_within_50ms=float((err<=.05).sum()/len(q)),
                   beat_within_100ms=float((err<=.1).sum()/len(q)),
                   beat_within_quarter=float((relative<=.25).sum()/len(q)),
                   beat_catastrophic_fraction=float((relative>1).sum()/len(q)))
        # Save prediction before opening reference correspondence files.
        target=Path(cache)/model/(hashlib.sha256(sample.performance_key.encode()).hexdigest()[:20]+'.npz')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            with np.load(target) as previous:
                old_pairs=set(zip(previous['score_ids'].tolist(),previous['performance_ids'].tolist()))
            new_pairs={(s.note_id,p.note_id) for s,p in alignment.matches}
            row['exact_kernel_previous_pairs_equal']=old_pairs==new_pairs
        row['cost_kernel']='exact_numba_dtw_loops'
        np.savez_compressed(target,score_ids=np.array([s.note_id for s,p in alignment.matches]),
                            performance_ids=np.array([p.note_id for s,p in alignment.matches]),
                            positions=positions,times=times,reference_quarters=q,reference_seconds=real,
                            covered=covered,errors_seconds=err,errors_relative=relative)
        row['prediction_cache']=str(target)
        if sample.note_alignment_path:
            try:
                row.update(note_reference_metrics(alignment,score,midi,sample.note_alignment_path,
                                                  sample.robust_note_alignment))
            except Exception as e:
                row.update(note_evaluation_eligible=False,note_reference_error=str(e))
        row['input_sha256']=hashlib.sha256(sample.score_path.read_bytes()+sample.performance_path.read_bytes()).hexdigest()
    except Exception as exc:
        row.update(status='error',error=f'{type(exc).__name__}: {exc}',beat_within_quarter=0.,
                   beat_within_100ms=0.,beat_within_50ms=0.,beat_coverage=0.,accepted=False)
    row['total_seconds']=time.perf_counter()-start
    return row


def paired_ci(values,seed=42):
    values=np.asarray(values,float); rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    return dict(mean=float(values.mean()),low=float(np.percentile(means,2.5)),high=float(np.percentile(means,97.5)),works=len(values))


def refresh_reference(task):
    row,sample=task
    if row['status']!='ok' or not sample.note_alignment_path: return row
    try:
        from preprocessing.research_alignment import alignment_from_predictions
        score=load_score_midi(sample.score_path); midi=load_midi(sample.performance_path)
        with np.load(row['prediction_cache']) as z:
            pred=[dict(label='match',score_id=str(s),performance_id=str(p))
                  for s,p in zip(z['score_ids'],z['performance_ids'])]
        alignment=alignment_from_predictions(pred,score,midi)
        row.update(note_reference_metrics(alignment,score,midi,sample.note_alignment_path,
                                          sample.robust_note_alignment))
        row['reference_sha256']=hashlib.sha256(sample.note_alignment_path.read_bytes()).hexdigest()
    except Exception as e: row.update(note_evaluation_eligible=False,note_reference_error=str(e))
    return row


def summarize(out, paired_only=False):
    rows=[json.loads(line) for line in (out/'results.jsonl').read_text(encoding='utf-8').splitlines() if line]
    frame=pd.DataFrame(rows).drop_duplicates(['key','model'],keep='last')
    observed_performances=frame.key.nunique()
    if paired_only:
        counts=frame.groupby('key').model.nunique()
        paired_keys=counts[counts==len(MODELS)].index
        frame[~frame.key.isin(paired_keys)].to_csv(out/'unpaired_observations.csv',index=False,encoding='utf-8-sig')
        frame=frame[frame.key.isin(paired_keys)].copy()
        if frame.empty: raise ValueError('No paired performances available')
    if 'inference_backend' not in frame: frame['inference_backend']=np.nan
    frame['inference_backend']=frame.inference_backend.fillna(frame.model.map({'gluenote':'torch','dual_dtw':'not_neural'}))
    for column,default in [('runtime_workers',4),('runtime_torch_threads',2)]:
        if column not in frame: frame[column]=default
        frame[column]=frame[column].fillna(default).astype(int)
    frame.to_csv(out/'per_performance.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for model,d in frame.groupby('model'):
        ok=d[d.status=='ok']; eligible=ok[ok.get('note_evaluation_eligible',pd.Series(False,index=ok.index))==True]
        works=d.groupby('work')[['beat_within_quarter','beat_within_100ms','beat_coverage']].mean()
        summary.append(dict(model=model,attempted=len(d),completed=len(ok),failures=len(d)-len(ok),
                            works=d.work.nunique(),beat_within_quarter=works.beat_within_quarter.mean(),
                            beat_within_100ms=works.beat_within_100ms.mean(),beat_coverage=works.beat_coverage.mean(),
                            beat_mae_seconds=ok.groupby('work').beat_mae_seconds.mean().mean(),
                            median_performance_p90_seconds=ok.beat_p90_seconds.median(),
                            beat_catastrophic_fraction=ok.groupby('work').beat_catastrophic_fraction.mean().mean(),
                            median_runtime_seconds=ok.alignment_runtime_seconds.median(),
                            eligible_note_performances=len(eligible),
                            note_precision=eligible.groupby('work').note_precision.mean().mean() if len(eligible) else np.nan,
                            note_recall=eligible.groupby('work').note_recall.mean().mean() if len(eligible) else np.nan,
                            note_f1=eligible.groupby('work').note_f1.mean().mean() if len(eligible) else np.nan,
                            note_onset_mae_seconds=eligible.groupby('work').note_onset_error_mae_seconds.mean().mean()
                                if len(eligible) and 'note_onset_error_mae_seconds' in eligible else np.nan,
                            conflicting_pairs_removed=int(ok.get('conflicting_prediction_pairs_removed',pd.Series(dtype=int)).sum()),
                            accepted_fraction=d.accepted.fillna(False).mean()))
    summary=pd.DataFrame(summary); summary.to_csv(out/'model_summary.csv',index=False,encoding='utf-8-sig')
    frame[frame.status=='ok'].groupby(['model','inference_backend','runtime_workers','runtime_torch_threads']).agg(
        performances=('key','count'),median_runtime_seconds=('alignment_runtime_seconds','median'),
        median_score_notes=('score_notes','median')).to_csv(out/'runtime_conditions.csv',encoding='utf-8-sig')
    # Primary criterion: work-macro beat accuracy, with uncovered/failed beats penalized.
    # Secondary criterion: robust-reference note F1; runtime only breaks near ties.
    pivot=frame.pivot_table(index='work',columns='model',values='beat_within_quarter',aggfunc='mean').dropna()
    ci=paired_ci(pivot['gluenote']-pivot['dual_dtw'])
    notes=frame[(frame.status=='ok')&(frame.note_evaluation_eligible==True)].pivot_table(
        index='key',columns='model',values='note_f1').dropna()
    note_ci=None
    if len(notes) and set(MODELS).issubset(notes.columns):
        notes['work']=frame.drop_duplicates('key').set_index('key').work.reindex(notes.index)
        note_ci=paired_ci(notes.groupby('work').apply(lambda d:(d.gluenote-d.dual_dtw).mean()).to_numpy())
    if ci['low']>0: selected='gluenote'; reason='work-macro beat accuracy paired CI > 0'
    elif ci['high']<0: selected='dual_dtw'; reason='work-macro beat accuracy paired CI < 0'
    elif note_ci and note_ci['low']>0: selected='gluenote'; reason='beat comparison inconclusive; paired note F1 favors gluenote'
    elif note_ci and note_ci['high']<0: selected='dual_dtw'; reason='beat comparison inconclusive; paired note F1 favors dual_dtw'
    else:
        selected=summary.sort_values('median_runtime_seconds').iloc[0].model
        reason='accuracy comparison inconclusive; practical runtime tie-break, no accuracy superiority claim'
    result=dict(selected_model=selected,selection_reason=reason,beat_difference_gluenote_minus_dual=ci,
                paired_only=paired_only,evaluated_performances=int(frame.key.nunique()),
                evaluated_works=int(frame.work.nunique()),observed_performances=int(observed_performances),
                sampling_policy='user-requested stop; completed paired prefix in composer/title order' if paired_only else 'full scheduled corpus',
                note_difference_gluenote_minus_dual=note_ci,
                caveats=['TheGlueNote training includes (n)ASAP; this is not a held-out generalization benchmark.',
                         '(n)ASAP notes are algorithm-produced references; agreement is not independent human note accuracy.',
                         'MIDI score has no ornament annotation; both receive identical MIDI-only inputs.',
                         'ATEPP consists of automatic transcriptions: ASAP winner may not transfer.'])
    (out/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    composer=frame.groupby(['composer','model']).agg(performances=('key','count'),
        beat_within_quarter=('beat_within_quarter','mean'),beat_within_100ms=('beat_within_100ms','mean'))
    composer.to_csv(out/'by_composer.csv',encoding='utf-8-sig')
    print(summary.to_string(index=False),flush=True); print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--asap',type=Path,default=Path('../ASAP_dataset/asap-dataset'))
    p.add_argument('--nasap',type=Path,default=Path('../ASAP_dataset/nasap-dataset'))
    p.add_argument('--output',type=Path,default=Path('classicfy-ai/analysis/alignment_comparison'))
    p.add_argument('--workers',type=int,default=2); p.add_argument('--limit',type=int)
    p.add_argument('--torch-threads',type=int,default=2)
    p.add_argument('--glue-backend',choices=['torch','openvino_cpu','openvino_gpu'],default='torch')
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--paired-only',action='store_true',help='Evaluate only performances with saved results from both models')
    p.add_argument('--retry-errors',action='store_true')
    p.add_argument('--refresh-reference',action='store_true',help='Re-evaluate saved pairs; no model inference')
    args=p.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    os.environ['CLASSICFY_WORKERS']=str(args.workers)
    os.environ['CLASSICFY_TORCH_THREADS']=str(args.torch_threads)
    os.environ['CLASSICFY_GLUE_BACKEND']=args.glue_backend
    if args.summarize_only: summarize(out,args.paired_only); return
    loader=ASAPLoader(args.asap,note_alignment_root=args.nasap)
    samples=[]; excluded=[]
    for key in loader._rows:
        try:
            s=loader.get_sample(key)
            if s.aligned: samples.append(s)
            else: excluded.append(dict(key=key,reason='ASAP aligned flag false'))
        except Exception as e: excluded.append(dict(key=key,reason=str(e)))
    # Interleave pieces for pilots; full run includes every eligible performance.
    samples=sorted(samples,key=lambda s:(s.composer,s.title,s.performance_key))
    if args.limit: samples=samples[:args.limit]
    if args.refresh_reference:
        latest={}
        for row in map(json.loads,(out/'results.jsonl').read_text(encoding='utf-8').splitlines()):
            latest[(row['key'],row['model'])]=row
        by_key={s.performance_key:s for s in samples}
        tasks=[(r,by_key[r['key']]) for r in latest.values()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            refreshed=list(pool.map(refresh_reference,tasks,chunksize=4))
        (out/'results.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False,allow_nan=False)
            for r in refreshed)+'\n',encoding='utf-8')
        summarize(out,args.paired_only); return
    pd.DataFrame(excluded).to_csv(out/'excluded_inputs.csv',index=False,encoding='utf-8-sig')
    done=set()
    if (out/'results.jsonl').exists():
        latest={}
        for r in map(json.loads,(out/'results.jsonl').read_text(encoding='utf-8').splitlines()):
            latest[(r['key'],r['model'])]=r
        done={key for key,r in latest.items() if not args.retry_errors or r['status']=='ok'}
    tasks=[(s,m,str(out/'predictions')) for s in samples for m in MODELS if (s.performance_key,m) not in done]
    versions={name:importlib.metadata.version(name) for name in ['parangonar','partitura','torch','numpy']}
    dist=importlib.metadata.distribution('parangonar')
    weights=dist.locate_file('parangonar/assets/thegluenote_small_checkpoint.pt')
    (out/'environment.json').write_text(json.dumps(dict(versions=versions,python=sys.version,
        corpus_performances=len(samples),models=list(MODELS),input='identical score MIDI and performance MIDI',
        references_not_provided_to_matchers=True,selection_primary='work-macro fraction of all annotated beats within 0.25 local beat',
        threads_per_worker=args.torch_threads,workers=args.workers,checkpoint='thegluenote_small_checkpoint.pt',
        checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        dual_dtw_cost_kernel='float64 exact compiled membership and scalar pitchwise DTW; original traceback',
        gluenote_backend=args.glue_backend,
        conflict_policy='remove all conflicting edges without choosing a replacement'),indent=2),encoding='utf-8')
    print(f'Corpus {len(samples)} performances; remaining tasks {len(tasks)}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate,t):t for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            row=f.result()
            with (out/'results.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            print(f'{i}/{len(tasks)} {row["model"]} {row["key"]}: {row["status"]} '
                  f'{row.get("beat_within_quarter",0):.3f} {row["total_seconds"]:.1f}s {row.get("error","")}',flush=True)
    summarize(out,args.paired_only)


if __name__=='__main__': main()
