"""Download inventory -> MIDI audit -> approximate alignment -> existing features.

Run from repository root with .venv/Scripts/python.exe classicfy-ai/scripts/extend_atepp.py.
Large sequence caches are outside Git, at ../ATEPP_dataset/features.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from preprocessing import load_midi
from preprocessing.atepp_alignment import canonical_path, load_score_midi, align_score_performance
from features import (extract_dynamics, extract_pedaling, extract_articulation, BeatSequence,
                      PieceFeatureInput, separate_piece_feature, standardize_piece_feature,
                      TempoInput, extract_piece_tempo_features, extract_piece_rubato_features,
                      fit_residual_scale, standardize_sequence)

CHANNELS = ('tempo', 'rubato', 'dynamics', 'articulation', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes')
BAD_QUALITY = {'corrupted', 'low quality', 'background noise', 'applause'}
ERA = {'Johann Sebastian Bach':'Baroque', 'George Frideric Handel':'Baroque', 'Domenico Scarlatti':'Baroque',
       'Franz Joseph Haydn':'Classical', 'Wolfgang Amadeus Mozart':'Classical',
       'Ludwig van Beethoven':'Classical/Romantic', 'Claude Debussy':'Impressionist', 'Maurice Ravel':'Impressionist',
       'Dmitri Shostakovich':'20th century', 'Anton Webern':'20th century', 'Paul Hindemith':'20th century',
       'Sergei Prokofiev':'20th century', 'Béla Bartók':'20th century',
       'Sergei Rachmaninoff':'Late Romantic', 'Alexander Scriabin':'Late Romantic'}


@lru_cache(maxsize=32)
def score_notes(path):
    return load_score_midi(path)


def process_record(task):
    row, root, backend = task
    root = Path(root)
    result = dict(row)
    result['aligner'] = backend
    result['status'] = 'error'
    try:
        path = Path(row['local_midi'])
        if not path.is_file():
            result.update(status='missing_midi', error='Metadata MIDI not found')
            return result
        with warnings.catch_warnings(record=True) as caught:
            midi = load_midi(path)
        result['midi_warnings'] = '|'.join(sorted({str(w.message) for w in caught}))
        onsets = np.array([n.start for n in midi.notes])
        duration = np.array([n.duration for n in midi.notes])
        velocity = np.array([n.velocity for n in midi.notes])
        result.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), notes=len(midi.notes),
                      duration_seconds=midi.duration, pedal_events=len(midi.pedals),
                      pedal_value_levels=len({p.value for p in midi.pedals}),
                      velocity_mean=float(velocity.mean()) if len(velocity) else np.nan,
                      short_note_fraction=float(np.mean(duration < .02)) if len(duration) else np.nan,
                      invalid_notes=int(sum(n.start < 0 or n.end <= n.start or not 1 <= n.velocity <= 127
                                            or not 0 <= n.pitch <= 127 or not np.isfinite([n.start, n.end]).all()
                                            for n in midi.notes)),
                      invalid_pedals=int(sum(not 0 <= p.value <= 127 or not np.isfinite(p.time) or p.time < 0
                                             for p in midi.pedals)))
        if not len(onsets) or result['invalid_notes'] or result['invalid_pedals']:
            result['status'] = 'invalid_midi'
            return result
        if row['quality'] in BAD_QUALITY:
            result['status'] = 'excluded_quality'
            return result
        if not row['local_score']:
            result['status'] = 'no_score_midi'
            return result
        score = score_notes(row['local_score'])
        if len(score)<20:
            result.update(status='invalid_score_midi',score_notes=len(score),
                          error='Score MIDI has fewer than 20 usable notes')
            return result
        if backend == 'heuristic':
            alignment, beats, perf_beats, metrics = align_score_performance(score, midi)
        else:
            from preprocessing.research_alignment import align_research, tempo_grid
            alignment, beats, perf_beats, metrics = align_research(score, midi, backend)
            anchors, times, _, _ = tempo_grid(alignment)
            alignment_path = root / 'alignments' / backend / (row['perf_id'] + '.npz')
            alignment_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(alignment_path, score_ids=np.array([s.note_id for s,p in alignment.matches]),
                performance_ids=np.array([p.note_id for s,p in alignment.matches]),
                score_onsets=np.array([s.onset_beats for s,p in alignment.matches]),
                score_offsets=np.array([s.offset_beats for s,p in alignment.matches]),
                performance_onsets=np.array([p.onset for s,p in alignment.matches]),
                performance_offsets=np.array([p.offset for s,p in alignment.matches]),
                pitch=np.array([s.pitch for s,p in alignment.matches]),
                anchors=anchors, times=times, score_beats=beats, performance_beats=perf_beats)
            result['alignment_cache'] = str(alignment_path)
        result.update(metrics)
        if not metrics['accepted']:
            result['status'] = 'rejected_alignment'
            return result
        dynamics = extract_dynamics(midi, perf_beats).sequence
        pedal = extract_pedaling(midi, perf_beats)
        articulation = extract_articulation(alignment, perf_beats).sequence
        # Absence of a transcribed CC64 channel is not evidence of no pedal use.
        if not midi.pedals:
            pedal_sequences = [np.full(len(dynamics), np.nan)] * 3
        else:
            pedal_sequences = [pedal.depth.values, pedal.down_ratio.values, pedal.changes.values]
        values = np.column_stack([dynamics.values, articulation.values, *pedal_sequences])
        feature_root = root / 'features' if backend == 'heuristic' else root / 'features' / backend
        cache = feature_root / 'raw' / (row['perf_id'] + '.npz')
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, score_beats=beats, performance_beats=perf_beats,
                            dpa=values, dpa_mask=np.isfinite(values),
                            match_score_onsets=np.array([s.onset_beats for s, p in alignment.matches]),
                            match_score_offsets=np.array([s.offset_beats for s, p in alignment.matches]),
                            match_performance_onsets=np.array([p.onset for s, p in alignment.matches]),
                            match_performance_offsets=np.array([p.offset for s, p in alignment.matches]),
                            match_pitch=np.array([s.pitch for s, p in alignment.matches]))
        result['raw_cache'] = str(cache)
        result['status'] = 'aligned'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def inventory(root):
    metadata = pd.read_csv(root / 'ATEPP-metadata-1.2.csv', dtype={'perf_id': str}).fillna('')
    files = [p for p in (root / 'ATEPP-1.2').rglob('*') if p.is_file() and '__MACOSX' not in p.parts]
    midi_index = {p.stem: p for p in files if p.suffix == '.mid' and p.stem[:1].isdigit()}
    score_index = {canonical_path(p.relative_to(root / 'ATEPP-1.2').as_posix()): p
                   for p in files if p.suffix == '.midi'}
    metadata['era'] = metadata.composer.map(lambda c: ERA.get(c, 'Romantic'))
    metadata['local_midi'] = metadata.perf_id.map(lambda p: str(midi_index.get(p, '')))
    metadata['local_score'] = metadata.score_path.map(
        lambda p: str(score_index.get(canonical_path(p + '.midi'), '')) if p else '')
    # Same-directory fallback only when unambiguous; never choose a different movement.
    for i, r in metadata.iterrows():
        if r.score_path and not r.local_score and r.local_midi:
            nearby = list(Path(r.local_midi).parent.glob('*.midi'))
            if len(nearby) == 1:
                metadata.at[i, 'local_score'] = str(nearby[0])
    return metadata


def normalize(audit, root, output, backend='heuristic'):
    summary, scales, group_log = [], [], []
    good = audit[audit.status == 'aligned'].copy()
    good['norm_group'] = good.composition_id.astype(str) + '::' + good.local_score.astype(str)
    for group, rows in good.groupby('norm_group', sort=True):
        records = []
        for r in rows.to_dict('records'):
            with np.load(r['raw_cache']) as z:
                records.append((r, {k: z[k].copy() for k in ('score_beats', 'performance_beats', 'dpa')}))
        if len(records) < 2:
            group_log.append(dict(group=group, status='singleton', performances=len(records)))
            continue
        start = max(z['score_beats'][0] for r, z in records)
        end = min(z['score_beats'][-1] for r, z in records)
        grid = np.arange(start, end + 1.)
        if len(grid) < 17:
            group_log.append(dict(group=group, status='short_overlap', performances=len(records)))
            continue
        tempi, arrays = [], []
        for r, z in records:
            left, right = np.searchsorted(z['score_beats'], [start, end])
            pb = z['performance_beats'][left:right + 1]
            arrays.append(z['dpa'][left:right])
            tempi.append(TempoInput(r['perf_id'], grid * .5, pb, ['b'] * len(grid), ['b'] * len(grid)))
        tempo = extract_piece_tempo_features(tempi)
        rubato = extract_piece_rubato_features(tempo)
        raw, relative, normalized, masks = {}, {}, {}, {}
        for r, arr in zip(rows.to_dict('records'), arrays):
            key = r['perf_id']
            raw[key] = np.column_stack([[i.score_relative_tempo if i.mask else np.nan for i in tempo[key].intervals],
                                        rubato[key].absolute_rubato_sequence.values, arr])
            relative[key] = np.full_like(raw[key], np.nan)
            relative[key][:, 0] = tempo[key].individual_tempo_sequence.values
            relative[key][:, 1] = rubato[key].relative_rubato_sequence.values
            normalized[key] = np.full_like(raw[key], np.nan)
        for c, name in enumerate(CHANNELS):
            if c < 2:
                scale = fit_residual_scale([BeatSequence(relative[k][:, c], np.isfinite(relative[k][:, c]))
                                            for k in relative], method='mad')
                for key in relative:
                    normalized[key][:, c] = standardize_sequence(
                        BeatSequence(relative[key][:, c], np.isfinite(relative[key][:, c])), scale).values
            else:
                items = [PieceFeatureInput(k, group, grid, BeatSequence(raw[k][:, c], np.isfinite(raw[k][:, c])))
                         for k in raw]
                separated = separate_piece_feature(items)
                fitted = standardize_piece_feature(list(separated.values()), feature_name=name)
                scale = next(iter(fitted.values())).scale
                for key in fitted:
                    relative[key][:, c] = fitted[key].relative.values
                    normalized[key][:, c] = fitted[key].standardized.values
            scales.append(dict(group=group, channel=name, denominator=scale.value,
                               method=scale.method, valid_count=scale.valid_count))
        for r, z in records:
            key = r['perf_id']
            feature_root = root / 'features' if backend == 'heuristic' else root / 'features' / backend
            target = feature_root / 'normalized' / (key + '.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, channels=np.array(CHANNELS), score_beats=grid,
                                raw=raw[key], relative=relative[key], normalized=normalized[key],
                                mask=np.isfinite(normalized[key]))
            s = {k: r[k] for k in ['perf_id','composition_id','composer','artist','artist_id','era','quality','track']}
            s['sha256']=r['sha256']
            s.update(norm_group=group, normalized_cache=str(target), beats=len(grid)-1,
                     retained_grid_fraction=(end-start)/(z['score_beats'][-1]-z['score_beats'][0]),
                     pedal_events=r['pedal_events'], pedal_value_levels=r['pedal_value_levels'],
                     score_recall=r['score_recall'], performance_precision=r['performance_precision'])
            for layer, data in [('raw',raw[key]), ('relative',relative[key]), ('norm',normalized[key])]:
                for c, name in enumerate(CHANNELS):
                    v = data[:, c]; v = v[np.isfinite(v)]
                    s[f'{layer}_{name}_mean'] = float(np.mean(v)) if len(v) else np.nan
                    s[f'{layer}_{name}_range'] = float(np.diff(np.percentile(v,[5,95]))[0]) if len(v) else np.nan
                    s[f'{layer}_{name}_abs'] = float(np.median(np.abs(v))) if len(v) else np.nan
            summary.append(s)
        group_log.append(dict(group=group, status='normalized', performances=len(records), intervals=len(grid)-1))
    pd.DataFrame(summary).to_csv(output / 'performance_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(scales).to_csv(output / 'scales.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(group_log).to_csv(output / 'normalization_groups.csv', index=False, encoding='utf-8-sig')
    print(f'Normalized {len(summary)} performances across {sum(x["status"]=="normalized" for x in group_log)} groups', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('../ATEPP_dataset'))
    parser.add_argument('--output', type=Path, default=Path('classicfy-ai/analysis/atepp'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--normalize-only', action='store_true')
    parser.add_argument('--aligner', choices=['heuristic','dual_dtw','gluenote','selected'], default='heuristic')
    parser.add_argument('--selection', type=Path, default=Path('classicfy-ai/analysis/alignment_comparison/selection.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); root=args.root.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    backend = args.aligner
    if backend == 'selected':
        backend = json.loads(args.selection.read_text(encoding='utf-8'))['selected_model']
    if backend not in ('heuristic','dual_dtw','gluenote'): raise ValueError('Invalid selection file')
    config_path=out/'run_config.json'
    if args.resume and config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8'))['aligner']!=backend:
            raise ValueError('Resume folder belongs to a different aligner')
    config_path.write_text(json.dumps(dict(aligner=backend,dataset_root=str(root)),indent=2),encoding='utf-8')
    if args.resume and (out/'provenance.json').exists():
        previous_backend=json.loads((out/'provenance.json').read_text(encoding='utf-8')).get('alignment')
        if previous_backend != backend:
            raise ValueError('Selected aligner changed; use a separate output folder to avoid mixing caches')
    if args.normalize_only:
        audit = pd.read_csv(out/'midi_audit.csv', dtype={'perf_id':str})
    else:
        meta = inventory(root)
        meta.to_csv(out/'inventory.csv',index=False,encoding='utf-8-sig')
        print(f'Metadata={len(meta)}, MIDI found={(meta.local_midi!="").sum()}, score MIDI found={(meta.local_score!="").sum()}',flush=True)
        if args.limit:
            # Pilot scored works; the full run audits every metadata row.
            meta = meta[(meta.local_score!='') & ~meta.quality.isin(BAD_QUALITY)].head(args.limit)
        results=[]
        if args.resume and (out/'midi_audit.partial.csv').exists():
            previous=pd.read_csv(out/'midi_audit.partial.csv',dtype={'perf_id':str}).fillna('')
            results=previous[previous.status!='error'].to_dict('records')
        done={r['perf_id'] for r in results}
        tasks=[(r,str(root),backend) for r in meta.to_dict('records') if r['perf_id'] not in done]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(process_record,tasks,chunksize=2),1):
                results.append(r)
                if i%100==0 or i==len(tasks):
                    print(f'Processed {i}/{len(tasks)}; aligned={sum(x["status"]=="aligned" for x in results)}',flush=True)
                    pd.DataFrame(results).to_csv(out/'midi_audit.partial.csv',index=False,encoding='utf-8-sig')
        audit=pd.DataFrame(results)
        audit.to_csv(out/'midi_audit.csv',index=False,encoding='utf-8-sig')
        source_files=[Path(__file__),Path(__file__).parents[1]/'src/preprocessing/atepp_alignment.py',
                      root/'ATEPP-metadata-1.2.csv']
        if backend!='heuristic':
            source_files.append(Path(__file__).parents[1]/'src/preprocessing/research_alignment.py')
            source_files.append(Path(__file__).parents[1]/'src/preprocessing/exact_dtw_kernel.py')
            if backend=='gluenote': source_files.append(Path(__file__).parents[1]/'src/preprocessing/gluenote_openvino.py')
        provenance=dict(version='ATEPP-1.2', source='https://github.com/tangjjbetsy/ATEPP',
                        archive_sha256=hashlib.sha256((root/'ATEPP-1.2.zip').read_bytes()).hexdigest(),
                        code_metadata_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest(),
                        alignment=backend, automatic_alignment_not_ground_truth=True,
                        selection=str(args.selection.resolve()) if args.aligner=='selected' else None,
                        normalization='descriptive candidate-pool median and per-work scale; transductive, not train/test evaluation',
                        era_policy='composer-level coarse mapping; Beethoven transitional; not composition-date labels',
                        statuses=audit.status.value_counts().to_dict())
        if backend!='heuristic':
            provenance['versions']={name:importlib.metadata.version(name) for name in
                                    ['parangonar','partitura','torch','symusic','miditok','numpy']}
            package=importlib.metadata.distribution('parangonar')
            weights=package.locate_file('parangonar/assets/thegluenote_small_checkpoint.pt')
            provenance['checkpoint_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
            provenance['checkpoint']='thegluenote_small_checkpoint.pt' if backend=='gluenote' else 'not used'
            provenance['dual_dtw_kernel']='exact compiled float64 membership recurrence, original traceback'
            provenance['inference_backend']=os.environ.get('CLASSICFY_GLUE_BACKEND','torch') if backend=='gluenote' else 'not_neural'
            if backend=='gluenote' and provenance['inference_backend']!='torch':
                provenance['versions']['openvino']=importlib.metadata.version('openvino')
            if args.aligner=='selected':
                provenance['selection_sha256']=hashlib.sha256(args.selection.read_bytes()).hexdigest()
        (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    normalize(audit,root,out,backend)


if __name__=='__main__':
    main()
