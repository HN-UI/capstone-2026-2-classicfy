"""Step 2: read four-bin temporal trajectories from frozen representations.

All probe fitting, feature normalization, PCA and penalty selection use training
works only. Original validation/test data and saved encoders remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare_sequence_autoencoders import prepare, SEEDS
from evaluate_temporal_order import encode
from embedding.extended_experiments import load, sha, dump
from embedding.order_evaluation import summary_vectors, select_complete_nonoverlapping
from embedding.temporal_probe import (quarter_targets, sample_weights, work_folds,
    fit_ridge, predict_ridge, fit_pca, project_pca, channel_energy,
    feature_average, normalized_error, evaluate_channels)


ARMS = ("cnn64", "bilstm64", "transformer64")
STAGES = ("initial", "best_extended")
ALPHAS = (1e-4, .001, .01, .1, 1., 10.)


def identifiers(metadata):
    return np.array([m["piece"] for m in metadata]), np.array([m["key"] for m in metadata])


def fit_probe(features, targets, metadata, folds, *, pca=False):
    pieces, keys = identifiers(metadata)
    cv = []
    for fold in range(5):
        train, held = folds != fold, folds == fold
        assert not set(pieces[train]) & set(pieces[held])
        tw = sample_weights(pieces[train], keys[train])
        hw = sample_weights(pieces[held], keys[held])
        tx, hx = features[train], features[held]
        if pca:
            transform = fit_pca(tx, tw)
            tx, hx = project_pca(transform, tx), project_pca(transform, hx)
        energy = channel_energy(targets[train], tw)
        for alpha in ALPHAS:
            model = fit_ridge(tx, targets[train], tw, alpha)
            error = normalized_error(targets[held], predict_ridge(model, hx), hw, energy)
            cv.append(dict(fold=fold, alpha=alpha, normalized_mse=error,
                train_works=len(set(pieces[train])), held_works=len(set(pieces[held])),
                train_windows=int(train.sum()), held_windows=int(held.sum())))
    scores = {alpha:np.average([r["normalized_mse"] for r in cv if r["alpha"] == alpha],
                 weights=[r["held_works"] for r in cv if r["alpha"] == alpha]) for alpha in ALPHAS}
    alpha = min(ALPHAS, key=lambda a:scores[a])
    weights = sample_weights(pieces, keys)
    transform = fit_pca(features, weights) if pca else None
    x = project_pca(transform, features) if pca else features
    model = fit_ridge(x, targets, weights, alpha)
    return model, transform, cv, scores


def metric_rows(method, arm, stage, seed, targets, predictions, metadata, split):
    pieces, keys = identifiers(metadata)
    values = evaluate_channels(targets, predictions, sample_weights(pieces, keys))
    common = dict(method=method, arm=arm, stage=stage, seed=seed, split=split)
    channels = [dict(**common, channel=ci, **{k:float(v[ci]) for k,v in values.items()},
                     works=len(set(pieces)), performances=len(set(keys)), windows=len(keys)) for ci in range(7)]
    summary = dict(**common, dynamics_score_percent=float(values["score_percent"][2]),
        macro_score_percent=feature_average(values["score_percent"]),
        dynamics_mse=float(values["mse"][2]), dynamics_flat_mse=float(values["flat_mse"][2]),
        dynamics_shape=float(values["centered_cosine"][2]), dynamics_amplitude_percent=float(values["amplitude_percent"][2]),
        works=len(set(pieces)), performances=len(set(keys)), windows=len(keys))
    works = []
    for piece in sorted(set(pieces)):
        keep = pieces == piece
        current = evaluate_channels(targets[keep], predictions[keep], sample_weights(pieces[keep], keys[keep]))
        for ci in range(7):
            works.append(dict(**common, piece=piece, channel=ci,
                **{k:float(v[ci]) for k,v in current.items()},
                performances=len(set(keys[keep])), windows=int(keep.sum())))
    return summary, channels, works


def main():
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=data / "temporal_embedding_cluster_run01")
    parser.add_argument("--cache", type=Path, default=data / "feature_normalization_cluster_raw.npz")
    parser.add_argument("--asap-root", type=Path, default=data / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=data / "nASAP")
    parser.add_argument("--previous-runs", type=Path, default=data / "temporal_order_evaluation_run01")
    parser.add_argument("--runs", type=Path, default=data / "temporal_probe_run01")
    parser.add_argument("--out", type=Path, default=ai / "analysis/temporal_probe")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        from plot_temporal_probe import render
        render(args.out)
        return
    if args.runs.exists() or args.out.exists():
        raise ValueError("Preserve prior outputs: choose new --runs/--out or --report-only")
    torch.set_num_threads(2)
    prepared = prepare(args)
    previous = json.loads((args.previous_runs / "protocol.json").read_text())
    assert sha(args.reference / "dataset.json") == previous["reference_dataset_sha256"]
    assert sha(args.cache) == previous["cache_sha256"]
    checkpoint_list = [c for c in previous["checkpoints"] if c["stage"] in STAGES]
    for checkpoint in checkpoint_list:
        if sha(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Saved encoder changed")
    inputs, targets, metadata, supports = {}, {}, {}, []
    selected_rows = []
    for split in ("train", "validation", "test"):
        dataset = prepared[4][split]
        x, valid = prepared[5][split]
        selected = select_complete_nonoverlapping(dataset, valid.numpy())
        inputs[split] = x.numpy()[selected].copy()
        targets[split] = quarter_targets(inputs[split])
        metadata[split] = []
        for index in selected:
            si,start = dataset.items[index]
            sequence = dataset.sequences[si]
            row = dict(window=int(index),piece=sequence.piece,key=sequence.key,start=int(start))
            metadata[split].append(row)
            selected_rows.append(dict(split=split,**row))
        pieces,keys = identifiers(metadata[split])
        supports.append(dict(split=split,original_works=len({s.piece for s in dataset.sequences}),
            original_performances=len(dataset.sequences),original_windows=len(dataset),
            works=len(set(pieces)),performances=len(set(keys)),windows=len(selected),
            excluded_works=sorted({s.piece for s in dataset.sequences} - set(pieces))))
        if split != "train":
            with np.load(args.previous_runs / f"{split}_inputs.npz",allow_pickle=False) as stored:
                np.testing.assert_array_equal(inputs[split],stored["original"])
    pieces,keys = identifiers(metadata["train"])
    folds,mapping = work_folds(pieces)
    assert not set(pieces) & {m['piece'] for s in ('validation','test') for m in metadata[s]}
    previous_features = [args.previous_runs / f"{c['arm']}_{c['seed']}_{c['stage']}_{split}.npz"
                         for c in checkpoint_list for split in ("validation","test")]
    sources = [Path(__file__),ai/"src/embedding/temporal_probe.py",ai/"scripts/plot_temporal_probe.py",
               ai/"scripts/evaluate_temporal_order.py",ai/"src/embedding/order_evaluation.py"]
    protocol = dict(task="Frozen representations -> linear prediction of observed four-quarter channel means minus window mean",
        primary="Dynamics: 100*(1-MSE_probe/MSE_flat) on mean-removed four-bin trajectory",
        secondary="Same task for all seven channels; equal five-feature macro (three pedal channels form one feature)",
        bins=4,bin_beats=16,encoder_policy="Saved 64D encoder fixed, eval mode, no synthetic masking, no L2-normalization; no encoder updates or new checkpoint selection",
        checkpoint_stages=list(STAGES),checkpoints=checkpoint_list,
        target="Original [7,64] -> means of each consecutive 16 beats -> subtract each channel's 64-beat mean -> [7,4]",
        probe="One multi-output affine ridge map per representation; 28 outputs. Weighted train-only input mean/std. Objective sum(w*error^2)+alpha*||coeff||^2; intercept unpenalized.",
        penalty_selection=dict(alphas=list(ALPHAS),folds=5,seed=20261010,
            mapping={str(k):int(v) for k,v in mapping.items()},
            metric="Fold-heldout MSE normalized by fold-training channel target energy; equal five features; folds weighted by held-out work count.",
            isolation="All tuning and fold-specific normalization/PCA use only train works. Validation/test targets never used for fitting or penalty choice."),
        baselines=["flat zero trajectory","fixed training-average trajectory","existing summary14 + same ridge",
                   "train-only standardized weighted PCA64 + same ridge","original feature direct target calculation (identity reference, not learned)"],
        pca="Fold-specific/train-only mean/std, weighted covariance top 64 eigenvectors. PCA outputs are standardized again by ridge using its training subset.",
        aggregation="Windows equally within each performance; performances equally within each work; works equally. Channel score is 100*(1-work-macro-MSE/work-macro-flat-MSE), then equal five-feature average.",
        selection="Same fully valid chronological nonoverlapping 64-beat windows as order evaluation; train selection added with identical rule",
        support=supports,model_seeds=list(SEEDS),
        reference_dataset_sha256=sha(args.reference/"dataset.json"),cache_sha256=sha(args.cache),
        previous_protocol_sha256=sha(args.previous_runs/"protocol.json"),
        previous_embedding_files={str(p):sha(p) for p in previous_features},
        input_sha256={s:hashlib.sha256(x.tobytes()).hexdigest() for s,x in inputs.items()},
        sources={str(p.relative_to(ai)):sha(p) for p in sources},
        runtime=dict(torch=str(torch.__version__),numpy=str(np.__version__),device="cpu",threads=2),
        limits=["Linear accessibility of coarse within-window trajectories, not all temporal information or cosine retrieval quality.",
                "Targets derive from the same extracted features; evaluates representation fidelity, not rule correctness or perceived musical similarity.",
                "Prior validation/test have already been observed; not a pristine holdout.",
                "Fully valid subset excludes two test works; original cohort-based normalization remains unchanged.",
                "Different representations have different dimensions/nonlinearity; train-only penalty selection and PCA64/random encoder controls mitigate but do not remove every difference.",
                "Model-seed min/max is not a confidence interval. No user preference claims.",
                "Four-bin means ignore within-16-beat detail and full-performance ordering."])
    args.runs.mkdir(parents=True)
    args.out.mkdir(parents=True)
    dump(args.runs/"protocol.json",protocol)
    dump(args.out/"protocol.json",protocol)
    pd.DataFrame(selected_rows).to_csv(args.out/"selected_windows.csv",index=False)
    for split in inputs:
        np.savez_compressed(args.runs/f"{split}_inputs.npz",values=inputs[split],targets=targets[split])
    methods = [("summary14","summary14","baseline",-1), ("pca64","pca64","baseline",-1)]
    methods += [(f"{c['arm']}_{c['seed']}_{c['stage']}",c['arm'],c['stage'],c['seed']) for c in checkpoint_list]
    summary_rows,channel_rows,work_rows,cv_rows,parameters,examples = [],[],[],[],[],[]
    example_indices = {}
    for split in ("validation","test"):
        current = metadata[split]
        works = sorted({m['piece'] for m in current})
        example_indices[split] = []
        for wi in (0,len(works)//2,len(works)-1):
            keys_in_work = sorted({m['key'] for m in current if m['piece']==works[wi]})
            candidates = [i for i,m in enumerate(current) if m['key']==keys_in_work[len(keys_in_work)//2]]
            example_indices[split].append(candidates[len(candidates)//2])

    def record(method,arm,stage,seed,predictions):
        for split,prediction in predictions.items():
            np.savez_compressed(args.runs/f"prediction_{method}_{split}.npz",prediction=prediction)
            s,c,w = metric_rows(method,arm,stage,seed,targets[split],prediction,metadata[split],split)
            summary_rows.append(s);channel_rows.extend(c);work_rows.extend(w)
            for number,index in enumerate(example_indices[split],1):
                for channel in range(7):
                    for quarter in range(4):
                        examples.append(dict(method=method,arm=arm,stage=stage,seed=seed,split=split,
                            example=number,channel=channel,quarter=quarter,**metadata[split][index],
                            target=float(targets[split][index,channel,quarter]),
                            prediction=float(prediction[index,channel,quarter])))

    train_mean = np.einsum('n,ncq->cq',sample_weights(pieces,keys),targets['train'])
    for method in ("flat","train_mean","raw_reference"):
        prediction = {s:np.zeros_like(targets[s]) if method=='flat' else
                      np.broadcast_to(train_mean,targets[s].shape).copy() if method=='train_mean' else targets[s].copy()
                      for s in ('validation','test')}
        record(method,method,"baseline",-1,prediction)
    for method,arm,stage,seed in methods:
        if arm=='summary14':
            features = {s:summary_vectors(x) for s,x in inputs.items()}
        elif arm=='pca64':
            features = {s:x.reshape(len(x),-1).astype(float) for s,x in inputs.items()}
        else:
            checkpoint = next(c for c in checkpoint_list if c['arm']==arm and c['seed']==seed and c['stage']==stage)
            model,_ = load(checkpoint['path'])
            features = {'train':encode(model,inputs['train']).astype(float)}
            for split in ('validation','test'):
                with np.load(args.previous_runs/f"{method}_{split}.npz",allow_pickle=False) as stored:
                    features[split] = stored['original'].astype(float)
        for split,x in features.items():
            np.savez_compressed(args.runs/f"features_{method}_{split}.npz",features=x)
        model,pca,cv,scores = fit_probe(features['train'],targets['train'],metadata['train'],folds,pca=arm=='pca64')
        np.savez_compressed(args.runs/f"probe_{method}.npz",**model)
        if pca is not None:
            np.savez_compressed(args.runs/f"pca_{method}.npz",**pca)
        prediction = {}
        for split in ('validation','test'):
            x = project_pca(pca,features[split]) if pca is not None else features[split]
            prediction[split] = predict_ridge(model,x)
        record(method,arm,stage,seed,prediction)
        cv_rows.extend(dict(method=method,arm=arm,stage=stage,seed=seed,**row) for row in cv)
        parameters.append(dict(method=method,arm=arm,stage=stage,seed=seed,dimensions=len(model['mean']),
            alpha=float(model['alpha']),cv_normalized_mse=scores[float(model['alpha'])],
            coefficients=int(model['coefficients'].size),intercepts=len(model['intercept'])))
        print(f"Probed {method}: alpha={float(model['alpha']):g}, train-only CV normalized MSE={scores[float(model['alpha'])]:.5f}",flush=True)
    for name,rows in (("summary",summary_rows),("channel_metrics",channel_rows),("work_metrics",work_rows),
                      ("cv",cv_rows),("probe_parameters",parameters),("example_quarters",examples)):
        pd.DataFrame(rows).to_csv(args.out/f"{name}.csv",index=False)
    assert all(sha(c['path'])==c['sha256'] for c in checkpoint_list)
    dump(args.out/"evaluation_audit.json",dict(status="passed",encoder_checkpoints_unchanged=len(checkpoint_list),
         train_eval_works_disjoint=True,heldout_inputs_match_previous_exactly=True,
         no_test_or_validation_tuning=True,methods=len(methods)+3,
         csv_sha256={p.name:sha(p) for p in args.out.glob('*.csv')}))
    from plot_temporal_probe import render
    render(args.out)
    print(pd.DataFrame(summary_rows).groupby(['split','arm','stage'])[['dynamics_score_percent','macro_score_percent']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
