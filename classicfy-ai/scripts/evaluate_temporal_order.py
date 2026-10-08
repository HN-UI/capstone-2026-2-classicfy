"""Evaluate saved embeddings on small noise versus joint beat reordering.

No training, decoder, checkpoint selection or parameter tuning. Primary condition
is noise RMS=5% of window/channel std versus full permutation; others are fixed
sensitivity checks. Large per-window evidence and embeddings stay outside Git.
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
from embedding.extended_experiments import load, sha, dump
from embedding.order_evaluation import (ORDERS, make_variants, summary_vectors,
    summary_distance, raw_distance, cosine_distance, triplet_scores,
    select_complete_nonoverlapping)


ARMS = ("cnn64", "bilstm64", "transformer64")
STAGES = {"initial": "initial", "best20": "best20", "best_extended": "best"}
NOISE = (.02, .05, .10)
TRANSFORM_SEED = 20261009


def encode(model, values, batch_size=64):
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(values[start:start + batch_size])
            encoded.append(model.encode(x, torch.ones_like(x, dtype=torch.bool)).numpy())
    result = np.concatenate(encoded)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite embedding")
    return result


def evaluate_method(metadata, x, variants, distance, representations, *, split, arm, stage, seed):
    records = []
    a = representations["original"]
    for order in ORDERS:
        far = distance(a, representations[order])
        input_far = raw_distance(x, variants[order])
        for noise in NOISE:
            name = f"noise{noise:.2f}"
            near = distance(a, representations[name])
            scores, ties = triplet_scores(near, far)
            # Reversal can leave a constant/symmetric window unchanged. It is
            # not a valid order-change challenge and is excluded for all methods.
            scores[input_far <= 1e-10] = np.nan
            input_near = raw_distance(x, variants[name])
            for i, row in enumerate(metadata):
                records.append(dict(split=split, arm=arm, stage=stage, seed=seed,
                    order=order, noise=noise, **row,
                    d_noise=float(near[i]), d_order=float(far[i]),
                    input_noise_rms=float(input_near[i]), input_order_rms=float(input_far[i]),
                    eligible=bool(input_far[i] > 1e-10), score=float(scores[i]),
                    tie=float(ties[i]) if np.isfinite(scores[i]) else np.nan))
    return records


def aggregate(frame):
    keys = ["split", "arm", "stage", "seed", "order", "noise"]
    valid = frame[frame.score.notna()].copy()
    # Equal performances within each work, equal works within each split.
    performances = valid.groupby(keys + ["piece", "key"], as_index=False).agg(
        score=("score", "mean"), tie=("tie", "mean"),
        d_noise=("d_noise", "mean"), d_order=("d_order", "mean"),
        input_noise_rms=("input_noise_rms", "mean"), input_order_rms=("input_order_rms", "mean"),
        windows=("score", "size"))
    work = performances.groupby(keys + ["piece"], as_index=False).agg(
        score=("score", "mean"), tie=("tie", "mean"),
        d_noise=("d_noise", "mean"), d_order=("d_order", "mean"),
        input_noise_rms=("input_noise_rms", "mean"), input_order_rms=("input_order_rms", "mean"),
        windows=("windows", "sum"), performances=("key", "size"))
    summary = work.groupby(keys, as_index=False).agg(
        score=("score", "mean"), tie=("tie", "mean"),
        d_noise=("d_noise", "mean"), d_order=("d_order", "mean"),
        input_noise_rms=("input_noise_rms", "mean"), input_order_rms=("input_order_rms", "mean"),
        windows=("windows", "sum"), performances=("performances", "sum"), works=("piece", "size"))
    for table in (work, summary):
        table["score_percent"] = table.score * 100
        table["tie_percent"] = table.tie * 100
    pooled = valid.groupby(keys, as_index=False).score.mean().rename(columns={"score": "window_pooled_score"})
    summary = summary.merge(pooled, on=keys, validate="one_to_one")
    return summary, work


def main():
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=data / "temporal_embedding_cluster_run01")
    parser.add_argument("--cache", type=Path, default=data / "feature_normalization_cluster_raw.npz")
    parser.add_argument("--asap-root", type=Path, default=data / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=data / "nASAP")
    parser.add_argument("--models", type=Path, default=data / "extended_sequences_run01")
    parser.add_argument("--runs", type=Path, default=data / "temporal_order_evaluation_run01")
    parser.add_argument("--out", type=Path, default=ai / "analysis/temporal_order_evaluation")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        from plot_temporal_order_evaluation import render
        render(args.out)
        return
    if args.runs.exists() or args.out.exists():
        raise ValueError("Preserve existing outputs: use new --runs/--out, or --report-only")
    torch.set_num_threads(2)
    prepared = prepare(args)
    checkpoint_paths, checkpoints = {}, []
    model_protocol = json.loads((args.models / "protocol.json").read_text())
    for source, expected in model_protocol["sources"].items():
        if sha(ai / source) != expected:
            raise ValueError(f"Saved-model experiment source changed: {source}")
    for arm in ARMS:
        for seed in SEEDS:
            directory = args.models / arm / str(seed)
            training = json.loads((directory / "training.json").read_text())
            for stage, file in STAGES.items():
                path = directory / f"{file}.pt"
                digest = sha(path)
                if digest != training["checkpoint_sha256"][file]:
                    raise ValueError(f"Saved checkpoint changed: {path}")
                checkpoint_paths[arm, seed, stage] = path
                checkpoints.append(dict(arm=arm, seed=seed, stage=stage,
                                        path=str(path.resolve()), sha256=digest))
    all_inputs, selected_rows, support = {}, [], []
    for split in ("validation", "test"):
        dataset = prepared[4][split]
        values, valid = prepared[5][split]
        selected = select_complete_nonoverlapping(dataset, valid.numpy())
        if not len(selected):
            raise ValueError(f"No complete windows in {split}")
        x = values.numpy()[selected].copy()
        variants, permutations = make_variants(x, seed=TRANSFORM_SEED + (split == "test"))
        metadata = []
        for i, index in enumerate(selected):
            si, start = dataset.items[index]
            sequence = dataset.sequences[si]
            row = dict(window=int(index), piece=sequence.piece, key=sequence.key, start=int(start))
            metadata.append(row)
            selected_rows.append(dict(split=split, **row))
        kept_works = {row["piece"] for row in metadata}
        support.append(dict(split=split, total_works=len({s.piece for s in dataset.sequences}),
            total_performances=len(dataset.sequences), total_windows=len(dataset),
            full_windows=int(valid.all(dim=1).all(dim=1).sum()), selected_windows=len(selected),
            selected_performances=len({row["key"] for row in metadata}), selected_works=len(kept_works),
            excluded_works=sorted({s.piece for s in dataset.sequences} - kept_works)))
        all_inputs[split] = x, variants, permutations, metadata
    sources = [Path(__file__), ai / "src/embedding/order_evaluation.py",
               ai / "scripts/plot_temporal_order_evaluation.py"]
    protocol = dict(task="d(A, small_noise)<d(A, joint_temporal_reordering); tie gets 0.5",
        primary=dict(noise=.05, order="shuffle", stage="best_extended"),
        noise_levels=list(NOISE), orders=list(ORDERS), perturbation_seed=TRANSFORM_SEED,
        seed_policy="Validation uses perturbation_seed; test uses seed+1. Same transformations for every representation/model seed.",
        noise="Independent Gaussian direction per channel, centered and RMS-normalized, scaled by each original window/channel std; constant channels unchanged. Same direction across levels.",
        permutations="All 7 channels move jointly; full random permutation, shuffle eight 8-beat blocks, reversal, swap each adjacent beat pair.",
        selection="Original fixed 64-beat windows. Fully valid only; greedy chronological nonoverlapping windows per performance. No value-based selection.",
        zero_change="Exclude an order condition for a window when feature-weighted input RMS change <=1e-10, identically for all methods.",
        distances=dict(summary14="Existing 14D feature-weighted RMS (mean, Rubato median abs, p95-p5); sorted reduction ensures exact permutation invariance.",
                       raw="Feature-weighted RMS over all 64 ordered beats", networks="Cosine distance on unmasked 64D window embeddings; dropout off"),
        aggregation="Window score -> equal performances within work -> equal works within split. Then report three model-seed mean/min/max. No independent-window confidence intervals.",
        tie_absolute_tolerance=1e-8, arms=list(ARMS), stages=STAGES, model_seeds=list(SEEDS),
        checkpoint_selection="Reuse initial, first-20-epoch reconstruction-best, extended-budget reconstruction-best checkpoints. No new selection/training.",
        support=support, checkpoints=checkpoints,
        reference_dataset_sha256=sha(args.reference / "dataset.json"), cache_sha256=sha(args.cache),
        model_experiment_protocol_sha256=sha(args.models / "protocol.json"),
        sources={str(p.relative_to(ai)):sha(p) for p in sources},
        input_sha256={s:hashlib.sha256(v[0].tobytes()).hexdigest() for s,v in all_inputs.items()},
        runtime=dict(torch=str(torch.__version__), numpy=str(np.__version__), device="cpu", threads=2),
        limits=["Synthetic feature perturbations, not different actual performances or audio.",
                "Large permutation vs tiny noise is easy for raw distance/random encoders; a high score alone does not demonstrate learned musical structure.",
                "Different methods use different distance definitions; only within-method rankings/score rates are compared.",
                "One perturbation realization per window/order; model-seed ranges are not transformation uncertainty or confidence intervals.",
                "Validation and test works have been observed in earlier experiments; this is not a pristine holdout.",
                "Fully valid subset excludes works/performances with gaps. Reference-cohort normalization is unchanged.",
                "Window-level evaluation does not validate whole-performance pooled embedding or long-range ordering.",
                "No human preference or real cross-performance similarity claim."])
    args.runs.mkdir(parents=True)
    args.out.mkdir(parents=True)
    # Freeze settings before any model evaluation.
    dump(args.runs / "protocol.json", protocol)
    dump(args.out / "protocol.json", protocol)
    pd.DataFrame(selected_rows).to_csv(args.out / "selected_windows.csv", index=False)
    records, examples = [], []
    for split, (x, variants, permutations, metadata) in all_inputs.items():
        np.savez_compressed(args.runs / f"{split}_inputs.npz", original=x, **variants,
                            **{f"permutation_{k}":v for k,v in permutations.items()})
        representations = dict(original=x, **variants)
        for arm, distance, rep in (("raw", raw_distance, representations),
            ("summary14", summary_distance, {k:summary_vectors(v) for k,v in representations.items()})):
            records.extend(evaluate_method(metadata, x, variants, distance, rep,
                                           split=split, arm=arm, stage="baseline", seed=-1))
        # Fixed first/middle/last work; middle performance, middle selected window.
        pieces = sorted({m["piece"] for m in metadata})
        for number, pi in enumerate((0, len(pieces)//2, len(pieces)-1), 1):
            keys = sorted({m["key"] for m in metadata if m["piece"] == pieces[pi]})
            chosen = [i for i,m in enumerate(metadata) if m["key"] == keys[len(keys)//2]]
            i = chosen[len(chosen)//2]
            for ci in range(7):
                for beat in range(64):
                    examples.append(dict(split=split, example=number, channel=ci, beat=beat,
                        **metadata[i], original=float(x[i,ci,beat]),
                        noise=float(variants["noise0.05"][i,ci,beat]),
                        shuffle=float(variants["shuffle"][i,ci,beat]),
                        block8=float(variants["block8"][i,ci,beat]),
                        reverse=float(variants["reverse"][i,ci,beat]),
                        adjacent=float(variants["adjacent"][i,ci,beat])))
    for arm in ARMS:
        for seed in SEEDS:
            for stage in STAGES:
                model, checkpoint = load(checkpoint_paths[arm,seed,stage])
                for split, (x, variants, _, metadata) in all_inputs.items():
                    representations = {name:encode(model, value) for name,value in dict(original=x, **variants).items()}
                    np.savez_compressed(args.runs / f"{arm}_{seed}_{stage}_{split}.npz", **representations)
                    records.extend(evaluate_method(metadata, x, variants, cosine_distance, representations,
                        split=split, arm=arm, stage=stage, seed=seed))
                print(f"Evaluated {arm} / {seed} / {stage} (epoch {checkpoint['epoch']})", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.runs / "window_metrics.csv", index=False)
    summary, works = aggregate(frame)
    summary.to_csv(args.out / "summary.csv", index=False)
    works.to_csv(args.out / "work_metrics.csv", index=False)
    pd.DataFrame(examples).to_csv(args.out / "example_beats.csv", index=False)
    audit = dict(status="passed", rows=len(frame), undefined_eligible_rows=int((frame.eligible & frame.score.isna()).sum()),
        exact_permutation_summary_invariance=all(np.array_equal(summary_vectors(x),summary_vectors(variants[o]))
            for x,variants,_,_ in all_inputs.values() for o in ORDERS),
        checkpoint_integrity=all(sha(Path(c['path'])) == c['sha256'] for c in checkpoints),
        per_window_csv=str((args.runs / "window_metrics.csv").resolve()),
        output_sha256={p.name:sha(p) for p in args.out.glob("*.csv")})
    if not audit["exact_permutation_summary_invariance"] or not audit["checkpoint_integrity"] or audit["undefined_eligible_rows"]:
        audit["status"] = "failed"
    dump(args.out / "audit.json", audit)
    if audit["status"] != "passed":
        raise ValueError("Evaluation audit failed")
    from plot_temporal_order_evaluation import render
    render(args.out)
    primary = summary[(summary.noise == .05) & (summary.order == "shuffle")]
    print(primary.groupby(["split","arm","stage"]).score_percent.agg(["mean","min","max"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
