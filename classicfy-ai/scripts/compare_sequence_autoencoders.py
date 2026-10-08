"""Train BiLSTM first, then Transformer; evaluate a frozen CNN comparison protocol.

Run from repository root with classicfy-ai/.venv/bin/python. Large artifacts go
outside Git to datasets/sequence_autoencoders_run01. Completed runs are reused
only when their protocol and actual training schedule match the reference.
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.data import CHANNELS, WindowDataset, restore_sequences
from embedding.model import ModelConfig, error_totals, feature_loss
from embedding.sequence_models import (SequenceAutoencoder, SequenceConfig,
                                       load_sequence_model, save_sequence_model)
from embedding.training import (TrainingConfig, fixed_masks, loader,
                                extract_embeddings, cross_work_neighbors)
from embedding.evaluation import baseline_predictions
from validate_embedding import collect_groups, summarize
from validate_temporal_embedding import predictions, feature_average, select_examples
from train_temporal_ablation import evaluate_loss, load as load_cnn, channel_metrics
from analyze_temporal_failures import sha
from observe_temporal_neighbors import block_distances, compare_candidate, step_summary

SEEDS = (20261006, 20261007, 20261008)
METRICS = ("mse", "shape_correlation", "shape_amplitude_ratio", "slope_mse", "direction_percent")
REFERENCE_COMMIT = "20a4168"


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare(args):
    stored = json.loads((args.reference / "dataset.json").read_text())
    info = json.loads((args.reference / "training.json").read_text())
    if sha(args.cache) != stored["cache_sha256"]:
        raise ValueError("Feature cache differs from the CNN reference")
    # The merged develop adds four unrelated ATEPP collectors to preprocessing/.
    # Verify EVERY original module (including __init__) against the reference
    # Git tree; use its original hash inventory, without altering cache metadata.
    repo = Path(__file__).resolve().parents[2]
    inventory = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REFERENCE_COMMIT,
        "classicfy-ai/src/features", "classicfy-ai/src/preprocessing"], cwd=repo, text=True).splitlines()
    original = [p for p in inventory if p.endswith(".py")]
    for name in original:
        expected = subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:{name}"], cwd=repo)
        if (repo / name).read_bytes() != expected:
            raise ValueError(f"Original feature/preprocessing source changed: {name}")
    source_files = [repo / p for p in original if "/features/" in p and Path(p).name not in {"normalization.py", "__init__.py"}]
    source_files += [repo / p for p in original if "/preprocessing/" in p]
    groups, audits, scales, provenance = collect_groups(SimpleNamespace(
        asap_root=args.asap_root.resolve(), nasap_root=args.nasap_root.resolve(),
        cache=args.cache, **stored["cohort_settings"]), provenance_source_files=source_files)
    if provenance != stored["source_provenance"] or scales != stored["scales"]:
        raise ValueError("Feature provenance or normalization differs from reference")
    sequences = restore_sequences(groups, audits)
    datasets = {split: WindowDataset([s for s in sequences if stored["split"][s.piece] == split],
                    window_size=64, stride=stored["stride"]) for split in ("train", "validation", "test")}
    config = TrainingConfig(**info["training_config"])
    arrays, masks = {}, {}
    for split, dataset in datasets.items():
        batches = list(DataLoader(dataset, batch_size=config.batch_size))
        arrays[split] = (torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]))
        masks[split] = fixed_masks(dataset, config)
    print({s: len(d) for s, d in datasets.items()}, flush=True)
    return stored, ModelConfig(**info["model_config"]), config, sequences, datasets, arrays, masks


def train(train, validation, masks, config, model_config, architecture, directory, cnn_result):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "training.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["training_schedule_sha256"] != cnn_result["training_schedule_sha256"]:
            raise ValueError("Cached training schedule differs")
        for name in ("initial", "best_value", "final"):
            if sha(directory / f"{name}.pt") != result["checkpoint_sha256"][name]:
                raise ValueError("Cached checkpoint changed")
        print(f"Reusing {architecture}/{config.seed}", flush=True)
        return result
    if any(directory.iterdir()):
        raise ValueError(f"Preserve unfinished run and choose a new --runs: {directory}")
    torch.manual_seed(config.seed)
    model = SequenceAutoencoder(model_config, SequenceConfig(architecture))
    initial = evaluate_loss(model, validation, masks, config)
    meta = {"seed": config.seed, "training_config": asdict(config)}
    save_sequence_model(directory / "initial.pt", model, epoch=0, score=initial, **meta)
    save_sequence_model(directory / "best_value.pt", model, epoch=0, score=initial, **meta)
    best, best_epoch = initial["value_mse"], 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    shuffle_rng = torch.Generator().manual_seed(config.seed + 1)
    mask_rng = torch.Generator().manual_seed(config.seed + 3)
    schedule = hashlib.sha256()
    start_time = time.perf_counter()
    fields = ("epoch", "train_objective", "validation_value_mse", "validation_delta_mse", "clip_fraction", "seconds")
    with (directory / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(epoch=0, validation_value_mse=best, validation_delta_mse=initial["delta_mse"]))
        for epoch in range(1, config.epochs + 1):
            tick = time.perf_counter()
            model.train()
            losses, clipped = [], 0
            for values, valid, indices in loader(train, config, shuffle=True, generator=shuffle_rng):
                # Match the existing CNN schedule byte-for-byte, independent of
                # dropout RNG consumption in the alternative architecture.
                from embedding.data import make_hidden_mask
                hidden = make_hidden_mask(valid, fraction=config.mask_fraction,
                    block_size=config.mask_block_size, generator=mask_rng)
                schedule.update(indices.numpy().tobytes())
                schedule.update(hidden.numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(values, valid, hidden)
                loss = feature_loss(*error_totals(prediction, values, hidden))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                clipped += int(norm > 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            score = evaluate_loss(model, validation, masks, config)
            seconds = time.perf_counter() - tick
            writer.writerow(dict(epoch=epoch, train_objective=float(np.mean(losses)),
                validation_value_mse=score["value_mse"], validation_delta_mse=score["delta_mse"],
                clip_fraction=clipped / len(losses), seconds=seconds))
            file.flush()
            if score["value_mse"] < best:
                best, best_epoch = score["value_mse"], epoch
                save_sequence_model(directory / "best_value.pt", model, epoch=epoch, score=score, **meta)
            print(f"{architecture} seed={config.seed} epoch={epoch:02d}: value={score['value_mse']:.6f}, change={score['delta_mse']:.6f} ({seconds:.1f}s)", flush=True)
    save_sequence_model(directory / "final.pt", model, epoch=config.epochs, score=score, **meta)
    if schedule.hexdigest() != cnn_result["training_schedule_sha256"]:
        raise AssertionError("CNN and sequence model saw different training masks/order")
    result = dict(architecture=architecture, seed=config.seed, best_value_epoch=best_epoch,
        best_validation_value_mse=best, final_validation=score, epochs=config.epochs,
        seconds=time.perf_counter() - start_time,
        parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(p.numel() for n, p in model.named_parameters() if not n.startswith("decoder.")),
        decoder_parameters=sum(p.numel() for p in model.decoder.parameters()),
        training_schedule_sha256=schedule.hexdigest(),
        checkpoint_sha256={name: sha(directory / f"{name}.pt") for name in ("initial", "best_value", "final")})
    dump(result_path, result)
    return result


def metric_rows(model_name, seed, split, condition, target, prediction, hidden, works):
    channels = channel_metrics(target, prediction, hidden)
    summary = dict(model=model_name, seed=seed, split=split, condition=condition,
                   **{metric: feature_average(channels, metric) for metric in METRICS})
    detail = [dict(model=model_name, seed=seed, split=split, condition=condition, **row) for row in channels]
    by_work = []
    for piece in sorted(set(works)):
        keep = works == piece
        current = channel_metrics(target[keep], prediction[keep], hidden[keep])
        by_work.append(dict(model=model_name, seed=seed, split=split, condition=condition, piece=piece,
                           **{metric: feature_average(current, metric) for metric in METRICS}))
    return summary, detail, by_work


def evaluate(args, models, prepared, *, include_test=True):
    stored, mc, config, sequences, datasets, arrays, masks = prepared
    summaries, details, work_rows, histories, training_rows, examples = [], [], [], [], [], []
    validation = datasets["validation"]
    example_indices = select_examples(validation)
    example_predictions = {}
    splits = ("validation", "test") if include_test else ("validation",)
    for split in splits:
        x, v = arrays[split]
        works = np.array([datasets[split].sequences[si].piece for si, _ in datasets[split].items])
        for name, prediction in baseline_predictions(x.numpy(), v.numpy(), masks[split].numpy()).items():
            summary, detail, by_work = metric_rows(name, -1, split, "best", x.numpy(), prediction, masks[split].numpy(), works)
            summaries.append(summary); details.extend(detail); work_rows.extend(by_work)
            if split == "validation":
                example_predictions[name] = prediction[example_indices]
    for name in models:
        for seed in SEEDS:
            directory = args.cnn_runs / str(seed) / "baseline" if name == "cnn" else args.runs / name / str(seed)
            load = load_cnn if name == "cnn" else load_sequence_model
            result = json.loads((directory / "training.json").read_text())
            training_rows.append(dict(model=name, **result))
            histories.append(pd.read_csv(directory / "history.csv").assign(model=name, seed=seed))
            for split in splits:
                x, v = arrays[split]
                dataset = datasets[split]
                works = np.array([dataset.sequences[si].piece for si, _ in dataset.items])
                # Best epoch is chosen only from validation. Test is evaluated
                # once per frozen model/seed, without tuning after its result.
                conditions = ("initial", "final", "best", "shuffle0", "shuffle1", "shuffle2", "gap1", "gap8", "gap16") if split == "validation" else ("best",)
                for condition in conditions:
                    file = "best_value.pt" if condition not in ("initial", "final") else f"{condition}.pt"
                    model, _ = load(directory / file)
                    hidden = masks[split]
                    if condition.startswith("gap"):
                        hidden = fixed_masks(dataset, replace(config, mask_block_size=int(condition[3:])))
                    shuffle_seed = config.seed + 11 + int(condition[-1]) if condition.startswith("shuffle") else None
                    pred = predictions(model, x, v, hidden, batch_size=config.batch_size, shuffle_seed=shuffle_seed)
                    summary, detail, by_work = metric_rows(name, seed, split, condition, x.numpy(), pred, hidden.numpy(), works)
                    summaries.append(summary); details.extend(detail); work_rows.extend(by_work)
                    if split == "validation" and seed == SEEDS[0] and condition == "best":
                        example_predictions[name] = pred[example_indices]
                    if split == "validation" and condition.startswith("gap") and name == "cnn" and seed == SEEDS[0]:
                        for baseline, bp in baseline_predictions(x.numpy(), v.numpy(), hidden.numpy()).items():
                            s, d, w = metric_rows(baseline, -1, split, condition, x.numpy(), bp, hidden.numpy(), works)
                            summaries.append(s); details.extend(d); work_rows.extend(w)
    for position, index in enumerate(example_indices):
        si, start = validation.items[index]
        for ci, channel in enumerate(CHANNELS):
            for beat in range(mc.window_size):
                examples.append(dict(example=position + 1, key=validation.sequences[si].key,
                    piece=validation.sequences[si].piece, channel=channel, beat=start + beat,
                    valid=bool(arrays["validation"][1][index, ci, beat]),
                    hidden=bool(masks["validation"][index, ci, beat]),
                    target=float(arrays["validation"][0][index, ci, beat]),
                    **{name: float(p[position, ci, beat]) for name, p in example_predictions.items()}))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summaries), ("channel_metrics", details), ("work_metrics", work_rows), ("example_beats", examples)):
        pd.DataFrame(rows).to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(args.out / "history.csv", index=False)
    dump(args.out / "training_runs.json", training_rows)
    if include_test:
        retrieval(args, models, prepared)
    from plot_sequence_autoencoders import render
    render(args.out, models)
    print(pd.DataFrame(summaries).query("split == 'validation' and condition == 'best'")[["model", "seed", *METRICS]].to_string(index=False), flush=True)


def retrieval(args, models, prepared):
    stored, mc, config, sequences, datasets, arrays, masks = prepared
    all_windows = WindowDataset(sequences, window_size=mc.window_size, stride=stored["stride"])
    query_rows, neighbor_rows, descriptors = [], [], None
    for name in models:
        for seed in SEEDS:
            directory = args.cnn_runs / str(seed) / "baseline" if name == "cnn" else args.runs / name / str(seed)
            load = load_cnn if name == "cnn" else load_sequence_model
            model, _ = load(directory / "best_value.pt")
            rows, vectors, _ = extract_embeddings(model, all_windows, batch_size=config.batch_size)
            for row in rows:
                row["split"] = stored["split"][row["piece"]]
            np.savez_compressed(args.runs / f"{name}_{seed}_embeddings.npz", vectors=vectors,
                                keys=np.array([r["key"] for r in rows]))
            if descriptors is None:
                by_key = {s.key: s for s in sequences}
                compact = [summarize(by_key[r["key"]].values[:, by_key[r["key"]].valid.all(axis=0)]).reshape(7, 2) for r in rows]
                descriptors = {"representative": np.stack([v[:, 0] for v in compact]),
                               "width": np.stack([v[:, 1] for v in compact]),
                               "step_rms": np.stack([step_summary(by_key[r["key"]])[0] for r in rows])}
                distances = {key: block_distances(value) for key, value in descriptors.items()}
                pd.DataFrame([dict(key=r["key"], piece=r["piece"], split=r["split"],
                                  **{f"{key}_{channel}": float(values[i, ci]) for key, values in descriptors.items() for ci, channel in enumerate(CHANNELS)}) for i, r in enumerate(rows)]).to_csv(args.out / "feature_summaries.csv", index=False)
                row_keys = [r["key"] for r in rows]
            if [r["key"] for r in rows] != row_keys:
                raise ValueError("Embedding row alignment changed")
            neighbors = cross_work_neighbors(rows, vectors)
            by_key_index = {r["key"]: i for i, r in enumerate(rows)}
            pieces = np.array([r["piece"] for r in rows])
            for n in neighbors:
                neighbor_rows.append(dict(model=name, seed=seed, query_split=stored["split"][n["query_piece"]],
                    candidate_split=stored["split"][n["candidate_piece"]], **n))
                if n["rank"] != 1 or stored["split"][n["query_piece"]] != "test":
                    continue
                qi, ni = by_key_index[n["query"]], by_key_index[n["candidate"]]
                for descriptor, dist in distances.items():
                    query_rows.append(dict(model=name, seed=seed, descriptor=descriptor, query=n["query"],
                        query_piece=n["query_piece"], candidate=n["candidate"],
                        **compare_candidate(dist, pieces, qi, ni)))
    pd.DataFrame(neighbor_rows).to_csv(args.out / "neighbors.csv", index=False)
    pd.DataFrame(query_rows).to_csv(args.out / "retrieval_queries.csv", index=False)


def main():
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=data / "temporal_embedding_cluster_run01")
    parser.add_argument("--cnn-runs", type=Path, default=data / "temporal_controlled_run01")
    parser.add_argument("--runs", type=Path, default=data / "sequence_autoencoders_run01")
    parser.add_argument("--out", type=Path, default=ai / "analysis/sequence_autoencoders")
    parser.add_argument("--cache", type=Path, default=data / "feature_normalization_cluster_raw.npz")
    parser.add_argument("--asap-root", type=Path, default=data / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=data / "nASAP")
    parser.add_argument("--stage", choices=("bilstm", "transformer", "all", "evaluate"), default="all")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive threads required")
    torch.set_num_threads(args.threads)
    prepared = prepare(args)
    stored, mc, config, sequences, datasets, arrays, masks = prepared
    source_paths = [Path(__file__), ai / "src/embedding/sequence_models.py",
                    ai / "src/embedding/model.py", ai / "src/embedding/data.py",
                    ai / "src/embedding/evaluation.py", ai / "src/embedding/training.py",
                    ai / "scripts/train_temporal_ablation.py", ai / "scripts/validate_temporal_embedding.py",
                    ai / "scripts/validate_embedding.py", ai / "scripts/validate_feature_normalization.py"]
    protocol = dict(seeds=list(SEEDS), training_config=asdict(config), model_config=asdict(mc),
        sequence_configs={a: asdict(SequenceConfig(a)) for a in ("bilstm", "transformer")},
        reference_dataset_sha256=sha(args.reference / "dataset.json"), cache_sha256=sha(args.cache),
        original_collector_commit=REFERENCE_COMMIT,
        collector_compatibility="All original feature/preprocessing .py files byte-identical to reference; original inventory hash and data revisions verified; unrelated ATEPP additions excluded from original inventory only.",
        source_sha256={str(p.relative_to(ai)): sha(p) for p in source_paths},
        primary="validation-selected best checkpoint; final epoch shown separately; same 20 epoch budget",
        diagnostics="three fixed visible-order permutations and mask block sizes 1/4/8/16 on validation only",
        test="descriptive frozen-protocol evaluation; previously used test split, not a new independent holdout",
        evaluation_masks_sha256={s: hashlib.sha256(m.numpy().tobytes()).hexdigest() for s, m in masks.items()},
        limitations=["One work split, three seeds; seed ranges are not confidence intervals.",
            "Encoder parameter counts differ; not an equal-parameter comparison.",
            "Fixed optimizer and 20 epochs; no claim each architecture is optimally tuned.",
            "Candidate-included per-work references; not standalone unseen-MIDI inference.",
            "Window means/std aggregation loses global window order.",
            "Feature agreement is not listening similarity or retrieval accuracy."],
        counts={s:dict(works=len({v.piece for v in d.sequences}), performances=len(d.sequences), windows=len(d)) for s,d in datasets.items()})
    args.runs.mkdir(parents=True, exist_ok=True)
    protocol_path = args.runs / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol/source changed: preserve existing runs and use a new --runs")
    dump(protocol_path, protocol)
    for architecture in ("bilstm", "transformer"):
        if args.stage not in (architecture, "all"):
            continue
        for seed in SEEDS:
            cnn_result = json.loads((args.cnn_runs / str(seed) / "baseline/training.json").read_text())
            if cnn_result["epochs"] != config.epochs or sha(args.cnn_runs / str(seed) / "baseline/final.pt") != cnn_result["final_sha256"]:
                raise ValueError("CNN reference budget or checkpoint changed")
            train(datasets["train"], datasets["validation"], masks["validation"], replace(config, seed=seed),
                  mc, architecture, args.runs / architecture / str(seed), cnn_result)
        # Test/search evaluation is deferred until both model attempts finish.
        print(f"Completed {architecture} training; validation results stored per seed.", flush=True)
        if architecture == "bilstm":
            original_out = args.out
            args.out = original_out / "bilstm_first_validation"
            evaluate(args, ("cnn", "bilstm"), prepared, include_test=False)
            dump(args.out / "protocol.json", protocol)
            args.out = original_out
    if args.stage in ("all", "evaluate"):
        evaluate(args, ("cnn", "bilstm", "transformer"), prepared)
        dump(args.out / "protocol.json", protocol)


if __name__ == "__main__":
    main()
