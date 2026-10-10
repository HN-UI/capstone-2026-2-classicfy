"""Train a temporal CNN on existing normalized ASAP features, or reload it to encode."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.data import CHANNELS, WindowDataset, restore_sequences, split_by_piece
from embedding.model import ModelConfig
from embedding.training import (TrainingConfig, cross_work_neighbors, extract_embeddings,
                                fixed_masks, load_model, score_model, train_model)
from validate_embedding import collect_groups
from validate_tempo import chart, plt, save, setup_style, write_csv


def plot_training(out, result):
    import pandas as pd
    setup_style()
    history = pd.read_csv(out / "history.csv")
    fig, ax = chart("Masked reconstruction training", "Validation works and hidden positions stay fixed")
    ax.plot(history.epoch, history.train_loss, label="Train (changing masks)")
    ax.plot(history.epoch, history.validation_loss, label="Validation (fixed masks)")
    ax.axhline(result["baselines"]["linear_interpolation"], color="gray", linestyle="--", label="Linear interpolation")
    ax.set(xlabel="Epoch", ylabel="Mean squared error (five feature blocks)")
    ax.legend()
    save(fig, out / "loss.png", "Epoch 0 is the untrained model. Best checkpoint is selected using validation only.")
    with np.load(out / "reconstruction.npz", allow_pickle=False) as data:
        for channel, name in enumerate(CHANNELS):
            fig, ax = chart(f"Reconstruction: {name}", f"First validation window | {result['example']['key']}")
            beats = result["example"]["start_beat"] + np.arange(data["target"].shape[1])
            target = np.where(data["valid"][channel], data["target"][channel], np.nan)
            ax.plot(beats, target, color="black", label="Actual feature")
            selected = data["hidden"][channel]
            ax.plot(beats[selected], data["initial"][channel, selected], ".", color="gray", label="Untrained prediction")
            ax.plot(beats[selected], data["best"][channel, selected], "o", ms=4, color="#2a78d6", label="Best prediction")
            ax.scatter(beats[selected], target[selected], marker="x", color="#eb6834", label="Hidden target")
            ax.set(xlabel="Original score beat interval", ylabel="Standardized relative feature")
            ax.legend(fontsize=8)
            save(fig, out / f"reconstruction_{name}.png", "Missing and padded beats are excluded. This plot measures reconstruction, not human style similarity.")


def export_embeddings(model, dataset, split, out, batch_size):
    rows, vectors, windows = extract_embeddings(model, dataset, batch_size=batch_size)
    for row in rows:
        row["split"] = split[row["piece"]]
    np.savez_compressed(out / "embeddings.npz", vectors=vectors, window_vectors=windows,
                        keys=np.array([r["key"] for r in rows]),
                        manifest=np.array(json.dumps(rows, ensure_ascii=False)))
    write_csv(out / "embeddings.csv", [dict(row, **{f"z{i}": float(x) for i, x in enumerate(vector)})
                                       for row, vector in zip(rows, vectors)])
    neighbors = cross_work_neighbors(rows, vectors)
    by_key = {r["key"]: r["split"] for r in rows}
    for row in neighbors:
        row["query_split"] = by_key[row["query"]]
        row["candidate_split"] = by_key[row["candidate"]]
    write_csv(out / "neighbors.csv", neighbors)
    write_csv(out / "windows.csv", [{"key": dataset.sequences[si].key, "start_beat": start,
                                     "window_index": i} for i, (si, start) in enumerate(dataset.items)])
    return len(rows)


def main():
    ai = Path(__file__).resolve().parents[1]
    datasets = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "encode"), default="train")
    parser.add_argument("--asap-root", type=Path, default=datasets / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=datasets / "nASAP")
    parser.add_argument("--cache", type=Path, default=datasets / "feature_normalization_raw.npz")
    parser.add_argument("--out", type=Path, default=datasets / "temporal_embedding")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mask-fraction", type=float, default=.2)
    parser.add_argument("--mask-block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-performances", type=int, default=5)
    parser.add_argument("--min-beats", type=int, default=32)
    parser.add_argument("--min-coverage", type=float, default=.5)
    args = parser.parse_args()
    if args.threads < 1 or args.min_performances < 2 or args.min_beats < 2 or not 0 < args.min_coverage <= 1:
        parser.error("Invalid thread/cohort settings")
    try:
        config = TrainingConfig(args.epochs, args.batch_size, args.learning_rate, args.patience,
                                args.mask_fraction, args.mask_block_size, args.seed)
        model_config = ModelConfig(args.window_size, args.hidden_size, args.embedding_size, args.blocks)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "train" and args.checkpoint:
        parser.error("--checkpoint is for --mode encode; training starts from a seeded initialization")
    if args.mode == "encode" and args.checkpoint is None:
        parser.error("--mode encode requires --checkpoint")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("Choose a new/empty --out directory to preserve previous runs")
    torch.set_num_threads(args.threads)
    args.asap_root, args.nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    print("Loading the provenance-checked normalized feature cohort", flush=True)
    groups, audits, scales, provenance = collect_groups(args)
    sequences = restore_sequences(groups, audits)
    cache_hash = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if args.mode == "encode":
        model, checkpoint = load_model(args.checkpoint)
        model_config = ModelConfig(**checkpoint["model_config"])
        stored = json.loads((args.checkpoint.parent / "dataset.json").read_text(encoding="utf-8"))
        if stored["cache_sha256"] != cache_hash or stored["source_provenance"] != provenance:
            parser.error("Checkpoint input reference differs from this corpus")
        if stored["cohort_settings"] != {k: getattr(args, k) for k in stored["cohort_settings"]}:
            parser.error("Use the same --min-performances/--min-beats/--min-coverage as the checkpoint")
        split, args.stride = stored["split"], stored["stride"]
        # This first version re-encodes the same reference cohort, not new MIDI.
    else:
        split = split_by_piece(sequences, seed=config.seed)
    datasets_by_split = {
        name: WindowDataset([s for s in sequences if split[s.piece] == name],
                            window_size=model_config.window_size, stride=args.stride)
        for name in ("train", "validation", "test")}
    if any(not len(dataset) for dataset in datasets_by_split.values()):
        parser.error("Each split needs usable windows; inspect cohort/window settings")
    for name, dataset in datasets_by_split.items():
        print(f"{name}: {len({s.piece for s in dataset.sequences})} works, "
              f"{len(dataset.sequences)} performances, {len(dataset)} windows", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), *sorted((ai / "src/embedding").glob("*.py")),
                    Path(__file__).with_name("validate_embedding.py")]
    metadata = {"source_provenance": provenance, "cache_sha256": cache_hash,
                "code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in source_paths)).hexdigest(),
                "channels": list(CHANNELS), "split": split, "stride": args.stride,
                "normalization": "Existing per-piece/grid cohort common and scale; candidate-included reference; no cross-work fitting",
                "cohort_settings": {k: getattr(args, k) for k in ("min_performances", "min_beats", "min_coverage")},
                "scales": scales, "audit": audits,
                "skipped_keys": {name: ds.skipped_keys for name, ds in datasets_by_split.items()}}
    (args.out / "dataset.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.mode == "train":
        model, result = train_model(datasets_by_split["train"], datasets_by_split["validation"], args.out,
                                    model_config=model_config, config=config)
        plot_training(args.out, result)
        # Test is evaluated once, after choosing the best epoch using validation.
        test = datasets_by_split["test"]
        score = score_model(model, test, fixed_masks(test, config), config)
        (args.out / "test_reconstruction.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    all_windows = WindowDataset(sequences, window_size=model_config.window_size, stride=args.stride)
    count = export_embeddings(model, all_windows, split, args.out, config.batch_size)
    print(f"Exported {count} performance embeddings ({2 * model_config.embedding_size}D) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
