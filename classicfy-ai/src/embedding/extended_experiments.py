"""Frozen-budget experiments, shared mask schedules and resumable CPU training."""

from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from .data import make_hidden_mask
from .model import ModelConfig, TemporalAutoencoder, error_totals, feature_loss
from .sequence_models import SequenceAutoencoder, SequenceConfig
from .training import loader


ARMS = {"cnn64": ("cnn", 64), "bilstm64": ("bilstm", 64),
        "transformer64": ("transformer", 64), "bilstm128": ("bilstm", 128),
        "bilstm256": ("bilstm", 256)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_model(arm, seed, config=ModelConfig()):
    architecture, dimensions = ARMS[arm]
    torch.manual_seed(seed)
    if architecture == "cnn":
        return TemporalAutoencoder(config)
    model = SequenceAutoencoder(config, SequenceConfig(architecture))
    if dimensions != config.embedding_size:
        # Build the 64D reference first, then replace only the dimension-dependent
        # heads. Encoder and decoder output layer have identical initial weights
        # across bottleneck sizes. Default Linear initialization is retained.
        # Isolate head RNG so larger heads do not change the dropout RNG stream.
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000)
            model.bottleneck = nn.Sequential(nn.Flatten(),
                nn.Linear(config.hidden_size * config.window_size, dimensions))
            model.decoder[0] = nn.Linear(dimensions, config.hidden_size * 2)
        model.config = replace(config, embedding_size=dimensions)
    return model


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["extended_format_version"] != 1:
        raise ValueError("Unsupported extended experiment checkpoint")
    # Factory starts from the shared 64D reference before replacing larger heads.
    model = build_model(checkpoint["arm"], checkpoint["seed"],
                        replace(ModelConfig(**checkpoint["model_config"]), embedding_size=64))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def make_schedule(values, valid, config, path, expected_first20=None):
    """Precompute the exact original DataLoader/mask schedule once per seed.

    Reusing these tensors avoids repeating Python mask construction for every
    architecture. No target values determine which positions are hidden.
    """
    path = Path(path)
    signature = hashlib.sha256(values.numpy().tobytes() + valid.numpy().tobytes()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["input_sha256"] != signature or metadata["config"] != asdict(config):
                raise ValueError("Existing schedule differs from inputs/config")
            order, hidden = data["order"].copy(), data["hidden"].copy()
        if expected_first20 and metadata["first20_sha256"] != expected_first20:
            raise ValueError("Schedule differs from original 20 epoch reference")
        return order, hidden, metadata
    dataset = TensorDataset(values, valid, torch.arange(len(values)))
    shuffle_rng = torch.Generator().manual_seed(config.seed + 1)
    mask_rng = torch.Generator().manual_seed(config.seed + 3)
    orders, masks, hashes = [], [], []
    prefix = hashlib.sha256()
    for epoch in range(1, config.epochs + 1):
        order_parts, hidden_parts = [], []
        digest = hashlib.sha256()
        for _, selected_valid, indices in loader(dataset, config, shuffle=True, generator=shuffle_rng):
            hidden = make_hidden_mask(selected_valid, fraction=config.mask_fraction,
                block_size=config.mask_block_size, generator=mask_rng)
            for blob in (indices.numpy().tobytes(), hidden.numpy().tobytes()):
                digest.update(blob)
                if epoch <= 20:
                    prefix.update(blob)
            order_parts.append(indices.numpy()); hidden_parts.append(hidden.numpy())
        orders.append(np.concatenate(order_parts)); masks.append(np.concatenate(hidden_parts))
        hashes.append(digest.hexdigest())
    if expected_first20 and prefix.hexdigest() != expected_first20:
        raise AssertionError("Precomputed first 20 epochs differ from original schedule")
    metadata = {"input_sha256": signature, "config": asdict(config),
                "first20_sha256": prefix.hexdigest(), "epoch_sha256": hashes}
    path.parent.mkdir(parents=True, exist_ok=True)
    order, hidden = np.stack(orders), np.stack(masks)
    np.savez_compressed(path, order=order, hidden=hidden, metadata=np.array(json.dumps(metadata)))
    return order, hidden, metadata


def train(arm, seed, train_values, train_valid, validation, validation_masks,
          schedule, config, model_config, directory, *, score_function, minimum_epochs=20,
          reference_history=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "training.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        for name, expected in result["checkpoint_sha256"].items():
            if sha(directory / f"{name}.pt") != expected:
                raise ValueError("Completed checkpoint changed")
        print(f"Reuse {arm}/{seed}", flush=True)
        return result
    order, masks, schedule_info = schedule
    model = build_model(arm, seed, model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    fields = ("epoch", "train_objective", "validation_value_mse", "validation_delta_mse",
              "best_value_mse", "stale_epochs", "clip_fraction", "seconds", "cumulative_seconds")
    rows, first_epoch, elapsed = [], 1, 0.
    initial = score_function(model, validation, validation_masks, config)
    best, best_epoch, stale = initial["value_mse"], 0, 0
    meta = {"extended_format_version": 1, "arm": arm, "seed": seed,
            "model_config": asdict(model.config), "training_config": asdict(config)}

    def checkpoint(epoch, score):
        return {**meta, "state_dict": model.state_dict(), "epoch": epoch, "score": score}

    last_path = directory / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["arm"] != arm or saved["seed"] != seed or saved["training_config"] != asdict(config):
            raise ValueError("Resume configuration differs")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best, best_epoch, stale = saved["best"], saved["best_epoch"], saved["stale"]
        elapsed = saved["elapsed_seconds"]
        rows = saved["history"]
        first_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["torch_rng_state"])
        print(f"Resume {arm}/{seed} at epoch {first_epoch}", flush=True)
    else:
        if any(directory.iterdir()):
            raise ValueError("Unfinished directory without a resume checkpoint")
        atomic_save(directory / "initial.pt", checkpoint(0, initial))
        atomic_save(directory / "best.pt", checkpoint(0, initial))
        atomic_save(directory / "best20.pt", checkpoint(0, initial))
        rows = [dict(epoch=0, validation_value_mse=best, validation_delta_mse=initial["delta_mse"],
                     best_value_mse=best, stale_epochs=0, cumulative_seconds=0.)]

    # A resumed run may already satisfy the stop rule if interrupted immediately
    # after its last checkpoint. Do not train an extra epoch in that case.
    stopped = first_epoch > minimum_epochs and stale >= config.patience
    for epoch in range(first_epoch, config.epochs + 1):
        if stopped:
            break
        tick = time.perf_counter()
        model.train()
        losses, clipped = [], 0
        actual_schedule = hashlib.sha256()
        for start in range(0, len(train_values), config.batch_size):
            indices = order[epoch-1, start:start+config.batch_size]
            values, valid = train_values[indices], train_valid[indices]
            hidden = torch.from_numpy(masks[epoch-1, start:start+config.batch_size])
            actual_schedule.update(indices.tobytes()); actual_schedule.update(hidden.numpy().tobytes())
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
        if actual_schedule.hexdigest() != schedule_info["epoch_sha256"][epoch-1]:
            raise AssertionError("Actual training order/masks differ from shared schedule")
        score = score_function(model, validation, validation_masks, config)
        if score["value_mse"] < best:
            best, best_epoch, stale = score["value_mse"], epoch, 0
            atomic_save(directory / "best.pt", checkpoint(epoch, score))
        else:
            stale += 1
        if epoch <= 20 and score["value_mse"] < min(r["validation_value_mse"] for r in rows):
            atomic_save(directory / "best20.pt", checkpoint(epoch, score))
        if epoch == 20:
            atomic_save(directory / "epoch20.pt", checkpoint(epoch, score))
        seconds = time.perf_counter() - tick
        elapsed += seconds
        rows.append(dict(epoch=epoch, train_objective=float(np.mean(losses)),
            validation_value_mse=score["value_mse"], validation_delta_mse=score["delta_mse"],
            best_value_mse=best, stale_epochs=stale, clip_fraction=clipped/len(losses),
            seconds=seconds, cumulative_seconds=elapsed))
        atomic_save(last_path, {**checkpoint(epoch, score), "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "best": best, "best_epoch": best_epoch,
            "stale": stale, "elapsed_seconds": elapsed, "history": rows})
        with (directory / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        print(f"{arm} seed={seed} {epoch:02d}/{config.epochs}: value={score['value_mse']:.6f}, best={best:.6f}@{best_epoch}, stale={stale} ({seconds:.1f}s)", flush=True)
        stopped = epoch >= minimum_epochs and stale >= config.patience
    saved = torch.load(last_path, map_location="cpu", weights_only=True)
    atomic_save(directory / "final.pt", {k:v for k,v in saved.items() if k not in
        ("optimizer_state", "torch_rng_state", "history")})
    prefix_error = None
    if reference_history is not None:
        reference = {int(r["epoch"]): float(r["validation_value_mse"]) for r in reference_history}
        differences = [abs(r["validation_value_mse"]-reference[r["epoch"]]) for r in rows if r["epoch"] <= 20]
        prefix_error = max(differences)
        if prefix_error > 2e-6:
            raise AssertionError(f"First 20 epoch scores changed: {prefix_error}")
    result = {"arm": arm, "seed": seed, "dimensions": ARMS[arm][1],
        "best_epoch": best_epoch, "best_validation_mse": best,
        "completed_epochs": saved["epoch"], "maximum_epochs": config.epochs,
        "stop_reason": "patience" if stopped else "maximum_epochs",
        "seconds": elapsed, "first20_max_absolute_error_vs_original": prefix_error,
        "parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": sum(p.numel() for n,p in model.named_parameters()
                                  if not n.startswith(("bottleneck.", "decoder."))),
        "bottleneck_parameters": sum(p.numel() for p in model.bottleneck.parameters()),
        "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
        "schedule_first20_sha256": schedule_info["first20_sha256"],
        "schedule_epoch_sha256": schedule_info["epoch_sha256"][:saved["epoch"]],
        "checkpoint_sha256": {name: sha(directory / f"{name}.pt") for name in
                             ("initial", "best20", "epoch20", "best", "final")}}
    dump(result_path, result)
    return result
