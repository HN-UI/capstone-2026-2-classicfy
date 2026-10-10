"""CPU-first masked training and inference from the saved best checkpoint."""

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CHANNELS, make_hidden_mask
from .model import ModelConfig, TemporalAutoencoder, error_totals, feature_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = .001
    patience: int = 5
    mask_fraction: float = .2
    mask_block_size: int = 4
    seed: int = 20261006

    def __post_init__(self):
        if (min(self.epochs, self.batch_size, self.patience, self.mask_block_size) < 1
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not 0 < self.mask_fraction < 1):
            raise ValueError("Invalid training settings")


def fixed_masks(dataset, config):
    generator = torch.Generator().manual_seed(config.seed + 2)
    return torch.cat([make_hidden_mask(dataset[i][1][None], fraction=config.mask_fraction,
                     block_size=config.mask_block_size, generator=generator) for i in range(len(dataset))])


def loader(dataset, config, *, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def score_model(model, dataset, masks, config):
    model.eval()
    sums, counts = torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.int64)
    with torch.no_grad():
        for values, valid, indices in loader(dataset, config):
            prediction, _ = model(values, valid, masks[indices])
            errors, support = error_totals(prediction, values, masks[indices])
            sums += errors.double()
            counts += support
    loss = float(feature_loss(sums, counts))
    if not np.isfinite(loss):
        raise ValueError("Nonfinite validation loss")
    return {"loss": loss, **{name: float(sums[i] / counts[i]) if counts[i] else None
                             for i, name in enumerate(CHANNELS)}}


def score_baselines(dataset, masks):
    totals = {name: (torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.int64))
              for name in ("zero", "linear_interpolation")}
    for i in range(len(dataset)):
        values, valid, _ = dataset[i]
        hidden = masks[i]
        observed = valid & ~hidden
        interpolated = np.zeros_like(values.numpy())
        for channel in range(7):
            positions = np.flatnonzero(observed[channel].numpy())
            if len(positions):
                interpolated[channel] = np.interp(np.arange(values.shape[1]), positions,
                                                  values[channel, positions].numpy())
        for name, prediction in (("zero", torch.zeros_like(values)),
                                 ("linear_interpolation", torch.from_numpy(interpolated))):
            errors, counts = error_totals(prediction[None], values[None], hidden[None])
            totals[name][0].add_(errors.double())
            totals[name][1].add_(counts)
    return {name: float(feature_loss(*total)) for name, total in totals.items()}


def save_checkpoint(path, model, model_config, training_config, epoch, validation_loss):
    torch.save({"format_version": 1, "model_config": asdict(model_config),
                "training_config": asdict(training_config), "channels": list(CHANNELS),
                "epoch": epoch, "validation_loss": validation_loss,
                "state_dict": model.state_dict()}, path)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1 or checkpoint["channels"] != list(CHANNELS):
        raise ValueError("Unsupported checkpoint/channel order")
    model = TemporalAutoencoder(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train_model(train, validation, out, *, model_config=ModelConfig(), config=TrainingConfig()):
    if not len(train) or not len(validation):
        raise ValueError("Training and validation both need usable windows")
    if model_config.window_size != train.window_size or train.window_size != validation.window_size:
        raise ValueError("Window/model sizes differ")
    if {s.piece for s in train.sequences} & {s.piece for s in validation.sequences}:
        raise ValueError("Training and validation works overlap")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    model = TemporalAutoencoder(model_config)
    masks = fixed_masks(validation, config)
    baseline = score_baselines(validation, masks)
    initial_score = score_model(model, validation, masks, config)
    initial_values, initial_valid, _ = validation[0]
    with torch.no_grad():
        initial_prediction = model(initial_values[None], initial_valid[None], masks[:1])[0][0].numpy()
    save_checkpoint(out / "initial.pt", model, model_config, config, 0, initial_score["loss"])
    save_checkpoint(out / "best.pt", model, model_config, config, 0, initial_score["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    shuffle_rng = torch.Generator().manual_seed(config.seed + 1)
    mask_rng = torch.Generator().manual_seed(config.seed + 3)
    best, best_epoch, stale = initial_score["loss"], 0, 0
    history = [{"epoch": 0, "train_loss": None, "validation_loss": best,
                **{f"validation_{k}": v for k, v in initial_score.items() if k != "loss"}}]
    fields = list(history[0])
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(history[0])
        handle.flush()
        for epoch in range(1, config.epochs + 1):
            model.train()
            sums, counts = torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.int64)
            for values, valid, _ in loader(train, config, shuffle=True, generator=shuffle_rng):
                hidden = make_hidden_mask(valid, fraction=config.mask_fraction,
                                          block_size=config.mask_block_size, generator=mask_rng)
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(values, valid, hidden)
                errors, support = error_totals(prediction, values, hidden)
                loss = feature_loss(errors, support)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                sums += errors.detach().double()
                counts += support
            score = score_model(model, validation, masks, config)
            row = {"epoch": epoch, "train_loss": float(feature_loss(sums, counts)),
                   "validation_loss": score["loss"],
                   **{f"validation_{k}": v for k, v in score.items() if k != "loss"}}
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(f"Epoch {epoch}: train={row['train_loss']:.6f}, validation={score['loss']:.6f}", flush=True)
            if score["loss"] < best:
                best, best_epoch, stale = score["loss"], epoch, 0
                save_checkpoint(out / "best.pt", model, model_config, config, epoch, best)
            else:
                stale += 1
            if stale >= config.patience:
                break
    model, _ = load_model(out / "best.pt")
    with torch.no_grad():
        final_prediction = model(initial_values[None], initial_valid[None], masks[:1])[0][0].numpy()
    np.savez_compressed(out / "reconstruction.npz", target=initial_values.numpy(),
                        valid=initial_valid.numpy(), hidden=masks[0].numpy(),
                        initial=initial_prediction, best=final_prediction)
    si, start = validation.items[0]
    result = {"initial_validation": initial_score, "best_validation": score_model(model, validation, masks, config),
              "best_epoch": best_epoch, "completed_epochs": len(history) - 1, "baselines": baseline,
              "example": {"key": validation.sequences[si].key, "start_beat": start},
              "model_config": asdict(model_config), "training_config": asdict(config),
              "channels": list(CHANNELS), "torch_version": str(torch.__version__), "device": "cpu"}
    (out / "training.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return model, result


def extract_embeddings(model, dataset, *, batch_size=32):
    """No synthetic masks at inference; aggregate real windows per performance."""
    model.eval()
    windows = []
    with torch.no_grad():
        for values, valid, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            windows.append(model.encode(values, valid).numpy())
    if not windows:
        raise ValueError("No usable windows for inference")
    windows = np.concatenate(windows)
    rows, vectors = [], []
    for si, sequence in enumerate(dataset.sequences):
        indices = [i for i, (owner, _) in enumerate(dataset.items) if owner == si]
        if not indices:
            continue
        group = windows[indices]
        vector = np.concatenate((group.mean(axis=0), group.std(axis=0)))
        if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
            raise ValueError(f"Degenerate embedding: {sequence.key}")
        rows.append({"key": sequence.key, "piece": sequence.piece, "grid": sequence.grid,
                     "windows": len(indices)})
        vectors.append(vector)
    return rows, np.asarray(vectors), windows


def cross_work_neighbors(rows, vectors, *, top_k=5):
    if top_k < 1 or len(rows) != len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Invalid candidates")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Cosine similarity requires nonzero embeddings")
    normalized = vectors / norms[:, None]
    result = []
    for i, query in enumerate(rows):
        scores = normalized @ normalized[i]
        candidates = [j for j, candidate in enumerate(rows) if candidate["piece"] != query["piece"]]
        order = sorted(candidates, key=lambda j: (-float(scores[j]), rows[j]["key"]))[:top_k]
        result.extend({"query": query["key"], "candidate": rows[j]["key"],
                       "query_piece": query["piece"], "candidate_piece": rows[j]["piece"],
                       "rank": rank, "cosine_similarity": float(scores[j])}
                      for rank, j in enumerate(order, 1))
    return result
