"""Preserve beat gaps, split works before windows, and mask only real targets."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


CHANNELS = ("tempo", "rubato", "dynamics", "articulation",
            "pedal_depth", "pedal_down_ratio", "pedal_changes")


@dataclass(frozen=True)
class PerformanceSequence:
    key: str
    piece: str
    grid: str
    values: np.ndarray
    valid: np.ndarray


def restore_sequences(groups, audits):
    """Adapt the existing normalized cohort without making gap neighbors adjacent.

    All channels use the existing cohort-wide valid-beat intersection. Alignment
    variants remain separate references but share a work key for train/test splits.
    """
    lengths = {(a["piece"], a["grid"]): a["total_intervals"] for a in audits if a["accepted"]}
    sequences = []
    for group in groups:
        length = lengths[group["piece"], group["grid"]]
        indices = np.asarray(group["shared_indices"], dtype=int)
        if (indices.ndim != 1 or len(indices) < 2 or indices[0] < 0
                or indices[-1] >= length or np.any(np.diff(indices) <= 0)):
            raise ValueError("Invalid shared beat positions")
        compact = np.asarray(group["values"])
        if compact.shape != (len(group["keys"]), 7, len(indices)) or not np.isfinite(compact).all():
            raise ValueError("Expected seven finite normalized channels")
        for key, source in zip(group["keys"], compact):
            values = np.zeros((7, length), dtype=np.float32)
            valid = np.zeros((7, length), dtype=bool)
            values[:, indices] = source
            valid[:, indices] = True
            if not np.isfinite(values).all():
                raise ValueError("Feature values overflow float32")
            sequences.append(PerformanceSequence(key, group["piece"].split("::", 1)[0],
                                                 group["grid"], values, valid))
    if len({s.key for s in sequences}) != len(sequences):
        raise ValueError("Duplicate performance keys")
    return sequences


def split_by_piece(sequences, *, seed=20261006, validation_fraction=.15, test_fraction=.15):
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Invalid validation/test fractions")
    pieces = sorted({s.piece for s in sequences})
    if len(pieces) < 3:
        raise ValueError("Need at least three works for train/validation/test")
    order = np.random.default_rng(seed).permutation(len(pieces))
    nval = max(1, int(round(len(pieces) * validation_fraction)))
    ntest = max(1, int(round(len(pieces) * test_fraction)))
    if nval + ntest >= len(pieces):
        raise ValueError("Split leaves no training works")
    return {pieces[i]: "validation" if j < nval else "test" if j < nval + ntest else "train"
            for j, i in enumerate(order)}


class WindowDataset(Dataset):
    """Fixed windows; invalid positions and right padding have no target value."""

    def __init__(self, sequences, *, window_size=64, stride=32, minimum_coverage=.5):
        if window_size < 4 or not 1 <= stride <= window_size or not 0 < minimum_coverage <= 1:
            raise ValueError("Invalid window/stride/coverage")
        self.sequences = list(sequences)
        self.window_size = window_size
        self.items = []
        self.skipped_keys = []
        for si, sequence in enumerate(self.sequences):
            values, valid = sequence.values, sequence.valid
            if values.ndim != 2 or values.shape[0] != 7 or values.shape != valid.shape:
                raise ValueError("Expected matching [7, beat] values and validity")
            if valid.dtype != bool or not np.isfinite(values[valid]).all():
                raise ValueError("Valid targets must be finite with a boolean mask")
            count = values.shape[1]
            starts = list(range(0, max(1, count - window_size + 1), stride))
            if count > window_size and starts[-1] != count - window_size:
                starts.append(count - window_size)
            previous = len(self.items)
            for start in starts:
                # Coverage includes padded positions: short sequences need >=32
                # valid beats with the default 64-beat window and 50% threshold.
                beat_count = valid[:, start:start + window_size].all(axis=0).sum()
                if beat_count >= max(2, int(np.ceil(window_size * minimum_coverage))):
                    self.items.append((si, start))
            if len(self.items) == previous:
                self.skipped_keys.append(sequence.key)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        si, start = self.items[index]
        sequence = self.sequences[si]
        stop = min(start + self.window_size, sequence.values.shape[1])
        width = stop - start
        values = np.zeros((7, self.window_size), np.float32)
        valid = np.zeros((7, self.window_size), bool)
        valid[:, :width] = sequence.valid[:, start:stop]
        values[:, :width] = np.where(valid[:, :width], sequence.values[:, start:stop], 0)
        return torch.from_numpy(values), torch.from_numpy(valid), index


def make_hidden_mask(valid, *, fraction=.2, block_size=4, generator=None):
    """Hide short consecutive beat blocks while leaving real context visible.

    Returns a fresh [batch, 7, beat] mask; neither targets nor validity change.
    The requested fraction counts beats, not channels or padding.
    """
    if (valid.ndim != 3 or valid.shape[1] != 7 or valid.dtype != torch.bool
            or not 0 < fraction < 1 or block_size < 1):
        raise ValueError("Invalid mask settings")
    if valid.device.type != "cpu":
        raise ValueError("Generate reproducible masks on CPU before device transfer")
    hidden = torch.zeros_like(valid)
    for row in range(len(valid)):
        beats = valid[row].any(dim=0)
        count = int(beats.sum())
        if count < 2:
            raise ValueError("Need at least two valid beats to hide one and retain context")
        target = min(count - 1, max(1, int(round(count * fraction))))
        selected = torch.zeros_like(beats)
        while int(selected.sum()) < target:
            eligible = torch.nonzero(beats & ~selected).flatten()
            choice = int(torch.randint(len(eligible), (1,), generator=generator))
            start = int(eligible[choice])
            for beat in range(start, min(start + block_size, len(beats))):
                if beats[beat] and not selected[beat]:
                    selected[beat] = True
                    if int(selected.sum()) == target:
                        break
        hidden[row] = valid[row] & selected[None, :]
    return hidden
