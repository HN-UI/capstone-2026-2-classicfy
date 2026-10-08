"""Controlled order-sensitivity checks on complete feature windows.

This measures a prescribed perturbation ordering, not musical similarity or
preference. All seven channels are permuted together. No missing beats move.
"""

import numpy as np


ORDERS = ("shuffle", "block8", "reverse", "adjacent")
CHANNEL_WEIGHTS = np.array([1 / 5] * 4 + [1 / 15] * 3)


def make_variants(values, *, seed=20261009, noise_levels=(.02, .05, .10)):
    """Noise RMS equals the stated fraction of each window/channel's std.

    Zero-variance channels receive no noise. Same standardized noise direction
    is reused across levels. Permutations preserve each channel's full multiset
    and contemporaneous cross-channel tuples, but not their temporal ordering.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 7 or x.shape[2] % 8 or not np.isfinite(x).all():
        raise ValueError("Expected finite complete [batch, 7, beats divisible by 8]")
    if not len(x) or any(level <= 0 for level in noise_levels):
        raise ValueError("Need windows and positive noise levels")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape)
    noise -= noise.mean(axis=2, keepdims=True)
    noise /= np.sqrt(np.mean(noise ** 2, axis=2, keepdims=True))
    noise *= x.std(axis=2, keepdims=True)
    variants = {f"noise{level:.2f}": (x + level * noise).astype(np.float32)
                for level in noise_levels}
    permutations = {}
    length = x.shape[2]
    for name in ORDERS:
        indices = []
        for _ in x:
            if name == "shuffle":
                p = rng.permutation(length)
            elif name == "block8":
                order = rng.permutation(length // 8)
                if np.array_equal(order, np.arange(len(order))):
                    order = np.roll(order, 1)
                p = (order[:, None] * 8 + np.arange(8)).ravel()
            elif name == "reverse":
                p = np.arange(length)[::-1]
            else:
                p = np.arange(length).reshape(-1, 2)[:, ::-1].ravel()
            indices.append(p)
        permutations[name] = np.stack(indices)
        variants[name] = np.take_along_axis(x, permutations[name][:, None, :], axis=2).astype(np.float32)
    return variants, permutations


def summary_vectors(values):
    """Existing 14D summary, with a deterministic reduction for permuted inputs."""
    x = np.sort(np.asarray(values, dtype=np.float64), axis=2)
    representative = x.mean(axis=2)
    representative[:, 1] = np.median(np.abs(x[:, 1]), axis=1)
    width = np.percentile(x, 95, axis=2) - np.percentile(x, 5, axis=2)
    return np.stack((representative, width), axis=2).reshape(len(x), 14)


def summary_distance(left, right):
    squared = (np.asarray(left) - right).reshape(-1, 7, 2) ** 2
    return np.sqrt((squared.mean(axis=2) * CHANNEL_WEIGHTS).sum(axis=1))


def raw_distance(left, right):
    squared = (np.asarray(left, dtype=np.float64) - right) ** 2
    return np.sqrt((squared.mean(axis=2) * CHANNEL_WEIGHTS).sum(axis=1))


def cosine_distance(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(left), np.nan)
    defined = denominator > 1e-12
    result[defined] = 1 - np.clip(np.sum(left[defined] * right[defined], axis=1) / denominator[defined], -1, 1)
    return result


def triplet_scores(near, reordered, *, tolerance=1e-8):
    """One point for d(A,B)<d(A,C), half for a tie; undefined stays NaN."""
    near, reordered = np.asarray(near), np.asarray(reordered)
    valid = np.isfinite(near) & np.isfinite(reordered)
    tied = valid & (np.abs(near - reordered) <= tolerance)
    score = np.full(near.shape, np.nan)
    score[valid] = (near[valid] < reordered[valid] - tolerance).astype(float)
    score[tied] = .5
    return score, tied


def select_complete_nonoverlapping(dataset, valid):
    """Greedy chronological windows within each performance; never cross gaps."""
    complete = np.asarray(valid).all(axis=(1, 2))
    selected, next_start = [], {}
    for index, (sequence_index, start) in enumerate(dataset.items):
        if complete[index] and start >= next_start.get(sequence_index, -1):
            selected.append(index)
            next_start[sequence_index] = start + dataset.window_size
    return np.asarray(selected, dtype=int)
