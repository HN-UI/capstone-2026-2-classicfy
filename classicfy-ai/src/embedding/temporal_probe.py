"""Frozen-representation linear probes for four-bin, mean-removed trajectories.

Targets describe observed complete windows, not masked or future values. Train
statistics and work-separated folds are explicit to prevent held-out leakage.
"""

import numpy as np


GROUPS = ((0,), (1,), (2,), (3,), (4, 5, 6))


def quarter_targets(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (7, 64) or not np.isfinite(x).all():
        raise ValueError("Expected finite complete [batch,7,64] windows")
    means = x.reshape(-1, 7, 4, 16).mean(axis=3)
    return means - means.mean(axis=2, keepdims=True)


def sample_weights(pieces, performances):
    pieces, performances = np.asarray(pieces), np.asarray(performances)
    if not len(pieces) or pieces.shape != performances.shape:
        raise ValueError("Need aligned nonempty work/performance identifiers")
    weights = np.zeros(len(pieces), dtype=np.float64)
    works = np.unique(pieces)
    for piece in works:
        keep = pieces == piece
        keys = np.unique(performances[keep])
        for key in keys:
            current = keep & (performances == key)
            weights[current] = 1 / (len(works) * len(keys) * current.sum())
    return weights


def work_folds(pieces, *, count=5, seed=20261010):
    pieces = np.asarray(pieces)
    works = np.unique(pieces)
    if len(works) < count or count < 2:
        raise ValueError("Need at least as many training works as folds")
    order = np.random.default_rng(seed).permutation(works)
    mapping = {work: i % count for i, work in enumerate(order)}
    return np.array([mapping[p] for p in pieces], dtype=int), mapping


def normalize_fit(x, weights):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Finite matrix and positive weights summing to one required")
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2))
    scale[scale < 1e-12] = 1
    return mean, scale


def fit_ridge(x, targets, weights, alpha):
    if alpha <= 0:
        raise ValueError("Positive ridge penalty required")
    mean, scale = normalize_fit(x, weights)
    z = (x - mean) / scale
    y = np.asarray(targets, dtype=np.float64).reshape(len(x), -1)
    if not np.isfinite(y).all():
        raise ValueError("Finite targets required")
    intercept = weights @ y
    gram = z.T @ (weights[:, None] * z)
    coefficients = np.linalg.solve(gram + alpha * np.eye(z.shape[1]),
                                   z.T @ (weights[:, None] * (y - intercept)))
    return dict(mean=mean, scale=scale, intercept=intercept, coefficients=coefficients,
                alpha=np.array(alpha))


def predict_ridge(model, x):
    y = ((np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]) @ model["coefficients"] + model["intercept"]
    return y.reshape(-1, 7, 4)


def fit_pca(x, weights, dimensions=64):
    mean, scale = normalize_fit(x, weights)
    z = (x - mean) / scale
    covariance = z.T @ (weights[:, None] * z)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:dimensions]
    return dict(mean=mean, scale=scale, components=eigenvectors[:, order],
                eigenvalues=eigenvalues[order])


def project_pca(model, x):
    return ((np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]) @ model["components"]


def channel_energy(targets, weights):
    return np.sum(weights[:, None] * np.mean(np.asarray(targets) ** 2, axis=2), axis=0)


def feature_average(values):
    values = np.asarray(values, dtype=np.float64)
    available = [np.mean(values[list(group)][np.isfinite(values[list(group)])])
                 for group in GROUPS if np.isfinite(values[list(group)]).any()]
    return float(np.mean(available)) if available else float("nan")


def normalized_error(targets, prediction, weights, train_energy):
    mse = np.sum(weights[:, None] * np.mean((targets - prediction) ** 2, axis=2), axis=0)
    ratios = np.full(7, np.nan)
    np.divide(mse, train_energy, out=ratios, where=train_energy > 1e-12)
    return feature_average(ratios)


def evaluate_channels(targets, prediction, weights):
    target_energy = channel_energy(targets, weights)
    predicted_energy = channel_energy(prediction, weights)
    mse = np.sum(weights[:, None] * np.mean((targets - prediction) ** 2, axis=2), axis=0)
    product = np.sum(weights[:, None] * np.mean(targets * prediction, axis=2), axis=0)
    score, correlation, amplitude = [np.full(7, np.nan) for _ in range(3)]
    np.divide(mse, target_energy, out=score, where=target_energy > 1e-12)
    score = 100 * (1 - score)
    denominator = np.sqrt(target_energy * predicted_energy)
    np.divide(product, denominator, out=correlation, where=denominator > 1e-12)
    np.divide(predicted_energy, target_energy, out=amplitude, where=target_energy > 1e-12)
    amplitude = 100 * np.sqrt(amplitude)
    return dict(mse=mse, flat_mse=target_energy, score_percent=score,
                centered_cosine=correlation, amplitude_percent=amplitude)
