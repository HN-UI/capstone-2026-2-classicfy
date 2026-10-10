"""Separate overall reconstruction from within-window temporal shape."""

import numpy as np
import torch


def baseline_predictions(target, valid, hidden):
    """Baselines use visible values only; return fresh [window, channel, beat] arrays."""
    target, valid, hidden = np.asarray(target), np.asarray(valid), np.asarray(hidden)
    if target.shape != valid.shape or target.shape != hidden.shape or target.ndim != 3:
        raise ValueError("Expected matching window/channel/beat arrays")
    observed = valid & ~hidden
    mean, linear = np.zeros_like(target), np.zeros_like(target)
    for i in range(len(target)):
        for channel in range(target.shape[1]):
            positions = np.flatnonzero(observed[i, channel])
            if len(positions):
                values = target[i, channel, positions]
                mean[i, channel] = values.mean()
                linear[i, channel] = np.interp(np.arange(target.shape[2]), positions, values)
    return {"zero": np.zeros_like(target), "visible_mean": mean, "linear_interpolation": linear}


def shuffle_visible(values, valid, hidden, *, generator):
    """Jointly permute the seven observed feature channels, preserving their values.

    Hidden/missing/padded positions are unchanged. This is an inference-time
    perturbation diagnostic, not a trained order-free model comparison.
    """
    shuffled = values.clone()
    for i in range(len(values)):
        positions = torch.nonzero(valid[i].all(dim=0) & ~hidden[i].any(dim=0)).flatten()
        order = positions[torch.randperm(len(positions), generator=generator)]
        shuffled[i, :, positions] = values[i, :, order]
    return shuffled


def correlation(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def temporal_metrics(target, prediction, selected, *, direction_tolerance=1e-6):
    """Metrics for one channel; selected contains only hidden valid targets.

    Shape correlation removes each window's hidden-target mean. Internal slope
    metrics require both adjacent beats to be hidden, so visible truth cannot
    make a boundary look well reconstructed. Flat true slopes are excluded from
    direction accuracy; their count is reported separately.
    """
    target, prediction, selected = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float), np.asarray(selected)
    if (target.shape != prediction.shape or target.shape != selected.shape or target.ndim != 2
            or selected.dtype != bool or direction_tolerance < 0):
        raise ValueError("Expected matching [window, beat] values and boolean selection")
    if not np.isfinite(target[selected]).all() or not np.isfinite(prediction[selected]).all():
        raise ValueError("Selected values must be finite")
    errors = (prediction[selected] - target[selected]) ** 2
    actual_centered, predicted_centered = [], []
    window_errors = []
    for actual, predicted, keep in zip(target, prediction, selected):
        if keep.any():
            window_errors.append(float(np.mean((predicted[keep] - actual[keep]) ** 2)))
        if keep.sum() >= 2:
            actual_centered.extend(actual[keep] - actual[keep].mean())
            predicted_centered.extend(predicted[keep] - predicted[keep].mean())
    actual_centered, predicted_centered = np.asarray(actual_centered), np.asarray(predicted_centered)
    pair = selected[:, 1:] & selected[:, :-1]
    actual_change = np.diff(target, axis=1)[pair]
    predicted_change = np.diff(prediction, axis=1)[pair]
    nonflat = np.abs(actual_change) > direction_tolerance
    scale = float(np.std(actual_centered)) if len(actual_centered) else 0.
    return {"mse": float(errors.mean()) if len(errors) else None,
            "window_median_mse": float(np.median(window_errors)) if window_errors else None,
            "targets": int(selected.sum()), "windows": len(window_errors),
            "shape_correlation": correlation(actual_centered, predicted_centered),
            "shape_amplitude_ratio": float(np.std(predicted_centered) / scale) if scale > 1e-12 else None,
            "slope_mse": float(np.mean((predicted_change - actual_change) ** 2)) if len(actual_change) else None,
            "slope_correlation": correlation(actual_change, predicted_change),
            "direction_accuracy": float(np.mean(np.sign(predicted_change[nonflat]) == np.sign(actual_change[nonflat]))) if nonflat.any() else None,
            "hidden_adjacent_pairs": int(pair.sum()), "nonflat_pairs": int(nonflat.sum())}
