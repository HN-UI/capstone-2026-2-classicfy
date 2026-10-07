"""Bidirectional temporal CNN with a single reconstruction bottleneck."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    window_size: int = 64
    hidden_size: int = 64
    embedding_size: int = 64
    blocks: int = 4
    dropout: float = .1

    def __post_init__(self):
        if (self.window_size < 4 or min(self.hidden_size, self.embedding_size, self.blocks) < 1
                or self.blocks > 6 or not 0 <= self.dropout < 1):
            raise ValueError("Invalid CNN dimensions/dropout")


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, values):
        return values + self.layers(values)


class TemporalAutoencoder(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        # Value, validity and synthetic-mask channels distinguish a valid
        # residual zero, a missing/padded beat, and a deliberately hidden beat.
        self.input_projection = nn.Conv1d(21, config.hidden_size, 1)
        self.blocks = nn.Sequential(*(TemporalBlock(config.hidden_size, 2 ** i, config.dropout)
                                      for i in range(config.blocks)))
        # Flatten retains within-window order; no skip path bypasses the vector.
        self.bottleneck = nn.Sequential(nn.Flatten(),
            nn.Linear(config.hidden_size * config.window_size, config.embedding_size))
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_size, config.hidden_size * 2), nn.GELU(),
            nn.Linear(config.hidden_size * 2, 7 * config.window_size),
        )

    def encode(self, values, valid, hidden=None):
        if (values.ndim != 3 or values.shape[1:] != (7, self.config.window_size)
                or valid.shape != values.shape or valid.dtype != torch.bool):
            raise ValueError("Expected [batch, 7, configured window] and boolean validity")
        hidden = torch.zeros_like(valid) if hidden is None else hidden
        if hidden.shape != valid.shape or hidden.dtype != torch.bool or torch.any(hidden & ~valid):
            raise ValueError("Hidden targets must be a subset of valid positions")
        visible = valid & ~hidden
        if not torch.isfinite(values[visible]).all():
            raise ValueError("Visible values must be finite")
        clean = torch.where(visible, values, 0)
        inputs = torch.cat((clean, valid.to(values.dtype), hidden.to(values.dtype)), dim=1)
        features = self.blocks(self.input_projection(inputs))
        return self.bottleneck(features * valid.any(dim=1, keepdim=True).to(features.dtype))

    def forward(self, values, valid, hidden=None):
        embedding = self.encode(values, valid, hidden)
        prediction = self.decoder(embedding).reshape(-1, 7, self.config.window_size)
        return prediction, embedding


def error_totals(prediction, target, hidden):
    if prediction.shape != target.shape or hidden.shape != target.shape:
        raise ValueError("Prediction, target and mask must share their shape")
    if not torch.any(hidden):
        raise ValueError("No hidden targets")
    squared = torch.where(hidden, prediction - target, 0).square()
    return squared.sum(dim=(0, 2)), hidden.sum(dim=(0, 2))


def feature_loss(sums, counts):
    """Equal weight to the five musical features; three pedal channels are one."""
    channel = sums / counts.clamp_min(1)
    groups = ((0,), (1,), (2,), (3,), (4, 5, 6))
    available = [channel[list(group)][counts[list(group)] > 0].mean()
                 for group in groups if torch.any(counts[list(group)] > 0)]
    if not available:
        raise ValueError("No valid error counts")
    return torch.stack(available).mean()
