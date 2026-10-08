"""Alternative encoders with the existing CNN's ordered bottleneck and decoder.

These are bidirectional masked-reconstruction autoencoders, not forecasters.
Missing beats keep their original positions. All outputs pass through one 64D
vector; neither encoder has a skip connection to the reconstruction target.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .model import ModelConfig, TemporalAutoencoder


@dataclass(frozen=True)
class SequenceConfig:
    architecture: str = "bilstm"
    layers: int = 2
    heads: int = 4
    feedforward_size: int = 128

    def __post_init__(self):
        if self.architecture not in ("bilstm", "transformer"):
            raise ValueError("Expected bilstm or transformer")
        if min(self.layers, self.heads, self.feedforward_size) < 1:
            raise ValueError("Positive sequence dimensions required")


class SequenceAutoencoder(nn.Module):
    def __init__(self, config=ModelConfig(), sequence_config=SequenceConfig()):
        super().__init__()
        self.config = config
        self.sequence_config = sequence_config
        width = config.hidden_size
        if width % 2 or (sequence_config.architecture == "transformer" and width % sequence_config.heads):
            raise ValueError("Width must be even and divisible by attention heads")
        # Construct the reference first: for a given torch seed, bottleneck and
        # decoder start from exactly the CNN's weights, despite encoder changes.
        reference = TemporalAutoencoder(config)
        self.bottleneck = reference.bottleneck
        self.decoder = reference.decoder
        self.input_projection = nn.Linear(21, width)
        if sequence_config.architecture == "bilstm":
            self.encoder = nn.LSTM(width, width // 2, sequence_config.layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=config.dropout if sequence_config.layers > 1 else 0)
        else:
            # Independent construction avoids cloned layers sharing identical
            # initial values. No causal mask: both sides of a hidden beat exist.
            self.encoder = nn.ModuleList([
                nn.TransformerEncoderLayer(width, sequence_config.heads,
                    dim_feedforward=sequence_config.feedforward_size,
                    dropout=config.dropout, activation="gelu", batch_first=True,
                    norm_first=True) for _ in range(sequence_config.layers)])
            self.final_norm = nn.LayerNorm(width)
            positions = torch.arange(config.window_size).float()[:, None]
            frequencies = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
            encoding = torch.zeros(config.window_size, width)
            encoding[:, 0::2] = torch.sin(positions * frequencies)
            encoding[:, 1::2] = torch.cos(positions * frequencies)
            self.register_buffer("position_encoding", encoding[None])

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
        inputs = torch.cat((clean, valid.to(values.dtype), hidden.to(values.dtype)), dim=1).transpose(1, 2)
        features = self.input_projection(inputs)
        present = valid.any(dim=1)
        if not present.any(dim=1).all():
            raise ValueError("Each window needs a valid beat")
        if self.sequence_config.architecture == "bilstm":
            # Do not pack/compress gaps: a missing beat still occupies one step.
            features, _ = self.encoder(features)
        else:
            features = features + self.position_encoding.to(features.dtype)
            for layer in self.encoder:
                features = layer(features, src_key_padding_mask=~present)
            features = self.final_norm(features)
        features = features * present[..., None].to(features.dtype)
        # Match CNN's channel-major flatten layout and unchanged linear head.
        return self.bottleneck(features.transpose(1, 2).contiguous())

    def forward(self, values, valid, hidden=None):
        embedding = self.encode(values, valid, hidden)
        return self.decoder(embedding).reshape(-1, 7, self.config.window_size), embedding


def save_sequence_model(path, model, **metadata):
    torch.save({"sequence_format_version": 1, "model_config": asdict(model.config),
                "sequence_config": asdict(model.sequence_config),
                "state_dict": model.state_dict(), **metadata}, path)


def load_sequence_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["sequence_format_version"] != 1:
        raise ValueError("Unsupported sequence checkpoint")
    model = SequenceAutoencoder(ModelConfig(**checkpoint["model_config"]),
                                SequenceConfig(**checkpoint["sequence_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
