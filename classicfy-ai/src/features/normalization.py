"""Scale existing piece residuals without recomputing their common pattern."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .common_pattern import SeparatedFeature
from .models import BeatSequence


# Selected from ASAP residual diagnostics; see analysis/feature_normalization.
DEFAULT_SCALE_METHODS = MappingProxyType({
    "dynamics": "mad",
    "pedal_depth": "std",
    "pedal_down_ratio": "std",
    "pedal_changes": "std",
    "articulation": "mad",
})
MAD_NORMAL_FACTOR = 1.4826
IQR_NORMAL_FACTOR = 1.3489795003921634
SCALE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ResidualScale:
    """Fitted denominator and diagnostics, suitable for serialization/reuse.

    method is the method actually used (mad/iqr/std/unit/empty). MAD falls back
    to IQR then SD; IQR falls back to SD. Degenerate spread uses denominator 1,
    preserving residuals. Empty input has NaN scale and cannot be transformed.
    """

    value: float
    requested_method: str
    method: str
    valid_count: int
    mad: float
    iqr: float
    std: float

    def __post_init__(self):
        if self.requested_method not in {"mad", "iqr", "std"}:
            raise ValueError("Unknown requested scale method")
        if self.method not in {"mad", "iqr", "std", "unit", "empty"}:
            raise ValueError("Unknown fitted scale method")
        if self.valid_count < 0:
            raise ValueError("Scale support cannot be negative")
        if self.method == "empty":
            if self.valid_count != 0 or not np.isnan(self.value):
                raise ValueError("Empty scale requires zero support and NaN value")
        elif self.valid_count == 0 or not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("Nonempty scale requires positive finite value and support")


@dataclass(frozen=True)
class NormalizedFeature(SeparatedFeature):
    """Preserves raw/common/relative/support alongside standardized values."""

    standardized: BeatSequence
    scale: ResidualScale
    feature_name: str

    def __post_init__(self):
        super().__post_init__()
        if len(self.standardized) != len(self.relative):
            raise ValueError("Standardized sequence must match relative length")
        sequence = BeatSequence(self.standardized.values, self.standardized.mask)
        object.__setattr__(self, "standardized", sequence)


def fit_residual_scale(sequences: Sequence[BeatSequence], *, method: str = "mad") -> ResidualScale:
    """Fit one denominator to all finite, masked-in residual beats.

    The median is used only to estimate MAD, not to recenter the residuals.
    SD uses ddof=0. Values <= 1e-12 count as degenerate numerical spread, never
    as a denominator floor. No clipping, winsorization or log transform is used.
    Callers fitting a global training scale must exclude validation/test data.
    """
    if method not in {"mad", "iqr", "std"}:
        raise ValueError("method must be mad, iqr, or std")
    chunks = [s.values[s.mask & np.isfinite(s.values)] for s in sequences]
    values = np.concatenate(chunks) if chunks else np.zeros(0)
    if not len(values):
        return ResidualScale(np.nan, method, "empty", 0, np.nan, np.nan, np.nan)
    with np.errstate(over="raise", invalid="raise"):
        try:
            candidates = {
                "mad": float(MAD_NORMAL_FACTOR * np.median(np.abs(values - np.median(values)))),
                "iqr": float(np.diff(np.percentile(values, [25, 75]))[0] / IQR_NORMAL_FACTOR),
                "std": float(np.std(values)),
            }
        except FloatingPointError as exc:
            raise ValueError("Residual magnitudes overflow scale estimation") from exc
    chain = {"mad": ("mad", "iqr", "std"), "iqr": ("iqr", "std"), "std": ("std",)}[method]
    actual = next((name for name in chain if candidates[name] > SCALE_TOLERANCE), "unit")
    return ResidualScale(candidates.get(actual, 1.0), method, actual, len(values), **candidates)


def standardize_sequence(sequence: BeatSequence, scale: ResidualScale) -> BeatSequence:
    """Transform an existing residual with a fitted denominator; preserve validity.

    This can also scale existing Tempo/Rubato residuals if explicitly requested;
    it does not remove any common pattern or change their separate state models.
    """
    mask = sequence.mask & np.isfinite(sequence.values)
    values = np.full(len(sequence), np.nan)
    if scale.method == "empty":
        mask = np.zeros(len(sequence), dtype=bool)
    else:
        with np.errstate(over="raise", invalid="raise"):
            try:
                values[mask] = sequence.values[mask] / scale.value
            except FloatingPointError as exc:
                raise ValueError("Residual magnitudes overflow standardization") from exc
    return BeatSequence(values, mask)


def standardize_piece_feature(
    features: Sequence[SeparatedFeature], *, feature_name: str, method: str | None = None,
) -> dict[str, NormalizedFeature]:
    """Fit a piece scale and apply it to already separated D/P/A features.

    Uses the feature-specific default policy unless method is supplied. A single
    denominator is pooled across performances and beats of this piece/channel.
    The common pattern and original masks are copied, never recomputed.
    """
    if feature_name not in DEFAULT_SCALE_METHODS:
        raise ValueError(f"Unknown feature_name: {feature_name}")
    if len(features) < 2:
        raise ValueError("At least two separated performances are required")
    reference, keys = features[0], set()
    for feature in features:
        if feature.performance_key in keys:
            raise ValueError("Duplicate performance key")
        keys.add(feature.performance_key)
        if feature.piece_key != reference.piece_key:
            raise ValueError("All features must belong to the same piece")
        if len(feature.relative) != len(reference.relative) or not np.allclose(
            feature.score_beats, reference.score_beats, rtol=1e-9, atol=1e-9
        ) or feature.score_beat_types != reference.score_beat_types:
            raise ValueError("All features must use the same score grid and beat types")
        if feature.minimum_support != reference.minimum_support or not np.array_equal(
            feature.common_support, reference.common_support
        ) or not np.array_equal(feature.common.mask, reference.common.mask) or not np.array_equal(
            feature.common.values, reference.common.values, equal_nan=True
        ):
            raise ValueError("All features must share the same fitted common pattern")
    scale = fit_residual_scale([f.relative for f in features],
                               method=DEFAULT_SCALE_METHODS[feature_name] if method is None else method)
    return {
        f.performance_key: NormalizedFeature(
            performance_key=f.performance_key, piece_key=f.piece_key,
            score_beats=f.score_beats, score_beat_types=f.score_beat_types,
            raw=f.raw, common=f.common, relative=f.relative,
            common_support=f.common_support, minimum_support=f.minimum_support,
            standardized=standardize_sequence(f.relative, scale), scale=scale, feature_name=feature_name,
        ) for f in features
    }
