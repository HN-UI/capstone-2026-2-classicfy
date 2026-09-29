"""연주 내부의 Rubato와 작품 내 상대 Rubato를 분리한다."""

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median

import numpy as np

from .models import BeatSequence, readonly_array
from .sequence_stats import median_sequence_with_support
from .tempo import TempoFeature


@dataclass(frozen=True)
class RubatoFeature:
    """한 연주의 절대적·상대적 beat-level Rubato."""

    performance_key: str
    absolute_rubato_sequence: BeatSequence
    common_rubato_sequence: BeatSequence
    common_rubato_support: np.ndarray
    relative_rubato_sequence: BeatSequence

    def __post_init__(self) -> None:
        support = readonly_array(self.common_rubato_support, dtype=int)
        lengths = {
            len(self.absolute_rubato_sequence),
            len(self.common_rubato_sequence),
            len(self.relative_rubato_sequence),
            len(support),
        }
        if len(lengths) != 1:
            raise ValueError("Rubato sequences must have the same length")
        object.__setattr__(self, "common_rubato_support", support)


def extract_piece_rubato_features(
    tempo_features: Mapping[str, TempoFeature],
) -> dict[str, RubatoFeature]:
    """같은 작품의 Tempo feature에서 절대적·상대적 Rubato를 계산한다.

    절대적 Rubato는 각 연주의 score-relative tempo에서 그 연주의 전체 중앙값을
    뺀 국소 변화다. 공통 Rubato는 같은 score 위치의 절대적 Rubato 중앙값이며,
    상대적 Rubato는 절대적 Rubato에서 공통 Rubato를 뺀 연주자 고유 편차다.

    ``bR``, suspicious 및 0폭 interval처럼 Tempo mask에서 제외된 위치는 모든
    Rubato sequence에서 ``NaN + mask=False``를 유지한다.
    """
    if len(tempo_features) < 2:
        raise ValueError("At least two tempo features are required for piece comparison")

    interval_count: int | None = None
    absolute_by_key: dict[str, list[float | None]] = {}

    for performance_key, feature in tempo_features.items():
        if performance_key != feature.performance_key:
            raise ValueError(
                f"Tempo feature key mismatch: {performance_key!r} != "
                f"{feature.performance_key!r}"
            )
        if interval_count is None:
            interval_count = len(feature.intervals)
        elif len(feature.intervals) != interval_count:
            raise ValueError("All tempo features must have the same interval count")

        score_relative = [
            interval.score_relative_tempo
            if interval.mask
            else None
            for interval in feature.intervals
        ]
        valid_values = [value for value in score_relative if value is not None]
        baseline = median(valid_values) if valid_values else None
        absolute_by_key[performance_key] = [
            None if value is None or baseline is None else value - baseline
            for value in score_relative
        ]

    assert interval_count is not None
    common_sequence, common_support = median_sequence_with_support(
        absolute_by_key.values()
    )

    features = {}
    for performance_key, absolute_sequence in absolute_by_key.items():
        relative_sequence = [
            None
            if absolute is None or common is None
            else absolute - common
            for absolute, common in zip(absolute_sequence, common_sequence)
        ]
        features[performance_key] = RubatoFeature(
            performance_key=performance_key,
            absolute_rubato_sequence=BeatSequence.from_optional(absolute_sequence),
            common_rubato_sequence=BeatSequence.from_optional(common_sequence),
            common_rubato_support=common_support,
            relative_rubato_sequence=BeatSequence.from_optional(relative_sequence),
        )
    return features


def summarize_rubato(feature: RubatoFeature) -> dict[str, float]:
    """절대적·상대적 Rubato의 beat별 편차 크기를 절댓값 중앙값으로 요약한다."""

    def amount(sequence: BeatSequence) -> float:
        values = np.abs(sequence.values[sequence.mask])
        return float(median(values)) if len(values) else float("nan")

    return {
        "absolute_rubato_amount": amount(feature.absolute_rubato_sequence),
        "relative_rubato_amount": amount(feature.relative_rubato_sequence),
    }
