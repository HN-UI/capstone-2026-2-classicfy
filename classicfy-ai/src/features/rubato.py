"""연주 내부의 Rubato와 작품 내 상대 Rubato를 분리한다."""

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median

from .tempo import TempoFeature


@dataclass(frozen=True)
class RubatoFeature:
    """한 연주의 절대적·상대적 beat-level Rubato."""

    performance_key: str
    overall_score_relative_tempo: float | None
    absolute_rubato_sequence: list[float | None]
    common_rubato_sequence: list[float | None]
    common_rubato_support: list[int]
    relative_rubato_sequence: list[float | None]
    absolute_rubato_amount: float | None
    relative_rubato_amount: float | None


def _amount(sequence: list[float | None]) -> float | None:
    values = [abs(value) for value in sequence if value is not None]
    return median(values) if values else None


def extract_piece_rubato_features(
    tempo_features: Mapping[str, TempoFeature],
) -> dict[str, RubatoFeature]:
    """같은 작품의 Tempo feature에서 절대적·상대적 Rubato를 계산한다.

    절대적 Rubato는 각 연주의 score-relative tempo에서 그 연주의 전체 중앙값을
    뺀 국소 변화다. 공통 Rubato는 같은 score 위치의 절대적 Rubato 중앙값이며,
    상대적 Rubato는 절대적 Rubato에서 공통 Rubato를 뺀 연주자 고유 편차다.

    ``bR`` 및 suspicious interval처럼 Tempo에서 일반 통계에서 제외된 위치는
    모든 Rubato sequence에서 ``None``을 유지한다.
    """
    if len(tempo_features) < 2:
        raise ValueError("At least two tempo features are required for piece comparison")

    interval_count: int | None = None
    absolute_by_key: dict[str, list[float | None]] = {}
    baseline_by_key: dict[str, float | None] = {}

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
            if interval.status == "regular"
            else None
            for interval in feature.intervals
        ]
        valid_values = [value for value in score_relative if value is not None]
        baseline = median(valid_values) if valid_values else None
        baseline_by_key[performance_key] = baseline
        absolute_by_key[performance_key] = [
            None if value is None or baseline is None else value - baseline
            for value in score_relative
        ]

    assert interval_count is not None
    common_sequence: list[float | None] = []
    common_support: list[int] = []
    for index in range(interval_count):
        values = [
            sequence[index]
            for sequence in absolute_by_key.values()
            if sequence[index] is not None
        ]
        common_support.append(len(values))
        common_sequence.append(median(values) if len(values) >= 2 else None)

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
            overall_score_relative_tempo=baseline_by_key[performance_key],
            absolute_rubato_sequence=absolute_sequence,
            common_rubato_sequence=common_sequence.copy(),
            common_rubato_support=common_support.copy(),
            relative_rubato_sequence=relative_sequence,
            absolute_rubato_amount=_amount(absolute_sequence),
            relative_rubato_amount=_amount(relative_sequence),
        )
    return features
