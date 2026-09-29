"""동일 작품의 score와 여러 performance에서 상대 tempo를 추출한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isclose, log2
from statistics import median
from typing import Literal

import numpy as np

from preprocessing import AsapSample, BeatType

from .beat_grid import as_beat_array, build_beat_grid
from .models import BeatSequence, readonly_array
from .sequence_stats import median_sequence_with_support

TempoIntervalStatus = Literal["regular", "special", "suspicious", "invalid"]
SUSPICIOUS_LOG2_DEVIATION = 3.0
MIN_PEER_SUPPORT_FOR_SUSPICION = 3


@dataclass(frozen=True)
class TempoInput:
    """같은 작품의 연주 하나에서 가져온 정렬된 beat 정보."""

    performance_key: str
    score_beats: Sequence[float]
    performance_beats: Sequence[float]
    score_beat_types: Sequence[BeatType]
    performance_beat_types: Sequence[BeatType]

    @classmethod
    def from_asap_sample(cls, sample: AsapSample) -> TempoInput:
        """정렬된 ASAP 샘플의 beat 정보를 Tempo 입력으로 변환한다."""
        if not sample.aligned:
            raise ValueError(f"Performance is not aligned: {sample.performance_key}")
        return cls(
            performance_key=sample.performance_key,
            score_beats=sample.score_beats.copy(),
            performance_beats=sample.performance_beats.copy(),
            score_beat_types=sample.score_beat_types.copy(),
            performance_beat_types=sample.performance_beat_types.copy(),
        )


@dataclass(frozen=True)
class TempoInterval:
    """같은 악보 위치에 대응하는 score와 performance의 beat 구간."""

    beat_index: int
    score_duration: float
    performance_duration: float
    score_relative_tempo: float | None
    peer_log2_deviation: float | None
    mask: bool
    status: TempoIntervalStatus
    status_reason: str | None


@dataclass(frozen=True)
class TempoFeature:
    """작품의 공통 tempo 해석과 그에 대한 개별 연주의 편차."""

    performance_key: str
    intervals: tuple[TempoInterval, ...]
    common_tempo_sequence: BeatSequence
    common_tempo_support: np.ndarray
    individual_tempo_sequence: BeatSequence
    overall_individual_tempo: float | None

    def __post_init__(self) -> None:
        intervals = tuple(self.intervals)
        support = readonly_array(self.common_tempo_support, dtype=int)
        expected_length = len(intervals)
        if (
            len(self.common_tempo_sequence) != expected_length
            or len(self.individual_tempo_sequence) != expected_length
            or len(support) != expected_length
        ):
            raise ValueError("Tempo sequences must match the interval count")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "common_tempo_support", support)


def _validate_beat_types(
    beat_types: Sequence[BeatType], expected_length: int, field: str
) -> list[BeatType]:
    if len(beat_types) != expected_length or any(
        beat_type not in {"b", "db", "bR"} for beat_type in beat_types
    ):
        raise ValueError(f"Invalid {field}")
    return list(beat_types)


def _calculate_intervals(tempo_input: TempoInput) -> list[TempoInterval]:
    score_grid = build_beat_grid(tempo_input.score_beats, "score_beats")
    performance_grid = build_beat_grid(
        tempo_input.performance_beats, "performance_beats"
    )
    if len(score_grid.edges) != len(performance_grid.edges):
        raise ValueError(
            f"Mismatched aligned beats for {tempo_input.performance_key}: "
            f"score={len(score_grid.edges)}, performance={len(performance_grid.edges)}"
        )

    score_beat_types = _validate_beat_types(
        tempo_input.score_beat_types, len(score_grid.edges), "score_beat_types"
    )
    performance_beat_types = _validate_beat_types(
        tempo_input.performance_beat_types,
        len(performance_grid.edges),
        "performance_beat_types",
    )

    intervals = []
    for index, (score_duration, performance_duration) in enumerate(
        zip(score_grid.durations, performance_grid.durations)
    ):
        interval_mask = bool(score_grid.mask[index] and performance_grid.mask[index])
        contains_br = "bR" in {
            score_beat_types[index],
            score_beat_types[index + 1],
            performance_beat_types[index],
            performance_beat_types[index + 1],
        }
        status: TempoIntervalStatus = "regular"
        status_reason = None
        if not interval_mask:
            status = "invalid"
            status_reason = "zero_duration"
        elif contains_br:
            status = "special"
            status_reason = "contains_bR"

        intervals.append(
            TempoInterval(
                beat_index=index,
                score_duration=float(score_duration),
                performance_duration=float(performance_duration),
                score_relative_tempo=(
                    None
                    if not interval_mask or contains_br
                    else log2(score_duration / performance_duration)
                ),
                peer_log2_deviation=None,
                mask=interval_mask and not contains_br,
                status=status,
                status_reason=status_reason,
            )
        )
    return intervals


def extract_piece_tempo_features(
    performances: Sequence[TempoInput],
) -> dict[str, TempoFeature]:
    """같은 score에 정렬된 여러 연주의 상대 tempo feature를 계산한다.

    Score-relative tempo는 ``log2(score interval / performance interval)``이다.
    양수는 score MIDI보다 빠르고 음수는 느리다는 뜻이다. 같은 beat 위치에서
    연주자들의 중앙값을 공통 해석으로 사용하고, 이를 뺀 값을 개별 해석으로 반환한다.

    ``bR``이 시작 또는 끝에 포함된 interval은 삭제하지 않고 ``special`` 상태로
    보존한다. Score와 최소 3개 peer에서 모두 8배 이상 벗어난 극단 구간도 원시값을
    보존한 채 ``suspicious``로 구분한다. 길이가 0인 구간은 ``invalid``로 보존한다.
    세 상태 모두 mask에서 제외하고 tempo sequence에서는 ``NaN + mask=False``로
    표시한다. Interval에는 원시값과 상태를 그대로 보존한다.
    """
    if len(performances) < 2:
        raise ValueError("At least two performances are required for piece comparison")

    intervals_by_key: dict[str, list[TempoInterval]] = {}
    first_input = performances[0]
    first_score_beats = as_beat_array(first_input.score_beats, "score_beats").tolist()
    first_score_beat_types = list(first_input.score_beat_types)

    for tempo_input in performances:
        if not tempo_input.performance_key or tempo_input.performance_key in intervals_by_key:
            raise ValueError(
                f"Missing or duplicate performance key: {tempo_input.performance_key!r}"
            )
        current_score_beats = as_beat_array(
            tempo_input.score_beats, "score_beats"
        ).tolist()
        if len(current_score_beats) != len(first_score_beats) or any(
            not isclose(float(current), float(reference), rel_tol=1e-9, abs_tol=1e-9)
            for current, reference in zip(current_score_beats, first_score_beats)
        ):
            raise ValueError("All performances must use the same score beat positions")
        if list(tempo_input.score_beat_types) != first_score_beat_types:
            raise ValueError("All performances must use the same score beat types")
        intervals_by_key[tempo_input.performance_key] = _calculate_intervals(tempo_input)

    interval_count = len(next(iter(intervals_by_key.values())))
    for index in range(interval_count):
        for performance_key, intervals in intervals_by_key.items():
            interval = intervals[index]
            if not interval.mask:
                continue
            peer_durations = [
                peer_intervals[index].performance_duration
                for peer_key, peer_intervals in intervals_by_key.items()
                if peer_key != performance_key
                and peer_intervals[index].mask
            ]
            if len(peer_durations) < MIN_PEER_SUPPORT_FOR_SUSPICION:
                continue
            peer_duration = median(peer_durations)
            peer_deviation = abs(log2(peer_duration / interval.performance_duration))
            score_deviation = abs(interval.score_relative_tempo or 0.0)
            suspicious = (
                peer_deviation > SUSPICIOUS_LOG2_DEVIATION
                and score_deviation > SUSPICIOUS_LOG2_DEVIATION
            )
            intervals[index] = replace(
                interval,
                peer_log2_deviation=peer_deviation,
                mask=not suspicious,
                status="suspicious" if suspicious else "regular",
                status_reason=(
                    "extreme_score_and_peer_deviation" if suspicious else None
                ),
            )

    score_relative_sequences = [
        [
            interval.score_relative_tempo if interval.mask else None
            for interval in intervals
        ]
        for intervals in intervals_by_key.values()
    ]
    common_tempo_sequence, common_tempo_support = median_sequence_with_support(
        score_relative_sequences
    )

    features = {}
    for performance_key, intervals in intervals_by_key.items():
        individual_sequence = [
            (
                None
                if not interval.mask
                or interval.score_relative_tempo is None
                or common is None
                else interval.score_relative_tempo - common
            )
            for interval, common in zip(intervals, common_tempo_sequence)
        ]
        valid_individual_values = [
            value for value in individual_sequence if value is not None
        ]
        features[performance_key] = TempoFeature(
            performance_key=performance_key,
            intervals=tuple(intervals),
            common_tempo_sequence=BeatSequence.from_optional(common_tempo_sequence),
            common_tempo_support=common_tempo_support,
            individual_tempo_sequence=BeatSequence.from_optional(individual_sequence),
            overall_individual_tempo=(
                median(valid_individual_values) if valid_individual_values else None
            ),
        )
    return features
