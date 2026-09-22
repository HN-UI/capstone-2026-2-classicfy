"""동일 작품의 score와 여러 performance에서 상대 tempo를 추출한다."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isclose, isfinite, log2
from statistics import median
from typing import Literal


BeatType = Literal["b", "db", "bR"]
TempoIntervalStatus = Literal["regular", "special", "suspicious"]
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


@dataclass(frozen=True)
class TempoInterval:
    """같은 악보 위치에 대응하는 score와 performance의 beat 구간."""

    beat_index: int
    score_duration: float
    performance_duration: float
    score_relative_tempo: float | None
    peer_log2_deviation: float | None
    status: TempoIntervalStatus
    status_reason: str | None


@dataclass(frozen=True)
class TempoFeature:
    """작품의 공통 tempo 해석과 그에 대한 개별 연주의 편차."""

    performance_key: str
    intervals: list[TempoInterval]
    common_tempo_sequence: list[float | None]
    common_tempo_support: list[int]
    individual_tempo_sequence: list[float | None]
    overall_individual_tempo: float | None


def _validate_times(times: Sequence[float], field: str) -> list[float]:
    validated = []
    for index, time in enumerate(times):
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not isfinite(time):
            raise ValueError(f"{field}[{index}] must be a finite number: {time!r}")
        validated.append(float(time))

    for index, (current, following) in enumerate(zip(validated, validated[1:])):
        if following <= current:
            raise ValueError(
                f"{field} must be strictly increasing: "
                f"index {index}={current}, index {index + 1}={following}"
            )
    return validated


def _validate_beat_types(
    beat_types: Sequence[BeatType], expected_length: int, field: str
) -> list[BeatType]:
    if len(beat_types) != expected_length or any(
        beat_type not in {"b", "db", "bR"} for beat_type in beat_types
    ):
        raise ValueError(f"Invalid {field}")
    return list(beat_types)


def _calculate_intervals(tempo_input: TempoInput) -> list[TempoInterval]:
    score_beats = _validate_times(tempo_input.score_beats, "score_beats")
    performance_beats = _validate_times(
        tempo_input.performance_beats, "performance_beats"
    )
    if len(score_beats) != len(performance_beats):
        raise ValueError(
            f"Mismatched aligned beats for {tempo_input.performance_key}: "
            f"score={len(score_beats)}, performance={len(performance_beats)}"
        )
    if len(score_beats) < 2:
        raise ValueError(
            f"At least two aligned beats are required for {tempo_input.performance_key}"
        )

    score_beat_types = _validate_beat_types(
        tempo_input.score_beat_types, len(score_beats), "score_beat_types"
    )
    performance_beat_types = _validate_beat_types(
        tempo_input.performance_beat_types,
        len(performance_beats),
        "performance_beat_types",
    )

    intervals = []
    for index in range(len(score_beats) - 1):
        score_duration = score_beats[index + 1] - score_beats[index]
        performance_duration = performance_beats[index + 1] - performance_beats[index]
        contains_br = "bR" in {
            score_beat_types[index],
            score_beat_types[index + 1],
            performance_beat_types[index],
            performance_beat_types[index + 1],
        }
        intervals.append(
            TempoInterval(
                beat_index=index,
                score_duration=score_duration,
                performance_duration=performance_duration,
                score_relative_tempo=(
                    None if contains_br else log2(score_duration / performance_duration)
                ),
                peer_log2_deviation=None,
                status="special" if contains_br else "regular",
                status_reason="contains_bR" if contains_br else None,
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
    보존한 채 ``suspicious``로 구분한다. 두 상태 모두 tempo sequence에서는
    ``None``으로 표시한다.
    """
    if len(performances) < 2:
        raise ValueError("At least two performances are required for piece comparison")

    intervals_by_key: dict[str, list[TempoInterval]] = {}
    first_input = performances[0]
    first_score_beats = list(first_input.score_beats)
    first_score_beat_types = list(first_input.score_beat_types)

    for tempo_input in performances:
        if not tempo_input.performance_key or tempo_input.performance_key in intervals_by_key:
            raise ValueError(
                f"Missing or duplicate performance key: {tempo_input.performance_key!r}"
            )
        if len(tempo_input.score_beats) != len(first_score_beats) or any(
            not isclose(float(current), float(reference), rel_tol=1e-9, abs_tol=1e-9)
            for current, reference in zip(tempo_input.score_beats, first_score_beats)
        ):
            raise ValueError("All performances must use the same score beat positions")
        if list(tempo_input.score_beat_types) != first_score_beat_types:
            raise ValueError("All performances must use the same score beat types")
        intervals_by_key[tempo_input.performance_key] = _calculate_intervals(tempo_input)

    interval_count = len(next(iter(intervals_by_key.values())))
    for index in range(interval_count):
        for performance_key, intervals in intervals_by_key.items():
            interval = intervals[index]
            if interval.status != "regular":
                continue
            peer_durations = [
                peer_intervals[index].performance_duration
                for peer_key, peer_intervals in intervals_by_key.items()
                if peer_key != performance_key
                and peer_intervals[index].status == "regular"
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
                status="suspicious" if suspicious else "regular",
                status_reason=(
                    "extreme_score_and_peer_deviation" if suspicious else None
                ),
            )

    common_tempo_sequence: list[float | None] = []
    common_tempo_support: list[int] = []
    for index in range(interval_count):
        values = [
            intervals[index].score_relative_tempo
            for intervals in intervals_by_key.values()
            if intervals[index].status == "regular"
            and intervals[index].score_relative_tempo is not None
        ]
        common_tempo_support.append(len(values))
        common_tempo_sequence.append(median(values) if len(values) >= 2 else None)

    features = {}
    for performance_key, intervals in intervals_by_key.items():
        individual_sequence = [
            (
                None
                if interval.status != "regular"
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
            intervals=intervals,
            common_tempo_sequence=common_tempo_sequence.copy(),
            common_tempo_support=common_tempo_support.copy(),
            individual_tempo_sequence=individual_sequence,
            overall_individual_tempo=(
                median(valid_individual_values) if valid_individual_values else None
            ),
        )
    return features
