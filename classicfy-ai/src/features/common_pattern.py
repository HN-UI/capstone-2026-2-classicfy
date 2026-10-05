"""같은 작품의 beat-level feature를 데이터 공통 패턴과 연주별 편차로 분리한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from preprocessing import AsapSample, BeatType

from .beat_grid import as_beat_array
from .models import BeatSequence, readonly_array
from .sequence_stats import median_sequence_with_support


@dataclass(frozen=True)
class PieceFeatureInput:
    """한 연주의 기존 feature와 작품·악보 위치 정보.

    piece_key는 반복 구조까지 같은 작품을 식별해야 한다. sequence에는 Dynamics,
    Pedaling의 각 지표 또는 Articulation의 기존 BeatSequence를 전달한다.
    """

    performance_key: str
    piece_key: str
    score_beats: Sequence[float]
    sequence: BeatSequence
    score_beat_types: Sequence[BeatType] | None = None

    def __post_init__(self) -> None:
        for name in ("performance_key", "piece_key"):
            key = getattr(self, name)
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{name} must be a non-empty string")
        beats = as_beat_array(self.score_beats, "score_beats")
        if len(self.sequence) != len(beats) - 1:
            raise ValueError("Feature sequence must have length len(score_beats) - 1")
        if self.sequence.values.dtype.kind not in "iuf":
            raise ValueError("Feature values must contain real numbers")
        beat_types = None if self.score_beat_types is None else tuple(self.score_beat_types)
        if beat_types is not None and (
            len(beat_types) != len(beats)
            or any(beat_type not in {"b", "db", "bR"} for beat_type in beat_types)
        ):
            raise ValueError("Invalid score_beat_types")
        object.__setattr__(self, "score_beats", readonly_array(beats))
        object.__setattr__(self, "score_beat_types", beat_types)
        object.__setattr__(self, "sequence", BeatSequence(self.sequence.values, self.sequence.mask))

    @classmethod
    def from_asap_sample(
        cls, sample: AsapSample, sequence: BeatSequence, *, piece_key: str | None = None
    ) -> PieceFeatureInput:
        """정렬된 ASAP 샘플과 이미 추출한 sequence를 입력으로 변환한다.

        기본 작품 키는 score_path다. (n)ASAP에서 정렬 파일이 다른 작품 폴더로
        대응된 경우 그 폴더 이름도 포함한다. 추가적인 반복 구조 식별이 필요하면
        호출자가 piece_key를 명시한다. note 정렬 품질 필터링은 호출자가 결정한다.
        """
        if not sample.aligned:
            raise ValueError(f"Performance is not aligned: {sample.performance_key}")
        if len(sample.score_beats) != len(sample.performance_beats):
            raise ValueError(f"Mismatched aligned beats for {sample.performance_key}")
        if piece_key is None:
            piece_key = str(sample.score_path)
            if (
                sample.note_alignment_path is not None
                and sample.note_alignment_path.parent.name != sample.performance_path.parent.name
            ):
                piece_key += f"::{sample.note_alignment_path.parent.name}"
        return cls(
            performance_key=sample.performance_key,
            piece_key=piece_key,
            score_beats=sample.score_beats,
            sequence=sequence,
            score_beat_types=sample.score_beat_types,
        )


@dataclass(frozen=True)
class SeparatedFeature:
    """원본, 작품 중앙 패턴, 상대값과 위치별 지원 연주 수.

    raw.mask는 원본 mask를 보존한다. common.mask는 최소 지원 충족 여부,
    relative.mask는 그 연주에서 상대값을 사용할 수 있는지를 나타낸다.
    """

    performance_key: str
    piece_key: str
    score_beats: np.ndarray
    score_beat_types: tuple[BeatType, ...] | None
    raw: BeatSequence
    common: BeatSequence
    relative: BeatSequence
    common_support: np.ndarray
    minimum_support: int

    def __post_init__(self) -> None:
        beats = readonly_array(self.score_beats, dtype=float)
        support = readonly_array(self.common_support, dtype=int)
        expected_length = len(beats) - 1
        if any(len(sequence) != expected_length for sequence in (self.raw, self.common, self.relative)):
            raise ValueError("Separated sequences must match the score interval count")
        if len(support) != expected_length:
            raise ValueError("Common support must match the score interval count")
        object.__setattr__(self, "score_beats", beats)
        object.__setattr__(self, "common_support", support)
        if self.score_beat_types is not None:
            object.__setattr__(self, "score_beat_types", tuple(self.score_beat_types))
        for name in ("raw", "common", "relative"):
            sequence = getattr(self, name)
            object.__setattr__(self, name, BeatSequence(sequence.values, sequence.mask))


def separate_piece_feature(
    performances: Sequence[PieceFeatureInput], *, minimum_support: int = 2
) -> dict[str, SeparatedFeature]:
    """한 feature의 작품별 위치 중앙값을 제거해 relative = raw - common을 반환한다.

    같은 작품·score grid·길이·beat 종류를 검증한다. 원본 mask와 유한값 여부,
    score 구간의 양수 폭을 만족하는 값만 집계한다. 대상 연주 자신도 중앙값에
    포함한다. support 미달 위치는 NaN + mask=False다. 입력을 수정하지 않으며
    scale 표준화·clipping이나 Tempo/Rubato 재상대화는 수행하지 않는다.
    """
    if (
        isinstance(minimum_support, bool)
        or not isinstance(minimum_support, int)
        or minimum_support < 2
    ):
        raise ValueError("minimum_support must be an integer of at least 2")
    if len(performances) < 2:
        raise ValueError("At least two performances are required for piece comparison")
    reference = performances[0]
    keys: set[str] = set()
    for item in performances:
        if item.performance_key in keys:
            raise ValueError(f"Duplicate performance key: {item.performance_key!r}")
        keys.add(item.performance_key)
        if item.piece_key != reference.piece_key:
            raise ValueError("All performances must belong to the same piece")
        if len(item.sequence) != len(reference.sequence):
            raise ValueError("All performances must have the same interval count")
        if not np.allclose(item.score_beats, reference.score_beats, rtol=1e-9, atol=1e-9):
            raise ValueError("All performances must use the same score beat positions")
        if item.score_beat_types != reference.score_beat_types:
            raise ValueError("All performances must use the same score beat types")

    valid = [
        item.sequence.mask & np.isfinite(item.sequence.values) & (np.diff(item.score_beats) > 0)
        for item in performances
    ]
    common_values, support = median_sequence_with_support(
        [
            [float(value) if keep else None for value, keep in zip(item.sequence.values, mask)]
            for item, mask in zip(performances, valid)
        ],
        minimum_support=minimum_support,
    )
    common = BeatSequence.from_optional(common_values)
    results = {}
    for item, raw_valid in zip(performances, valid):
        mask = raw_valid & common.mask
        values = np.full(len(item.sequence), np.nan)
        values[mask] = item.sequence.values[mask] - common.values[mask]
        results[item.performance_key] = SeparatedFeature(
            performance_key=item.performance_key,
            piece_key=item.piece_key,
            score_beats=item.score_beats,
            score_beat_types=item.score_beat_types,
            raw=item.sequence,
            common=common,
            relative=BeatSequence(values, mask),
            common_support=support,
            minimum_support=minimum_support,
        )
    return results
