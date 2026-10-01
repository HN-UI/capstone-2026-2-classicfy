"""(n)ASAP note 정렬로 악보 대비 beat 단위 Articulation 특징을 만든다.

음 하나의 articulation은 "건반을 누르고 있던 시간 ÷ 악보 음가가 그 자리의 실제 빠르기에서
차지하는 시간"의 log2다. 0이면 악보 음가만큼, 양수면 더 길게(레가토 쪽), 음수면 더 짧게
(스타카토 쪽) 쳤다는 뜻이다.

악보 음가를 초로 바꿀 때는 연주 자체의 onset으로 만든 악보 위치 → 연주 시각 사상(tempo map)을
쓴다. 그래서 rubato로 늘어나거나 줄어든 시간은 기대 길이에 이미 반영되고, articulation에는
"음과 음 사이를 얼마나 끊거나 이었는가"만 남는다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from preprocessing import NoteAlignment

from .beat_grid import assign_windows, build_beat_grid
from .models import BeatSequence, readonly_array

# 장식음이 붙은 음은 연주에서 여러 음으로 쪼개져 정렬된 음 하나의 길이가 음가를 대표하지 못한다.
ORNAMENT_ATTRIBUTES = frozenset({"trillmark", "tremolo", "mordent", "invertedmordent", "turn"})
# 같은 악보 위치에 정렬된 음들의 연주 onset 중앙값을 그 위치의 연주 시각으로 쓴다.
_SCORE_POSITION_DECIMALS = 6


@dataclass(frozen=True)
class NoteArticulation:
    """계산에 쓰인 음들의 연주 onset(초), log2 articulation 비율, 원래 정렬에서의 위치."""

    onsets: np.ndarray
    values: np.ndarray
    match_indices: np.ndarray  # alignment.matches에서의 인덱스. 악보 속성 등을 다시 찾을 때 쓴다
    excluded_counts: dict[str, int]

    def __post_init__(self) -> None:
        onsets = readonly_array(self.onsets, dtype=float)
        values = readonly_array(self.values, dtype=float)
        match_indices = readonly_array(self.match_indices, dtype=int)
        if not onsets.shape == values.shape == match_indices.shape:
            raise ValueError("Note articulation arrays must have the same shape")
        object.__setattr__(self, "onsets", onsets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "match_indices", match_indices)
        object.__setattr__(self, "excluded_counts", dict(self.excluded_counts))


@dataclass(frozen=True)
class ArticulationFeature:
    """beat 구간별 articulation 중앙값과 그 값에 쓰인 음의 개수."""

    sequence: BeatSequence  # 구간에서 시작한 음들의 log2 비율 중앙값. 음이 없으면 NaN
    note_counts: np.ndarray  # (T,) 구간에서 시작해 계산에 쓰인 음의 개수

    def __post_init__(self) -> None:
        note_counts = readonly_array(self.note_counts, dtype=int)
        if len(note_counts) != len(self.sequence):
            raise ValueError("Articulation note counts must match the beat sequence")
        object.__setattr__(self, "note_counts", note_counts)


def _longest_increasing(values: np.ndarray) -> np.ndarray:
    """값이 순증가하는 가장 긴 부분열의 인덱스를 돌려준다(O(n log n))."""
    tails: list[float] = []
    tail_indices: list[int] = []
    previous = np.full(len(values), -1)
    for index, value in enumerate(values):
        position = int(np.searchsorted(tails, value, side="left"))
        if position > 0:
            previous[index] = tail_indices[position - 1]
        if position == len(tails):
            tails.append(value)
            tail_indices.append(index)
        else:
            tails[position] = value
            tail_indices[position] = index

    kept = []
    index = tail_indices[-1] if tail_indices else -1
    while index >= 0:
        kept.append(index)
        index = previous[index]
    return np.array(kept[::-1], dtype=int)


def build_tempo_map(alignment: NoteAlignment) -> tuple[np.ndarray, np.ndarray]:
    """악보 위치(beat)와 그 위치의 연주 시각(초)을 짝지은 단조 증가 점들을 만든다.

    꾸밈음은 본음보다 앞당겨 치므로 제외한다. 같은 악보 위치의 음들은 연주 onset
    중앙값으로 묶고, 잘못 정렬된 음이 만든 역행을 없애기 위해 연주 시각이 순증가하는
    가장 긴 부분열만 남긴다.
    """
    onsets_by_position: dict[float, list[float]] = {}
    for score_note, performed_note in alignment.matches:
        if score_note.is_grace:
            continue
        position = round(score_note.onset_beats, _SCORE_POSITION_DECIMALS)
        onsets_by_position.setdefault(position, []).append(performed_note.onset)

    positions = np.array(sorted(onsets_by_position), dtype=float)
    times = np.array(
        [np.median(onsets_by_position[position]) for position in positions], dtype=float
    )
    kept = _longest_increasing(times)
    return positions[kept], times[kept]


def extract_note_articulation(alignment: NoteAlignment) -> NoteArticulation:
    """정렬된 음마다 log2(실제 누른 시간 / tempo map으로 환산한 악보 음가)를 구한다."""
    positions, times = build_tempo_map(alignment)
    excluded = {"grace": 0, "ornament": 0, "outside_tempo_map": 0, "non_positive": 0}
    onsets = []
    values = []
    match_indices = []
    if len(positions) < 2:
        excluded["outside_tempo_map"] = sum(
            not score_note.is_grace for score_note, _ in alignment.matches
        )
        excluded["grace"] = len(alignment.matches) - excluded["outside_tempo_map"]
        return NoteArticulation(
            onsets=onsets, values=values, match_indices=match_indices, excluded_counts=excluded
        )

    first, last = positions[0], positions[-1]
    for match_index, (score_note, performed_note) in enumerate(alignment.matches):
        if score_note.is_grace:
            excluded["grace"] += 1
            continue
        if ORNAMENT_ATTRIBUTES.intersection(score_note.attributes):
            excluded["ornament"] += 1
            continue
        # tempo map 밖(곡의 마지막 음 등)은 외삽해야 하므로 쓰지 않는다.
        if score_note.onset_beats < first or score_note.offset_beats > last:
            excluded["outside_tempo_map"] += 1
            continue

        expected = float(
            np.interp(score_note.offset_beats, positions, times)
            - np.interp(score_note.onset_beats, positions, times)
        )
        if expected <= 0 or performed_note.duration <= 0:
            excluded["non_positive"] += 1
            continue
        onsets.append(performed_note.onset)
        values.append(np.log2(performed_note.duration / expected))
        match_indices.append(match_index)

    order = np.argsort(onsets, kind="stable")
    return NoteArticulation(
        onsets=np.asarray(onsets, dtype=float)[order],
        values=np.asarray(values, dtype=float)[order],
        match_indices=np.asarray(match_indices, dtype=int)[order],
        excluded_counts=excluded,
    )


def extract_articulation(
    alignment: NoteAlignment, beats: Sequence[float]
) -> ArticulationFeature:
    """연주 beat 구간마다 그 안에서 시작한 음들의 articulation 중앙값을 구한다."""
    grid = build_beat_grid(beats)
    window_count = len(grid.durations)
    notes = extract_note_articulation(alignment)

    windows = assign_windows(notes.onsets, grid.edges)
    inside = windows >= 0
    counts = np.bincount(windows[inside], minlength=window_count)

    values = np.full(window_count, np.nan)
    order = np.argsort(windows[inside], kind="stable")
    grouped_values = notes.values[inside][order]
    boundaries = np.concatenate([[0], np.cumsum(counts)])
    for window in np.flatnonzero(counts):
        values[window] = np.median(grouped_values[boundaries[window] : boundaries[window + 1]])

    mask = (counts > 0) & grid.mask
    values[~mask] = np.nan
    return ArticulationFeature(
        sequence=BeatSequence(values=values, mask=mask),
        note_counts=counts,
    )


def summarize_articulation(feature: ArticulationFeature) -> dict[str, float]:
    """곡 전체 요약. Range는 극단값에 덜 흔들리도록 5~95 백분위 차이로 잰다."""
    valid = feature.sequence.values[feature.sequence.mask]
    if len(valid) == 0:
        return {"articulation_mean": float("nan"), "articulation_range": float("nan")}
    low, high = np.percentile(valid, [5, 95])
    return {
        "articulation_mean": float(valid.mean()),
        "articulation_range": float(high - low),
    }
