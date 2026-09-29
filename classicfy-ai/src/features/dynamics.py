"""연주 MIDI의 velocity에서 beat 단위 Dynamics 특징을 만든다.

악보 MIDI의 velocity는 상수라 대비할 값이 없으므로, Dynamics는 연주 velocity만으로 계산한다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from preprocessing import MidiData
from .beat_grid import assign_windows, build_beat_grid
from .models import BeatSequence, readonly_array

MAX_VELOCITY = 127


@dataclass(frozen=True)
class DynamicsFeature:
    """beat 구간별 Dynamics 값과 그 값이 유효한지를 담는다."""

    sequence: BeatSequence  # 평균 velocity / 127. 음이 없으면 NaN + mask=False
    onset_counts: np.ndarray  # (T,) 구간에서 시작한 음의 개수

    def __post_init__(self) -> None:
        onset_counts = readonly_array(self.onset_counts, dtype=int)
        if len(onset_counts) != len(self.sequence):
            raise ValueError("Dynamics onset counts must match the beat sequence")
        object.__setattr__(self, "onset_counts", onset_counts)


def extract_dynamics(performance: MidiData, beats: Sequence[float]) -> DynamicsFeature:
    """구간에서 시작한 음의 평균 velocity를 0~1 범위로 구한다."""
    grid = build_beat_grid(beats)
    window_count = len(grid.durations)

    onsets = [note.start for note in performance.notes]
    velocities = np.array([note.velocity for note in performance.notes], dtype=float)
    windows = assign_windows(onsets, grid.edges)
    inside = windows >= 0

    counts = np.bincount(windows[inside], minlength=window_count)
    totals = np.bincount(windows[inside], weights=velocities[inside], minlength=window_count)

    mask = (counts > 0) & grid.mask
    values = np.full(window_count, np.nan)
    values[mask] = totals[mask] / counts[mask] / MAX_VELOCITY
    return DynamicsFeature(
        sequence=BeatSequence(values=values, mask=mask),
        onset_counts=counts,
    )


def summarize_dynamics(feature: DynamicsFeature) -> dict[str, float]:
    """곡 전체 요약. Range는 극단값에 덜 흔들리도록 5~95 백분위 차이로 잰다."""
    valid = feature.sequence.values[feature.sequence.mask]
    if len(valid) == 0:
        return {"dynamics_mean": float("nan"), "dynamics_range": float("nan")}
    low, high = np.percentile(valid, [5, 95])
    return {"dynamics_mean": float(valid.mean()), "dynamics_range": float(high - low)}
