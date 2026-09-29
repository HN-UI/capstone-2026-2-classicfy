"""연주 MIDI의 velocity에서 beat 단위 Dynamics 특징을 만든다.

악보 MIDI의 velocity는 상수라 대비할 값이 없으므로, Dynamics는 연주 velocity만으로 계산한다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..midi_loader import MidiData
from .beat_grid import as_beat_array, assign_windows

MAX_VELOCITY = 127


@dataclass
class DynamicsFeature:
    """beat 구간별 Dynamics 값과 그 값이 유효한지를 담는다."""

    values: np.ndarray  # (T,) 구간에서 시작한 음의 평균 velocity / 127. 음이 없으면 NaN
    mask: np.ndarray  # (T,) 구간에 시작한 음이 있으면 True
    onset_counts: np.ndarray  # (T,) 구간에서 시작한 음의 개수


def extract_dynamics(performance: MidiData, beats: Sequence[float]) -> DynamicsFeature:
    """구간에서 시작한 음의 평균 velocity를 0~1 범위로 구한다."""
    edges = as_beat_array(beats)
    window_count = len(edges) - 1

    onsets = [note.start for note in performance.notes]
    velocities = np.array([note.velocity for note in performance.notes], dtype=float)
    windows = assign_windows(onsets, edges)
    inside = windows >= 0

    counts = np.bincount(windows[inside], minlength=window_count)
    totals = np.bincount(windows[inside], weights=velocities[inside], minlength=window_count)

    mask = counts > 0
    values = np.full(window_count, np.nan)
    values[mask] = totals[mask] / counts[mask] / MAX_VELOCITY
    return DynamicsFeature(values=values, mask=mask, onset_counts=counts)


def summarize_dynamics(feature: DynamicsFeature) -> dict[str, float]:
    """곡 전체 요약. Range는 극단값에 덜 흔들리도록 5~95 백분위 차이로 잰다."""
    valid = feature.values[feature.mask]
    if len(valid) == 0:
        return {"dynamics_mean": float("nan"), "dynamics_range": float("nan")}
    low, high = np.percentile(valid, [5, 95])
    return {"dynamics_mean": float(valid.mean()), "dynamics_range": float(high - low)}
