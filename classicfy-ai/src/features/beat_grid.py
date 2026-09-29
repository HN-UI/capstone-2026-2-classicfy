"""연주 시각을 ASAP beat 구간에 배정하는 공통 도구다.

특징 시퀀스의 시간축은 연주 beat 사이의 구간이다. beat가 B개면 구간은 T = B - 1개이고,
i번째 구간은 [beats[i], beats[i + 1])이다. 첫 beat 이전과 마지막 beat 이후의 이벤트는
어느 구간에도 속하지 않는다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeatGrid:
    """beat 경계와 각 구간의 길이 및 유효 여부를 담는다."""

    edges: np.ndarray
    durations: np.ndarray
    mask: np.ndarray


def as_beat_array(beats: Sequence[float], field: str = "beats") -> np.ndarray:
    """beat 시각을 float 배열로 바꾸고 구간을 만들 수 없는 입력이면 ValueError를 낸다."""
    try:
        raw = np.asarray(beats)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}") from exc
    if raw.ndim != 1 or len(raw) < 2:
        raise ValueError(f"At least two {field} are required")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{field} must contain numbers")

    array = raw.astype(float)
    if not np.all(np.isfinite(array)) or np.any(np.diff(array) < 0):
        raise ValueError(f"{field} must be finite and non-decreasing")
    return array


def build_beat_grid(beats: Sequence[float], field: str = "beats") -> BeatGrid:
    """beat 경계에서 구간 길이와 0폭 구간을 제외하는 mask를 만든다."""
    edges = as_beat_array(beats, field)
    durations = np.diff(edges)
    return BeatGrid(edges=edges, durations=durations, mask=durations > 0)


def assign_windows(times: Sequence[float], beats: Sequence[float]) -> np.ndarray:
    """각 시각이 속한 beat 구간 번호를 돌려준다. 어느 구간에도 속하지 않으면 -1이다."""
    edges = as_beat_array(beats)
    index = np.searchsorted(edges, np.asarray(times, dtype=float), side="right") - 1
    index[(index < 0) | (index >= len(edges) - 1)] = -1
    return index
