"""연주 MIDI의 서스테인 페달(CC64)에서 beat 단위 Pedaling 특징을 만든다.

페달 이벤트는 값이 바뀌는 시점만 담고 있으므로, 다음 이벤트까지 값이 유지되는 계단 신호로
복원한 뒤 beat 구간 안에서 시간 가중으로 요약한다. 첫 이벤트 이전에는 페달이 떼어져 있다고 본다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..midi_loader import MidiData
from .beat_grid import as_beat_array, assign_windows

MAX_PEDAL_VALUE = 127
PEDAL_ON_THRESHOLD = 64  # MIDI 규격에서 CC64가 64 이상이면 페달 on이다.


@dataclass
class PedalingFeature:
    """beat 구간별 페달 사용 정도를 담는다."""

    depth: np.ndarray  # (T,) 구간 안 페달 깊이(값 / 127)의 시간 가중 평균
    down_ratio: np.ndarray  # (T,) 구간 안에서 페달이 on(값 >= 64)이었던 시간 비율
    changes: np.ndarray  # (T,) 구간에서 일어난 페달 on/off 전환 횟수
    mask: np.ndarray  # (T,) 구간 폭이 0보다 크면 True


def _step_integral(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """계단 신호를 첫 이벤트부터 at까지 적분한 값. 첫 이벤트 이전의 신호는 0이다."""
    cumulative = np.concatenate([[0.0], np.cumsum(values[:-1] * np.diff(times))])
    index = np.searchsorted(times, at, side="right") - 1
    started = index >= 0
    safe = np.where(started, index, 0)
    return np.where(started, cumulative[safe] + values[safe] * (at - times[safe]), 0.0)


def extract_pedaling(performance: MidiData, beats: Sequence[float]) -> PedalingFeature:
    """beat 구간별 페달 깊이, on 비율, 전환 횟수를 구한다."""
    edges = as_beat_array(beats)
    window_count = len(edges) - 1
    widths = np.diff(edges)
    mask = widths > 0

    depth = np.zeros(window_count)
    down_ratio = np.zeros(window_count)
    changes = np.zeros(window_count, dtype=int)
    if performance.pedals:
        times = np.array([pedal.time for pedal in performance.pedals])
        raw = np.array([pedal.value for pedal in performance.pedals], dtype=float)
        is_down = (raw >= PEDAL_ON_THRESHOLD).astype(float)

        for signal, output in ((raw / MAX_PEDAL_VALUE, depth), (is_down, down_ratio)):
            area = np.diff(_step_integral(times, signal, edges))
            output[mask] = area[mask] / widths[mask]

        previous = np.concatenate([[0.0], is_down[:-1]])
        windows = assign_windows(times[is_down != previous], edges)
        changes = np.bincount(windows[windows >= 0], minlength=window_count)

    return PedalingFeature(depth=depth, down_ratio=down_ratio, changes=changes, mask=mask)


def summarize_pedaling(feature: PedalingFeature) -> dict[str, float]:
    """곡 전체 요약. 전환 빈도는 같은 작품의 연주끼리 속도 차이에 흔들리지 않도록 초가 아니라 beat당 횟수로 잰다.

    beat 하나의 길이는 작품마다 달라서(느린 곡은 beat가 몇 초), 작품 사이에서 이 값을 그대로 비교하면 안 된다.
    """
    valid = feature.mask
    if not valid.any():
        return {
            "pedal_depth_mean": float("nan"),
            "pedal_usage": float("nan"),
            "pedal_change_rate": float("nan"),
        }
    return {
        "pedal_depth_mean": float(feature.depth[valid].mean()),
        "pedal_usage": float(feature.down_ratio[valid].mean()),
        "pedal_change_rate": float(feature.changes[valid].mean()),
    }
