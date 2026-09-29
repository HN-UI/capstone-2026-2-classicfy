"""Feature extractor들이 공유하는 결과 모델."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def readonly_array(
    values: Sequence[float | int | bool] | np.ndarray,
    *,
    dtype: type | None = None,
) -> np.ndarray:
    """입력과 메모리를 공유하지 않는 읽기 전용 1차원 배열을 만든다."""
    array = np.asarray(values, dtype=dtype).copy()
    if array.ndim != 1:
        raise ValueError("Feature arrays must be one-dimensional")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BeatSequence:
    """beat 구간별 값과 각 값의 유효 여부를 함께 보관한다."""

    values: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        values = readonly_array(self.values)
        mask = readonly_array(self.mask, dtype=bool)
        if values.shape != mask.shape:
            raise ValueError("BeatSequence values and mask must have the same shape")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def from_optional(cls, values: Sequence[float | None]) -> BeatSequence:
        """``None``을 ``NaN + mask=False``로 바꾼 sequence를 만든다."""
        mask = np.array([value is not None for value in values], dtype=bool)
        numeric = np.array(
            [float(value) if value is not None else np.nan for value in values],
            dtype=float,
        )
        return cls(values=numeric, mask=mask)
