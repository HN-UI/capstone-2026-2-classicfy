"""여러 연주의 같은 beat 위치를 집계하는 공통 통계 도구."""

from collections.abc import Iterable, Sequence
from statistics import median


def median_sequence_with_support(
    sequences: Iterable[Sequence[float | None]],
    minimum_support: int = 2,
) -> tuple[list[float | None], list[int]]:
    """위치별 유효 값의 중앙값과 값의 개수를 반환한다."""
    rows = [list(sequence) for sequence in sequences]
    if not rows:
        raise ValueError("At least one sequence is required")
    if minimum_support < 1:
        raise ValueError("minimum_support must be positive")

    sequence_length = len(rows[0])
    if any(len(row) != sequence_length for row in rows[1:]):
        raise ValueError("All sequences must have the same length")

    common_sequence: list[float | None] = []
    support: list[int] = []
    for index in range(sequence_length):
        values = [row[index] for row in rows if row[index] is not None]
        support.append(len(values))
        common_sequence.append(
            float(median(values)) if len(values) >= minimum_support else None
        )
    return common_sequence, support
