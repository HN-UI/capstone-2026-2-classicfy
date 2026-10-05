# 기존 10작품 Rubato 분석

이전 그림과 CSV를 내용 변경 없이 보존한 폴더다. 여러 작품의 공통 변화와 상대 변화량을
한 번에 비교할 때 사용한다.

| 자료 | 확인할 내용 |
|---|---|
| [공통·상대 Rubato 곡선](rubato_common_relative_10_pieces.png) | 작품별 공통 곡선과 상대 편차의 중앙 50% 범위 |
| [절대·상대 Rubato 양 비교](rubato_absolute_vs_relative_amount_10_pieces.png) | 작품별 공통 제거 전후 변화량 |
| [작품별 요약 CSV](rubato_absolute_relative_summary_10_pieces.csv) | 각 작품의 연주 수와 log2 단위 요약값 |

기존 그림은 여러 패널과 작품 전체의 요약 범위를 사용해 개별 연주 차이를 읽기 어렵다.
이전 두 그림을 재생성하는 전용 스크립트는 없으며, 새 `validate_rubato.py`는 한 작품의
실제 연주를 비교하는 별도 분석이다.

[새 Rubato 그림과 읽기 안내](../README.md) · [전체 분석 목록](../../README.md)
