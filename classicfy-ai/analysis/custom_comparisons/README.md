# 선택 작품 비교

선택한 작품의 여러 연주를 곡선과 히트맵으로 비교한다. Feature 주제별 하위 폴더를 사용한다.

현재 [Dynamics·Pedaling 비교](dynamics_pedaling/README.md)는 Chopin op.10/1과 Liszt
La campanella의 기존 비교 그림을 담고 있다.

저장소 루트에서 실행하는 예시:

```bash
.venv/bin/python classicfy-ai/scripts/validate_dynamics_pedaling.py \
  --asap-root ../datasets/ASAP \
  --works Chopin/Etudes_op_10/1 Liszt/Gran_Etudes_de_Paganini/2_La_campanella \
  --out classicfy-ai/analysis/custom_comparisons/dynamics_pedaling
.venv/bin/python classicfy-ai/scripts/validate_articulation.py \
  --asap-root ../datasets/ASAP --nasap-root ../datasets/nASAP \
  --works Bach/Fugue/bwv_883 Chopin/Barcarolle \
  --out classicfy-ai/analysis/custom_comparisons/articulation
```

두 스크립트 모두 `--works`를 주면 선택 작품의 `*_custom.png` 두 개만 생성한다.
다른 feature의 결과는 별도 하위 폴더로 저장한다.

[전체 분석 목록](../README.md)
