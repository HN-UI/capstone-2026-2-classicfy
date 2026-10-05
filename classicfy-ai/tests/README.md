# Classicfy AI tests

테스트는 검증 대상의 책임에 따라 나눈다.

```text
tests/
├── unit/
│   ├── features/       # beat grid, 결과 모델, 다섯 feature와 summary 계산
│   └── preprocessing/  # ASAP annotation, MIDI 원본, (n)ASAP match 파일 로딩
└── integration/        # loader 출력이 feature 입력까지 이어지는 전체 흐름
```

`unit/features/test_characterization.py`는 하나의 대표 입력에 대한 Tempo, Rubato,
Dynamics, Pedaling의 beat-level 값과 summary를 함께 고정한다. 각 feature의 경계 조건과
계산 세부사항은 같은 디렉터리의 개별 테스트에서 검증한다.

전체 테스트는 `classicfy-ai` 디렉터리에서 실행한다.

```bash
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -v
```

`integration/test_asap_pipeline.py`의 실제 ASAP 테스트는 저장소 상위
`datasets/ASAP`을 기본 경로로 사용한다. 데이터셋이 없으면 해당 테스트만 건너뛰며,
다른 위치를 쓰려면 `ASAP_ROOT` 환경 변수를 지정한다.
(n)ASAP note 정렬을 쓰는 테스트는 `datasets/nASAP` 또는 `NASAP_ROOT`를 쓰고, 둘 중
하나라도 없으면 건너뛴다.

`unit/features/test_common_pattern.py`는 작품 공통 중앙값 제거와 원본 보존·support·grid
검증을 다룬다. `integration/test_common_pattern_pipeline.py`는 세 feature의 추출부터
공통 패턴 분리까지 MIDI와 `.match` fixture로 검증하고 실제 ASAP/(n)ASAP 테스트도 제공한다.

`unit/features/test_normalization.py`는 MAD/IQR/SD 수식, 0 scale fallback, mask 보존,
분리 결과 재사용과 작품별 scale 공유를 검증한다. 공통 pipeline 통합 테스트에서도 다섯
D/P/A 채널의 표준화값을 확인한다. 공식 데이터셋 다운로드 후 전체 123개 테스트가 skip 없이
통과했다. 데이터 없이 실행하면 실제 데이터에 의존하는 4개 테스트만 skip된다.

`unit/features/test_embedding_validation.py`는 임베딩 분석의 동점 검색 점수, 순위,
Pedaling 묶음 가중치, feature 제외, 결측 seed 및 입력 불변을 검증한다.
해당 분석 추가 후 전체 130개 테스트가 실제 데이터 환경에서 skip 없이 통과했다.

`unit/features/test_interpretation_examples.py`는 다른 작품 최근접 선택, B·C의 작품 일치,
동점 대조 선택, 표시 백분위의 동점 처리와 입력 보존을 검증한다.
해당 분석 추가 후 전체 134개 테스트가 실제 데이터 환경에서 skip 없이 통과했다.

`unit/features/test_feature_search_roles.py`는 후보 제외·동점 순위/Top 5·Pedaling 묶음 가중치,
검색에서 제외한 feature의 독립 검사·무작위 대조 50% 기준·중간 거리 대조·작품 동일 가중 집계를 검증한다.
해당 분석 추가 후 전체 142개 테스트가 실제 데이터 환경에서 skip 없이 통과했다.
