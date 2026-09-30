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
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`integration/test_asap_pipeline.py`의 실제 ASAP 테스트는 저장소 상위
`datasets/ASAP`을 기본 경로로 사용한다. 데이터셋이 없으면 해당 테스트만 건너뛰며,
다른 위치를 쓰려면 `ASAP_ROOT` 환경 변수를 지정한다.
(n)ASAP note 정렬을 쓰는 테스트는 `datasets/nASAP` 또는 `NASAP_ROOT`를 쓰고, 둘 중
하나라도 없으면 건너뛴다.
