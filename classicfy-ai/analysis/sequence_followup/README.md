# 후속 실험 1·3: 학습 예산과 BiLSTM 압축 차원

요청한 **1번(최대 80 epoch 학습)**과 **3번(BiLSTM 64/128/256차원 비교)**의 실제 실행 결과입니다.
같은 조건의 3 seed를 사용하며, 복원 오차와 시간 변화의 복원을 함께 비교합니다.

- [1번: 학습 예산 확대](budget/README.md)
- [3번: BiLSTM 임베딩 차원 비교](bottleneck/README.md)


## 공통 조건

train 39작품·395연주·5803구간, validation 8작품·93연주·935구간.
64-beat 구간·stride 32·기존 7채널의 정규화와 cohort reference를 그대로 사용했습니다.
seed 20261006/07/08, AdamW lr 0.001, batch 32, dropout 0.1, clipping 1, CPU 2 threads.
학습 시 유효 위치 약 20%를 최대 4-beat 연속 블록으로 숨깁니다. 결측·padding은 정답에서 제외합니다.
최소 20 epoch 이후 검증 값 MSE가 10회 연속 엄격히 개선되지 않으면 종료하며 최대 80 epoch입니다.
검증 mask는 모든 실행에서 고정합니다. **test는 평가·모델 선택에 사용하지 않았습니다.**

## 실제 선택과 종료

| 모델 | seed | 최선 epoch | 학습 종료 epoch | 종료 사유 |
|---|---:|---:|---:|---|
| 1D CNN 64D | 20261006 | 18 | 28 | 10회 무개선 |
| 1D CNN 64D | 20261007 | 21 | 31 | 10회 무개선 |
| 1D CNN 64D | 20261008 | 20 | 30 | 10회 무개선 |
| BiLSTM 64D | 20261006 | 23 | 33 | 10회 무개선 |
| BiLSTM 64D | 20261007 | 21 | 31 | 10회 무개선 |
| BiLSTM 64D | 20261008 | 31 | 41 | 10회 무개선 |
| Transformer 64D | 20261006 | 34 | 44 | 10회 무개선 |
| Transformer 64D | 20261007 | 48 | 58 | 10회 무개선 |
| Transformer 64D | 20261008 | 63 | 73 | 10회 무개선 |
| BiLSTM 128D | 20261006 | 23 | 33 | 10회 무개선 |
| BiLSTM 128D | 20261007 | 30 | 40 | 10회 무개선 |
| BiLSTM 128D | 20261008 | 23 | 33 | 10회 무개선 |
| BiLSTM 256D | 20261006 | 24 | 34 | 10회 무개선 |
| BiLSTM 256D | 20261007 | 21 | 31 | 10회 무개선 |
| BiLSTM 256D | 20261008 | 23 | 33 | 10회 무개선 |

## 비교를 어떻게 보장했는가?

각 seed의 원래 DataLoader 순서와 mask를 미리 생성해 모든 모델에 재사용했습니다.
실제 각 epoch에서 사용한 batch index와 hidden mask를 다시 hash하여 같은지 검사했습니다.
첫 20 epoch schedule의 SHA-256은 기존 실험과 정확히 같아야 합니다.
64D 모델의 첫 20 epoch 검증 곡선은 원래 실행과 최대 절대 차이 2×10⁻⁶ 이내인지 검사합니다.
초기·20 epoch 내 최선·20 epoch·확대 예산 최선·마지막 checkpoint를 별도로 보존합니다.
중단 후에는 optimizer와 dropout RNG를 복원하며, 중단 없이 실행한 결과와 같은지 자동 테스트했습니다.

## 지표와 figure 해석

- **임베딩·압축 차원**: 인코더가 한 64-beat 구간을 요약해 복원기로 전달하는 숫자의 개수입니다.
  64D는 64개, 128D는 128개, 256D는 256개입니다. 원래 입력의 beat 수나 feature 수를 늘린 것이 아닙니다.
- **epoch**: 학습용 5803구간을 한 번씩 사용한 단위입니다. 검증 작품은 gradient 계산에 사용하지 않습니다.
- **조기 종료·checkpoint**: 고정 검증 오차가 10회 연속 낮아지지 않으면 학습을 멈춥니다.
  checkpoint는 특정 epoch의 저장 가중치이며, 마지막 가중치 대신 최저 검증 오차의 가중치를 선택합니다.
- **값 MSE**: hidden이면서 유효한 위치의 제곱 오차. 낮을수록 좋습니다.
- **모양 상관**: 각 64-beat 구간의 hidden 평균을 실제·예측에서 각각 제거한 뒤 상관 계산.
  서로 떨어진 hidden 블록도 같은 64-beat 구간에 있으면 함께 중심을 제거합니다.
  채널별로 중심 제거된 모든 구간의 hidden 값을 합쳐 하나의 상관을 구합니다. 구간별 상관의 단순 평균이 아닙니다.
  범위는 −1~1이며, 1은 같은 방향의 선형 변화, 0 부근은 약한 선형 관계, −1은 반대 방향입니다.
  예측이 일정해 표준편차가 0이면 상관을 정의하지 않습니다. 모양 상관만으로 변화 크기의 일치를 뜻하지 않습니다.
- **변화 폭 비율**: 같은 중심 제거 값의 예측 표준편차 / 실제 표준편차 ×100. 100%는 크기의 일치입니다.
  20%라면 실제의 약 1/5 변화 폭이며, 100%를 넘으면 실제보다 큰 변화 폭입니다.
- **변화량 MSE**: 원래 이웃한 두 beat가 모두 hidden·유효할 때 실제와 예측의 차분 오차.
- **방향 일치**: 그 인접 쌍 중 |실제 변화|>10⁻⁶인 쌍에서 부호가 같은 비율. 50%는 균등 무작위 부호 기준입니다.
- **보이는 값 평균 기준선**: 각 구간·채널에서 가리지 않은 유효 값의 평균으로 hidden 위치를 모두 예측합니다.
- **직선 보간 기준선**: 그 관측점들을 원래 beat 위치에서 직선으로 연결합니다. 양 끝 바깥은 최근접 관측값,
  관측점이 전혀 없는 경우는 0을 사용합니다. 두 기준선 모두 가린 정답을 계산에 사용하지 않습니다.

채널별 지표를 먼저 구하고 Tempo/Rubato/Dynamics/Articulation/Pedaling의 5그룹을 동일 비중으로
평균합니다. 페달 세 채널은 한 그룹입니다. 점은 seed 평균, 범위는 최소~최대이며 신뢰구간이 아닙니다.
그림에 사용하는 고정 사례는 기존 검증 작품 이름순 첫·중간·마지막의 중앙 연주·중앙 구간입니다.
위 명칭은 복원 결과를 비교하는 통계량이며, 각 계산 방법을 함께 기록했습니다.

## 범위와 한계

이미 관찰한 하나의 validation split에서 개발한 결과입니다. 새로운 독립 holdout 결과가 아닙니다.
최선 epoch 선택과 결과 요약에 같은 validation을 사용하므로 확정적인 일반화 성능으로 주장하지 않습니다.
작품별 공통 패턴·scale은 기존의 해당 작품 reference cohort 전체 연주로 계산한 값을 재사용합니다.
따라서 새 작품의 한 연주만 들어오는 상황에서의 성능을 검증한 것은 아닙니다.
early stopping 때문에 실제 epoch 수와 계산량이 다릅니다. 마지막 epoch까지 계속 개선되면 예산 상한의 영향이 남습니다.
차원 확대는 head 파라미터 수와 초기화를 함께 바꾸므로 순수한 정보 용량만의 효과로 단정하지 않습니다.
학습 인코더가 좋아져도 구간 평균·표준편차로 전곡을 요약하면 전곡의 구간 배치 순서는 잃습니다.
이 실험은 복원 진단이며 추천 품질·사람의 청취 유사도를 입증하지 않습니다.

## 파일과 재현

`summary.csv`, `channel_metrics.csv`, `work_metrics.csv`, `history.csv`, `example_beats.csv`,
`training_runs.json`, `protocol.json`과 PNG/SVG figure를 제공합니다.
자동 테스트 기록은 [unit_tests.txt](unit_tests.txt)에 있습니다. 산출물 검증은 아래 명령으로
`audit.json`에 기록합니다.
`layout_audit.json`은 제목·축·범례·주석이 캔버스 안에 들어오는지 검사합니다.
글자 간 겹침을 모두 검출하지는 않으므로 대표 그림의 직접 확인을 함께 수행했습니다.
대용량 mask schedule과 checkpoint는 `Classicfy/datasets/extended_sequences_run01`에 보존합니다.
기존 20 epoch 산출물은 덮어쓰지 않습니다. protocol이 같으면 완료 실행을 재사용하고 중단 실행은 이어갑니다.

```bash
XDG_CACHE_HOME=/tmp/classicfy-cache MPLCONFIGDIR=/tmp/classicfy-mpl \
classicfy-ai/.venv/bin/python classicfy-ai/scripts/extend_sequence_experiments.py --stage all
XDG_CACHE_HOME=/tmp/classicfy-cache MPLCONFIGDIR=/tmp/classicfy-mpl \
classicfy-ai/.venv/bin/python classicfy-ai/scripts/check_extended_figure_layout.py
classicfy-ai/.venv/bin/python classicfy-ai/scripts/audit_extended_sequences.py
```

[이전 20 epoch 모델 비교](../sequence_autoencoders/README.md)
