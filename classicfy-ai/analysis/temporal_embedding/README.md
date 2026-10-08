# 1D CNN 20 epoch 학습과 연주 흐름 복원 확인

**검증 복원 오차는 감소했다. 시간 순서를 일부 활용하지만, 세밀한 beat별 변화는 아직 잘 복원하지 못한다.**

동일 seed·작품 split·검증 마스크로 실제 20 epoch를 학습했다. 검증 MSE는
초기 **1.5752 → 2 epoch 1.3856 → 최선 18 epoch 1.3029**로 감소했다.
초기 대비 17.3%, 2 epoch 대비 6.0% 감소다. 20 epoch에서는 1.3077로 조금 올라갔으므로
epoch마다 계속 감소하지는 않는다. 검색·평가에는 검증 오차가 가장 낮은 18 epoch 모델을 사용했다.

## 먼저 볼 그림

| 그림 | 읽는 방법 |
|---|---|
| [01 epoch별 오차](01_epoch_loss.png) | 파란 검증 곡선이 내려가고, 18 epoch에서 최저점을 기록한다. |
| [03 구간 내 변화 모양](03_shape_correlation.png) | 구간별 평균을 제거해도 실제 곡선의 모양이 맞는지 본다. 1이면 완전히 같은 모양이다. |
| [05 입력 순서 영향](05_order_diagnostic.png) | 관측 순서를 섞으면 오차가 1.303 → 1.444로 늘어난다. |
| [06 상승·하강 방향](06_direction_accuracy.png) | 숨긴 연속 beat의 상승·하강 일치는 약 51%로, 비교용 무작위 방향 기대값 50%와 가깝다. |
| [07 변화 크기](07_variation_amplitude.png) | 모델이 예측한 구간 내 변화 크기는 실제의 16~32%로 작다. |
| [02 feature별 값 오차](02_feature_errors.png) | 학습 전보다 각 채널의 오차가 줄지만, 구간 평균 기준보다 모든 채널이 좋은 것은 아니다. |
| [04 인접 beat 변화량 오차](04_change_errors.png) | 숨긴 두 beat 사이의 변화량 오차는 구간 평균 예측보다 개선되지 않았다. |

## 실제 학습 조건

- 모델·입력은 [구현 안내](../../src/embedding/README.md)의 기본 설정을 유지했다.
- 64 beat·7채널, 64차원 bottleneck, 양방향 dilated CNN 4블록, AdamW learning rate 0.001.
- batch 32, dropout 0.1, seed 20261006, CPU 2threads.
- 유효 beat 약 20%를 최대 4beat 연속 블록으로 숨겼다. 결측·padding은 정답에서 제외했다.
- 학습 39작품·395연주·5803구간, 검증 8작품·93연주·935구간, test 8작품·82연주·1670구간.
- 검증 작품과 숨긴 위치는 전체 epoch에서 고정했다. 작품·정렬/grid 변형은 split을 공유한다.
- 20 epoch를 모두 관찰하기 위해 `--patience 20`을 사용했다. seed·모델·학습률·마스킹은 2 epoch 실행과 같다.
- 작품별 common/scale은 해당 작품의 입력 후보 전체를 reference로 사용하는 기존 정책이다.
  모델 학습 작품과 검증/test 작품은 겹치지 않지만, reference가 없는 새 단독 연주 추론 평가는 아니다.

## epoch에 따른 검증 오차

| Epoch | 검증 MSE |
|---:|---:|
| 0 (학습 전) | 1.575197 |
| 2 | 1.385568 |
| 5 | 1.342774 |
| 10 | 1.316490 |
| 15 | 1.308661 |
| 18 (최선) | 1.302937 |
| 20 | 1.307689 |

전체 epoch·채널별 검증 오차는 [history.csv](history.csv)에 있다.
Test는 최선 모델을 선택한 뒤 평가했으며 MSE는 **1.336736**이다.
학습과 검증은 서로 다른 작품이고 학습 마스크도 변하므로, 두 곡선의 절대 높이를 직접 비교하지 않는다.

## 값 복원과 흐름 복원은 다르다

아래 MSE는 각 채널의 숨긴 유효 값을 모아 계산한다. Tempo/Rubato/Dynamics/Articulation
네 채널과 Pedaling 세 채널 평균을 다섯 feature 블록으로 동일 가중 평균한다.

| 방법 | 값 MSE | 숨긴 인접 beat 변화량 MSE |
|---|---:|---:|
| 학습 전 모델 | 1.575197 | 2.023691 |
| 2 epoch 모델 | 1.385568 | 2.049514 |
| 최선 18 epoch 모델 | **1.302937** | 2.027769 |
| 0 예측 | 1.569728 | 2.014847 |
| 보이는 값의 구간 평균 | 1.380311 | **2.014847** |
| 보이는 양끝의 직선 보간 | 1.746897 | 2.060576 |
| 최선 모델·관측 순서 섞음 | 1.444027 | 2.046970 |

최선 모델은 구간 평균 예측보다 값 MSE가 **5.6% 낮다**. 검증 8작품 중 7작품에서 구간
평균 예측보다 좋았고, BWV857에서는 구간 평균이 더 좋았다. 작품마다 같은 비중을 두면
모델 MSE 1.332893 / 구간 평균 1.380679이며, 위 표는 채널별 target을 pooling한 결과다.
두 집계는 작품 길이·구간 수의 비중이 다르다. [work_summary.csv](work_summary.csv)에서 확인할 수 있다.

## 시간 흐름을 얼마나 읽는가

**입력 순서 활용:** 보이는 7개 feature 묶음을 같이 섞어 관측값의 집합을 보존하고,
숨긴 정답과 mask는 고정했다. 모델 값 MSE가 **10.8% 증가**했다. 단순 평균은 순서가
바뀌어도 그대로이므로, 모델이 순서·위치 정보를 일부 활용하는 징후다.
다만 학습하지 않은 입력 변형에 대한 진단이며, 순서를 사용하지 않는 모델을 별도로
학습한 대조 실험이나 인과적 증명은 아니다.

**구간 내 모양:** 각 구간의 hidden 실제 값과 예측 값에서 각각 해당 구간의 hidden 평균을
뺀 뒤 상관을 계산했다. 연주마다 평균값만 잘 맞추는 효과를 제거한 지표다.
모델의 상관은 **0.075~0.326**으로 일부 관계가 있지만 강하지 않다.
강약·페달 깊이·사용 비율 등에서는 직선 보간의 모양 상관이 더 높다.
구간 평균 예측은 상수이므로 이 상관이 정의되지 않으며, 그림에서만 0으로 표시한다.

**변화 크기와 방향:** 중심을 제거한 예측값의 표준편차는 실제의 **16~32%**다.
숨긴 연속 beat의 실제 변화량이 0이 아닌 경우에 상승·하강 일치를 계산하면 **50.7~52.4%**다.
변화량 MSE도 구간 평균 기준보다 개선되지 않았다. 현재 모델은 급격한 변화보다
평균에 가까운 완만한 값을 예측하는 경향이 있다.

| 채널 | 구간 내 모양 상관 | 변화 크기 / 실제 | 방향 일치 |
|---|---:|---:|---:|
| Tempo | 0.205 | 27.9% | 51.3% |
| Rubato | 0.205 | 24.2% | 51.3% |
| Dynamics | 0.326 | 31.7% | 50.8% |
| Articulation | 0.098 | 17.7% | 51.2% |
| Pedal depth | 0.237 | 27.7% | 51.2% |
| Pedal down ratio | 0.195 | 25.1% | 50.7% |
| Pedal changes | 0.075 | 16.4% | 52.4% |

이 결과로 **“시간 흐름을 일부 반영하지만 beat별 변화까지 잘 복원한 것은 아니다”**라고
설명할 수 있다. 복원 오차 감소만으로 연주 성향 검색이 사람의 판단에 맞는다고 주장하지 않는다.

## 고정된 대표 구간

잘 복원한 사례를 찾지 않았다. 검증 작품 이름순 첫·중간·마지막 작품에서 중앙 순번 연주를
선택하고, 그 연주의 중앙에 가장 가까운 구간을 사용했다. 각 사례에 7채널 그림이 있다.

| 사례 | 연주·시작 beat | 예시 그림 |
|---|---|---|
| 1 | Bach BWV860 · TuanS01M · 64 | [Tempo](examples/01_tempo.png) · [Dynamics](examples/01_dynamics.png) |
| 2 | Beethoven Sonata 31-1 · Kavalerova01 · 128 | [Tempo](examples/02_tempo.png) · [Dynamics](examples/02_dynamics.png) |
| 3 | Chopin Op.25 No.11 · MiyashitaM03M · 160 | [Tempo](examples/03_tempo.png) · [Dynamics](examples/03_dynamics.png) |

검정=실제 곡선, 검정 X=숨긴 정답, 파랑=최선 모델, 회색=초기 모델,
보라=관측 구간 평균, 주황=직선 보간이다. 예측선은 숨긴 위치에만 표시하며 서로 떨어진
숨긴 구간을 이어 그리지 않았다. 실제 급격한 변화에 비해 파란선이 완만한 경우를 볼 수 있다.

## 산출물과 재현

- [metrics.csv](metrics.csv): 방법×7채널의 값·모양·변화량·방향·지원 수
- [work_metrics.csv](work_metrics.csv): 작품별 동일 지표
- [work_summary.csv](work_summary.csv): 작품별 5feature 동일 가중 요약
- [example_beats.csv](example_beats.csv): 대표 구간의 실제·각 방법 예측값
- [stats.json](stats.json): 모델·cache·평가 코드 hash, 조건, 선정, 집계
- [training.json](training.json): 원 학습 run의 모델·학습 설정, best epoch, 초기/최선 검증 오차
- [test_reconstruction.json](test_reconstruction.json): 최선 모델의 test 복원 오차 원기록
- [재현 스크립트](../../scripts/validate_temporal_embedding.py)

학습 checkpoint와 대용량 벡터는 저장소 밖 `Classicfy/datasets/temporal_embedding_run01`에 있다.
저장소 루트에서 새 출력 폴더를 지정해 실행한다.

```bash
.venv/bin/python classicfy-ai/scripts/train_temporal_embedding.py \
  --epochs 20 --patience 20 --out ../datasets/temporal_embedding_run02

.venv/bin/python classicfy-ai/scripts/validate_temporal_embedding.py \
  --run ../datasets/temporal_embedding_run02 \
  --early-run ../datasets/temporal_embedding_smoke \
  --out /tmp/classicfy-temporal-evaluation
```

기본 출력 폴더에 기존 결과가 있으면 덮어쓰지 않는다. 학습 없이 저장 모델만 평가할 때는
두 번째 명령의 `--run`을 기존 run01로 지정한다.
지표 계산과 순서 섞기의 의미는 `tests/unit/embedding/test_temporal_evaluation.py`에서 검증한다.
