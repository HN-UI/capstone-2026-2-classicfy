# 현재 1D CNN의 다른 작품 검색 관찰

**같은 후보 작품의 다른 연주보다 가까운 비율: 대표 성향 97.3%, 변화 폭 51.0%, beat별 변화 크기 56.0%.**

18 epoch의 저장 모델을 그대로 사용했다. 모델·학습 목표·검색 가중치를 수정하지 않았다.


사람의 유사도 응답은 아직 없으며 아래 숫자는 feature 기반 관찰이다.

## 먼저 볼 결과

[어떤 특징을 닮게 찾는가](01_feature_agreement.png)

학습에 쓰지 않은 test 8작품·82연주를 query로 사용했다. 전체 570연주 중 query와 같은 작품의

모든 연주를 제외하고, 현재 128차원 임베딩의 cosine 최근접 B를 선택했다.
B의 작품을 고정한 상태에서 그 작품의 나머지 연주들과 비교했다.
예를 들어 B 작품의 다른 연주 10개 중 9개보다 feature 거리가 가까우면 90%다.
이 비율을 82 query에 동일 비중으로 평균했다. 동점은 0.5점이다.

후보 작품의 연주를 무작위로 고르면 이 비교 비율의 기대값은 50%다.

| 관찰 기준 | 같은 후보 작품의 다른 연주보다 가까운 비율 | 다른 연주들의 평균 거리보다 가까운 query |
|---|---:|---:|
| 대표 성향 | 97.3% | 82/82 |
| 변화 폭 | 51.0% | 46/82 |
| beat별 변화 크기 | 56.0% | 50/82 |
| 대표 성향 + 변화 폭 | 72.8% | 68/82 |

이 비율은 **검색 정확도나 사람이 비슷하다고 평가한 비율이 아니다.**
CNN 후보를 입력 feature의 별도 요약으로 대조한 기술통계다.
현재는 **대표적인 연주 성향을 중심으로 검색하는 경향**이 있다. 변화 폭과 beat별 변화 크기의 일치는 그보다 약하다.
이 지표만으로 연주의 변화 패턴까지 유사하게 찾는다고 설명할 수는 없다.

작품마다 같은 비중으로 평균하면 **대표 성향 97.1% / 변화 폭 48.3% / beat별 변화 크기 56.3%**다. 같은 작품의 여러 query를 독립된 사람 평가로 해석하지 않는다.


## 고정한 세 사례

test 작품을 이름순으로 정렬해 첫·중간·마지막 작품의 중앙 순번 연주를 A로 선택했다.
성공한 사례를 탐색해 고르지 않았다. B는 다른 작품 전체의 CNN 최근접이다.
C는 B와 같은 작품의 연주 중 A와 CNN cosine이 가장 낮은 연주다.
이 대조는 CNN상 차이를 보여주는 사례이며 feature상 모든 축이 멀어야 하는 정답은 아니다.

### 사례 1

- A 기준: `Beethoven/Piano_Sonatas/18-1/Smirnov01.mid`
- B 최근접: `Liszt/Ballade_2/KIM_J04.mid`
- C 같은 후보 작품의 대조: `Liszt/Ballade_2/Sham05.mid`

CNN cosine: A–B **0.924**, A–C **0.487**.

[대표 성향](examples/01_representative.png) · [변화 폭](examples/01_width.png) · [beat별 변화 크기](examples/01_step_rms.png)

| 비교 기준 | A–B 거리 | A–C 거리 | 더 가까운 후보 |
|---|---:|---:|---|
| 대표 성향 | 0.247 | 0.861 | B |
| 변화 폭 | 2.376 | 2.079 | C |
| beat별 변화 크기 | 0.229 | 0.510 | B |

### 사례 2

- A 기준: `Chopin/Etudes_op_25/1/TongB02M.mid`
- B 최근접: `Bach/Fugue/bwv_883/Khmara04.mid`
- C 같은 후보 작품의 대조: `Bach/Fugue/bwv_883/KaiRuiR03.mid`

CNN cosine: A–B **0.889**, A–C **-0.527**.

[대표 성향](examples/02_representative.png) · [변화 폭](examples/02_width.png) · [beat별 변화 크기](examples/02_step_rms.png)

| 비교 기준 | A–B 거리 | A–C 거리 | 더 가까운 후보 |
|---|---:|---:|---|
| 대표 성향 | 0.327 | 1.953 | B |
| 변화 폭 | 2.369 | 2.315 | C |
| beat별 변화 크기 | 0.928 | 0.804 | C |

### 사례 3

- A 기준: `Liszt/Transcendental_Etudes/10/MCVEY02.mid`
- B 최근접: `Beethoven/Piano_Sonatas/21-1_no_repeat/Shi02.mid`
- C 같은 후보 작품의 대조: `Beethoven/Piano_Sonatas/21-1_no_repeat/Zuber01.mid`

CNN cosine: A–B **0.920**, A–C **0.173**.

[대표 성향](examples/03_representative.png) · [변화 폭](examples/03_width.png) · [beat별 변화 크기](examples/03_step_rms.png)

| 비교 기준 | A–B 거리 | A–C 거리 | 더 가까운 후보 |
|---|---:|---:|---|
| 대표 성향 | 0.201 | 0.868 | B |
| 변화 폭 | 0.654 | 0.445 | C |
| beat별 변화 크기 | 0.284 | 0.247 | C |

각 표의 거리는 같은 행 안에서만 비교한다. 대표 성향이 가까워도 변화 폭과 인접 변화가 어긋날 수 있다.
곡이 다르므로 같은 beat 번호의 값을 직접 맞춰 비교하지 않았다.

## 비교값의 의미

- **대표 성향:** 7채널 평균, Rubato만 절댓값 중앙값. 부호가 있는 값은 자기 작품의 공통 패턴 대비 경향이다.
  서로 다른 작품의 절대 BPM·음량이 같다는 뜻은 아니다.
- **변화 폭:** 7채널 각각 `p95-p5`. 시간 순서는 사용하지 않는다.
- **beat별 변화 크기:** 원래 시간축에서 인접한 두 유효 beat 차이의 RMS. 결측을 건너 이어 붙이지 않는다.
  이 지표 하나로 변화의 방향·순서·음악적 구절의 유사성을 측정할 수는 없다.
- **대표 성향 + 변화 폭:** 기존 14차원 통계 요약의 거리. 이 조합이 변화 순서를 담는 것은 아니다.

각 거리에서 Tempo/Rubato/Dynamics/Articulation/Pedaling 다섯 feature의 비중을 동일하게 두었다.
Pedaling의 세 채널은 한 묶음의 평균이다. 이 거리로 검색 후보를 다시 선택하지 않았다.
전체 다른 작품 후보 평균과 후보 작품 내부 평균을 모두 CSV에 기록했다.
작품마다 같은 비중으로 집계한 결과는 `stats.json`의 `test_work_macro`에 있다.

## 조건과 한계

- 입력 reference·정규화는 [기존 구현](../../src/embedding/README.md) 그대로다. 작품별 후보 cohort가 common/scale에 포함된다.
- test 작품은 모델 가중치 학습·checkpoint 선택에 사용하지 않았다. gallery에는 train/validation/test가 모두 포함된다.
- 전곡은 64beat 구간 벡터의 평균·표준편차 128차원이다. 전곡의 구간 배치 순서는 보존하지 않는다.
- 저장 checkpoint에서 570연주 벡터를 다시 추출해 기존 벡터와 일치하는지 확인했다.
- feature상 유사성이 귀로 느끼는 연주 성향·선호 유사성의 정답은 아니다. 같은 연주자 식별 평가도 아니다.
- [기존 청취 자료](../_listening_evaluation/README.md)는 통계 임베딩으로 선정한 다른 문항이며 실제 응답은 아직 없다.
  그 자료의 B/C를 이번 CNN 검색의 청취 정답으로 사용하지 않았다.

## 산출물과 재현

- [query_metrics.csv](query_metrics.csv): test연주 × 4관찰 기준의 후보/거리/비교비율
- [work_metrics.csv](work_metrics.csv): test작품별 동일 지표 평균
- [feature_summaries.csv](feature_summaries.csv): 전체연주·7채널 대표값/폭/인접 변화 RMS/지원 수
- [examples.csv](examples.csv): A/B/C 임베딩 유사도와 feature별 거리
- [top5.csv](top5.csv): test연주의 다른 작품 Top5
- [stats.json](stats.json): 조건·source/cache/checkpoint/code hash·집계·선정 기록
- [실행 스크립트](../../scripts/observe_temporal_neighbors.py)

저장소 루트에서 기존 checkpoint를 읽어 실행한다. 결과를 덮어쓰지 않도록 새 출력 폴더를 지정한다.

```bash
.venv/bin/python classicfy-ai/scripts/observe_temporal_neighbors.py \
  --run ../datasets/temporal_embedding_run01 \
  --out /tmp/classicfy-temporal-neighbors
```
