# 다른 작품 유사 연주 검색에서 다섯 feature의 역할

## 이번 분석이 답하는 질문

**다른 작품의 후보를 찾을 때, 각 feature가 검색 결정을 바꾸고 연주별 차이를 설명하는가?**
기존 14차원 수치 요약 임베딩·5묶음 동일 가중 거리·작품별 정규화를 고정했다.
학습된 신경망이나 청취 정답은 사용하지 않았다. 낮은 거리를 다시 검색 정확도라고 부르지 않는다.
이 보고서는 **검색 기여/민감도**, **feature 간 일치 정도**를 분리해서 보여준다.

## 청중에게 보여줄 순서

1. [한 feature를 빼면 기존 B의 순위가 어떻게 바뀌는가?](examples/01_anchor_rank.png)
   연주 한 개의 구체적인 예시로 질문을 소개한다. B의 순위가 움직이면 해당 feature가 검색 결정에 참여한다.
2. [B와 C를 구분하는 실제 feature는 무엇인가?](examples/02_anchor_advantage.png)
   오른쪽 막대는 B를 지지, 왼쪽은 C를 지지한다. 모든 feature가 B를 지지해야 할 필요는 없다.
3. [전체 작품에서도 검색 후보가 바뀌는가?](global/01_neighbor_changes.png)
   특정 연주에만 해당하는 현상인지, 55개 작품/grid 전체에서도 나타나는지 확인한다.
4. [검색에 쓰지 않은 feature도 닮았는가?](global/03_heldout_agreement.png)
   후보를 고른 기준과 평가 기준을 나눠, 같은 feature로 고르고 같은 feature로 칭찬하는 순환을 피한다.

위 네 장이 발표 핵심이다. 모든 PNG는 파일당 단일 그래프이고 나머지는 해석을 보충한다.

## 결과부터 보기

아래는 **작품별 평균을 낸 뒤 작품/grid에 같은 가중치**를 준 결과다.
괄호는 기준 작품/grid 재표본 95% 구간이며, 전체 후보 corpus를 고정한 기술통계다.

| feature | 제외하면 첫 후보가 바뀜 | 기존 Top 5가 유지됨 | 다른 네 축으로 고른 후보가 제외한 축에서도 가까움 |
|---|---:|---:|---:|
| Tempo | 68.7% | 42.8% | 61.1% (57.6~64.5) |
| Rubato | 46.5% | 62.3% | 60.0% (56.8~63.3) |
| Dynamics | 55.1% | 56.8% | 52.3% (49.7~54.9) |
| Articulation | 59.5% | 49.0% | 54.0% (51.0~56.6) |
| Pedaling | 64.1% | 49.9% | 52.3% (49.1~55.6) |

첫 번째 열이 크면 **검색 결정에 많이 참여한다**. 성능이 좋아졌다는 뜻은 아니다.
두 번째 열이 높으면 첫 후보가 바뀌어도 후보 목록 상당 부분은 유지된다는 뜻이다.
세 번째 열은 선택된 연주가 **그 후보 작품의 다른 연주**보다 사용하지 않은 feature에서도 가까운 비율이다.
무작위로 해당 작품의 연주를 고르면 동점을 포함한 기대값은 50%다.
이 실험에서 Tempo·Rubato의 일치 경향이 상대적으로 크고, 나머지 축은 50%에 더 가깝다.
Dynamics·Pedaling의 재표본 구간은 50%를 포함한다. Articulation은 54.0%로 약한 경향이다.
여러 축을 함께 쓰는 검색은 가능하지만, **모든 축이 하나의 일관된 해석 성향을 잡았다고 결론낼 수는 없다.**

## 분석 기준을 이렇게 정한 이유

### 데이터와 기준 연주

- 기존 원본 캐시 1,036개 정렬 연주에서 robust note alignment·동일 score grid·후보 수 ≥ 5,
  7채널 공통 유효 beat ≥ 32·비율 ≥ 50%를 통과한 **55개 작품/grid, 570개 연주**를 쓴다.
- 일부 좋은 예시만 기준으로 삼지 않고, **모든 연주를 한 번씩 기준 A**로 쓴다.
- A와 같은 작품은 검색 후보에서 모두 제외한다. 다른 반복/grid도 query 작품의 후보로 되돌려 넣지 않는다.
- B의 같은 작품 대조와 표시용 백분위는 B와 동일한 score grid의 연주만 사용한다.
- 공통값·scale·유효 beat·후보 목록은 feature 제거 전후 동일하다. 뺀 뒤 다시 정규화하지 않는다.
- 후보 수가 많은 작품이 결과를 지배하지 않도록 query 평균 → 기준 작품/grid 평균 → 작품 동일 가중으로 집계한다.
  평균 순위 비교에는 후보 수 차이를 보정한 `(B순위−1)/(후보수−1)`도 저장한다.
- 기준 작품/grid를 단위로 2000회 재표본한다. seed는 20261005다.
  query들은 독립 표본으로 간주하지 않는다. 후보 작품이 중복되는 의존성은 남아 있어 모집단 성능 CI로 해석하지 않는다.

### 검색 표현과 정규화

Tempo의 기존 individual, Rubato의 기존 relative를 유지한다. D/A/P는 위치별 common median을 제거한 residual이다.
각 작품/grid에서 T/R/D/A는 MAD, Pedaling depth/on/changes는 각각 SD로 나눈다.
MAD는 `1.4826 × median(|r−median(r)|)`, 0이면 IQR/1.349 → SD → unit 1이다.
residual 재중앙화·clipping·평활화는 없다. mask와 지원 수는 기존 규칙을 따른다.

7채널 각각의 성향(평균, Rubato만 절댓값 중앙값)과 `p95−p5` 폭을 요약해 14좌표를 만든다.
Tempo/Rubato/Dynamics/Articulation은 각 2좌표, Pedaling은 6좌표지만 각 묶음 내부를 평균한다.
**거리² = 다섯 feature 묶음의 평균 제곱 차이의 평균**이므로 Pedaling 좌표 수가 더 많다고 가중치가 커지지 않는다.
원래 단위가 아니라 작품 안에서의 상대적 편차를 비교한다. 표시용 profile 백분위는 거리 입력이 아니다.

### 1. 제거 실험: 검색에 참여하는가?

전체 feature에서의 최근접 B를 고정하고, 하나씩 뺀 검색에서 B의 순위·첫 후보 변경·Top 5 유지율을 계산한다.
feature 단독 검색도 CSV에 저장했다. Tempo+Rubato, Dynamics+Articulation 동시 제거도 확인한다.
Pedaling 제거는 세 하위 지표/6좌표 전체를 뺀다.

| 그림 | 청중에게 설명할 문장 |
|---|---|
| [첫 후보 변경](global/01_neighbor_changes.png) | “이 특징을 없애면 실제로 다른 연주를 고르는 경우가 이만큼 있습니다.” |
| [Top 5 유지](global/02_top5_retained.png) | “첫 후보의 변화와 후보 목록 전체의 변화는 구분해서 봤습니다.” |
| [두 feature 동시 제거](global/04_pair_removal.png) | “서로 연관된 특징을 함께 없앴을 때 검색이 어떻게 달라지는지도 확인했습니다.” |

변경률을 feature 중요도/정확도 순위로 바꾸지 않는다. 제거하면 거리가 바뀌는 것은 구성상 자연스럽다.
특히 **B는 정답 레이블이 아니다.** B의 순위 하락은 B를 선택한 이유를 보여줄 뿐이다.

### 2. 후보 간 차이 설명: 어떤 축이 B를 지지하는가?

이전 사례의 기준 A를 그대로 이어 쓴다.
A=`Bach/Fugue/bwv_848/Lou01M.mid`, B=`Bach/Prelude/bwv_857/ToA01M.mid`다.
새 대조 C=`Bach/Prelude/bwv_857/WangA01M.mid`는 B와 같은 작품의 나머지 연주를 A와의 전체 거리로 정렬한 **중간 거리** 후보다.
짝수가 되면 아래쪽 중간 순위를 쓴다. 이전 보고서의 최원거리 C와 목적이 다르며 원래 보고서는 보존했다.

| 조건 | 기존 B 순위 | 해당 검색의 첫 후보 |
|---|---:|---|
| 다섯 feature 전체 | 1위 | `Bach/Prelude/bwv_857/ToA01M.mid` |
| Tempo 제외 | 28위 | `Beethoven/Piano_Sonatas/31-1/Goh02.mid` |
| Rubato 제외 | 8위 | `Chopin/Etudes_op_25/11/WangV02.mid` |
| Dynamics 제외 | 1위 | `Bach/Prelude/bwv_857/ToA01M.mid` |
| Articulation 제외 | 7위 | `Bach/Prelude/bwv_854/WangA01M.mid` |
| Pedaling 제외 | 2위 | `Bach/Fugue/bwv_857/ToA01M.mid` |

각 feature f의 막대는 `제곱차이(A,C,f) − 제곱차이(A,B,f)`다.
양수면 B가 더 가깝고, 음수면 C가 더 가깝다. 다섯 막대 평균은 전체 거리²의 C−B 차이와 정확히 같다.
한 후보를 지지하는 축과 전체 후보에서 다른 경쟁자를 걸러내는 축은 다를 수 있다.
이전 기준 A의 중간 거리 대조에서는 Articulation의 B 지지가 가장 크고, 아래 중간 수준 사례에서는 Pedaling이 가장 크다.
어느 feature가 역할을 하는지는 기준 연주와 비교 상대에 따라 달라진다.

새로운 [중간 수준 사례 profile](examples/03_typical_profile.png), [같은 후보 작품의 거리](examples/04_typical_candidates.png),
[feature별 지지 방향](examples/05_typical_advantage.png)도 함께 제시한다.
이 사례는 기준 작품별 “하나 제거 후 B의 상대순위 변화”가 작품 중앙값에 가장 가까운 작품을 먼저 고르고,
그 안에서 해당 작품 평균에 가장 가까운 query를 고른다. 동점은 파일명 순이다.
A=`Bach/Fugue/bwv_860/Nikiforov05M.mid`, B=`Schubert/Impromptu_op.90_D.899/3/Ko08M.mid`, C=`Schubert/Impromptu_op.90_D.899/3/Hou06M.mid`다.
가장 큰 변화나 가장 예쁜 profile을 찾아 선택하지 않았다. 이 선정 자체도 현재 corpus를 이용한 기술적 예시다.

### 3. 사용하지 않은 feature 검사: 다른 축과 일치하는가?

예를 들어 Tempo를 검사한다면 **Rubato·Dynamics·Articulation·Pedaling만으로** 다른 작품의 B를 찾는다.
그다음 Tempo 단독 거리로 A–B를 계산하고, B 작품의 **다른 모든 연주**와 비교한다.
B가 더 가까우면 1, 같으면 0.5, 더 멀면 0점이다. 대조마다 평균 → query 평균 → 작품 동일 가중 순으로 계산한다.
가장 먼 C 하나와만 비교하지 않는다. 동점은 성공으로 올려 세지 않는다.
50%는 정답을 맞힐 확률이 아니라 **같은 작품의 두 후보 중 어느 쪽이 더 가까운가**를 무작위 선택했을 때의 기준이다.

검색에서 제외한 feature가 평가에 사용되므로, 그 feature를 잘 맞춰 놓고 다시 검사하는 문제는 줄어든다.
다만 **feature 간 통계적 연관 검사**다. Tempo와 Rubato는 같은 타이밍 원자료를 공유하고, 네 축이 다섯 번째 축을 대신할 수도 있다.
높으면 feature 필수성이 증명되는 것이 아니고, 낮으면 독립적인 선호 정보일 수도 있어 불필요하다고 단정하지 않는다.
공통값과 scale에는 corpus 후보가 포함되어 있다. 독립 작품/새 연주에 대한 일반화 실험도 아니다.

아래 개별 그림의 초록 연주는 **다른 네 축으로 선택된 연주**다.
회색 점 하나는 같은 후보 작품의 다른 연주 하나다. 왼쪽으로 갈수록 해당 축에서 A와 가깝다.
세로 점선은 다른 연주들의 거리 중앙값이다. 23개 후보가 있는 경우에도 이름을 모두 나열하지 않고 분포를 보여준다.
이 축에서 항상 1위일 필요는 없다. 기준 작품 평균이 전체 평균에 가까운 작품을 고르고,
그 작품 평균에 가까운 query를 고른 사례라 실패나 차이도 그대로 보인다.

| 개별 그림 | 기준 A → 네 축으로 선택한 B | 다른 같은 작품 연주보다 가까운 비율 |
|---|---|---:|
| [Tempo](heldout/01_Tempo.png) | `McNamara02` → `Shi02` | 50.0% |
| [Rubato](heldout/02_Rubato.png) | `Khmara01` → `BianF14` | 60.0% |
| [Dynamics](heldout/03_Dynamics.png) | `AbdelmoulaJS03M` → `Min03M` | 50.0% |
| [Articulation](heldout/04_Articulation.png) | `SCHU09` → `GuoE02M` | 50.0% |
| [Pedaling](heldout/05_Pedaling.png) | `Rizikov03M` → `HuangSW04` | 50.0% |

[14좌표와 7좌표의 민감도 비교](global/05_summary_sensitivity.png)는 변화 폭 포함 여부도 검사한다.
7좌표는 각 채널의 성향만 쓴다. 두 표현에서 선택과 검사 좌표가 함께 바뀌므로 수치로 표현의 음악적 우열을 결정하지 않는다.

## 무엇이 확인됐고 무엇을 더 확인해야 하나?

**각 feature는 실제 검색 결정과 후보 간 구분에 참여한다.** 연주 차이를 여러 축으로 설명할 수 있다.
그러나 “다섯 개가 모두 필요하다”, “음악적으로 유사한 연주를 정확하게 찾는다”, “추천 성능이 향상됐다”는 결론은 아직 없다.
Dynamics/Articulation 폭에는 작품 정보가 남을 수 있고, MIDI velocity·CC64는 장치 보정, note-off는 실제 울림과 다를 수 있다.
독립 청취자가 유사하다고 고른 후보 또는 사용자의 선호를 평가 기준으로 삼아 최종 제거 실험을 해야 한다.
특징이 달라지면 선택이 달라지는 것과, 선택이 사용자에게 도움이 되는 것을 구분한다.

## 재현과 원수치

저장소 루트에서 실행한다. 기본 데이터 경로는 저장소 밖 `../datasets/ASAP`, `../datasets/nASAP`, raw cache다.

```bash
.venv/bin/python classicfy-ai/scripts/validate_feature_search_roles.py
```

`--anchor-key`, `--seed`, `--bootstrap-samples`, `--out`과 데이터 경로 옵션을 제공한다.
PNG·CSV·JSON·이 문서를 같이 재생성한다. 계산에 필요한 후보/scale도 저장한다.

| 파일 | 내용 |
|---|---|
| [embeddings.csv](embeddings.csv) | 모든 연주 14좌표. |
| [search_changes.csv](search_changes.csv) | 모든 query/구성의 후보·B 순위·동점·Top 5 유지. |
| [heldout_agreement.csv](heldout_agreement.csv) | 검색에 쓰지 않은 축의 query별 비교 결과. |
| [work_search_changes.csv](work_search_changes.csv), [work_heldout_agreement.csv](work_heldout_agreement.csv) | 작품별 집계. |
| [summary.csv](summary.csv) | 작품 동일 가중 평균·재표본 구간. |
| [examples.csv](examples.csv) | B·C 차이의 feature별 정확한 분해. |
| [scales.csv](scales.csv), [cohort_audit.csv](cohort_audit.csv) | 정규화 scale과 작품 포함/제외 조건. |
| [stats.json](stats.json) | 데이터 hash·revision·선정 조건·해석 범위. |

MIDI 원자료와 같은 작품 곡선은 기존 [연주 차이/검색 사례 보고서](../_interpretation_embedding/README.md)에 있다.
이전 그림이나 원본 feature/정규화 정책은 수정하지 않았다.
