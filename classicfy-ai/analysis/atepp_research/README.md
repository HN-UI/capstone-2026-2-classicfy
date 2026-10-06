# ATEPP: 연구 기반 정렬 후 피처 재추출·검증

## 결과

ASAP 비교에서 선택한 **dual_dtw**을 ATEPP 악보–연주 MIDI에 적용했다. [선택 근거와 독립 beat 검증](../alignment_comparison/README.md).

원본 11674개 연주를 감사했다. 정렬 완료(quality gate 통과) **2714개**, 동일 작품 공통 패턴 제거·정규화 **2692개 / 233개 작품**이다. exact duplicate MIDI를 제외한 분석 데이터는 2691개다. **수치·shape·mask·domain 오류 0건**이다. 이것은 정렬/전사 오차가 없다는 뜻은 아니다.

| 상태 | 연주수 |
| --- | --- |
| no_score_midi | 5098 |
| excluded_quality | 2845 |
| aligned | 2714 |
| rejected_alignment | 997 |
| invalid_score_midi | 20 |

정렬기는 공식 구현을 사용했으며 임의 pitch-class 휴리스틱을 fallback으로 섞지 않았다. DualDTW는 학습 가중치를 사용하지 않는 알고리즘이다. 비교 대상 TheGlueNote에는 공개 small checkpoint를 사용했다. 실패·품질 기준 미달 사례는 제외 기록을 남겼다. 악보가 없는 연주는 악보 의존 피처를 생성하지 않는다.

## 파이프라인과 저장 파일

`score MIDI + performance MIDI → 선택 정렬기 → note pair → 단조 tempo map → quarter grid → 기존 5개 피처군 → 작품의 공통 패턴 제거 → 정규화 → 동일 검증`

- **5개 피처군**: Tempo, Rubato, Dynamics, Articulation, Pedaling. Pedaling은 depth/down-ratio/changes로 나뉘므로 저장 배열은 7채널이다.
- note 대응에서 같은 악보 onset의 연주 onset 중앙값을 취하고 순증가 부분열·선형 보간으로 시간 지도를 만든다. 입력 MIDI에는 XML의 장식음 정보가 충분하지 않으므로 Articulation의 장식음 제외 한계는 남아 있다.
- 이전과 동일한 quality gate를 사용했다: score 대응률 ≥0.70, performance 대응률 ≥0.60, 범위 coverage ≥0.90, 단조 anchor 유지율 ≥0.90, note 수 비율 0.65~1.6, 자체 map timing p90 ≤0.5박, 16개 이상 interval, 극단 박 길이 비율 ≤2%. 대응률은 **reference precision/recall이 아니다**. map으로 자신을 평가하는 잔차 역시 독립 정확도가 아니다.
- 정렬 전체는 `ATEPP_dataset/alignments/dual_dtw/<perf_id>.npz`에 저장하며 gate 미달이더라도 정렬기가 출력한 대응은 남긴다.
- raw: `ATEPP_dataset/features/dual_dtw/raw/<perf_id>.npz`.
- normalized: `ATEPP_dataset/features/dual_dtw/normalized/<perf_id>.npz`.
- normalized 배열: `raw`, `relative`, `normalized`, `mask`: (T,7), `score_beats`: (T+1). 결측은 NaN+mask로 표현한다. CC64 미제공을 무페달로 해석하지 않는다.
- 기존 휴리스틱 결과와 새 결과를 별도 경로에 보존했다. manifest에는 새 normalized 파일별 SHA256을 기록했다.

## 공통 요소 제거·정규화

같은 composition_id 및 같은 악보 MIDI의 연주를 묶고, 모두 관측되는 공통 악보 범위만 사용한다. 단일 연주 작품은 공통 패턴 추정에서 제외한다. Tempo의 score reference는 quarter당 0.5초이며 위치별 공통 패턴을 뺄 때 상수 reference가 상쇄된다. Rubato는 기존 median 기준 로직을 유지한다. D/A와 T/R은 기존 robust scale, pedal 각 채널은 SD 기준을 유지하며 clipping을 하지 않는다.

후보 pool 전체에서 위치별 중앙값과 작품별 scale을 추정한다. 따라서 **이 검증은 후보 집합에 대한 기술적/transductive 분석**이다. 새로운 작품의 완전히 독립적인 train/test 일반화 성능은 별도 실험이 필요하다.

공통/해석 분산과 차이 보존:

| channel | layer | total_variance | within_work_variance | within_fraction |
| --- | --- | --- | --- | --- |
| tempo | raw | 0.55908 | 0.01686 | 0.03015 |
| tempo | relative | 0.01727 | 0.01687 | 0.97663 |
| tempo | norm | 4.97009 | 3.82089 | 0.76878 |
| rubato | raw | 0.00614 | 0.00081 | 0.13116 |
| rubato | relative | 0.00085 | 0.00080 | 0.94859 |
| rubato | norm | 0.12991 | 0.11245 | 0.86567 |
| dynamics | raw | 0.00534 | 0.00143 | 0.26759 |
| dynamics | relative | 0.00144 | 0.00143 | 0.99164 |
| dynamics | norm | 0.47645 | 0.46526 | 0.97650 |
| articulation | raw | 0.20683 | 0.02150 | 0.10393 |
| articulation | relative | 0.02236 | 0.02149 | 0.96126 |
| articulation | norm | 0.32401 | 0.29402 | 0.90743 |
| pedal_depth | raw | 0.00974 | 0.00234 | 0.23992 |
| pedal_depth | relative | 0.00246 | 0.00234 | 0.94876 |
| pedal_depth | norm | 0.13314 | 0.12801 | 0.96149 |
| pedal_down_ratio | raw | 0.05462 | 0.01172 | 0.21461 |
| pedal_down_ratio | relative | 0.01635 | 0.01172 | 0.71702 |
| pedal_down_ratio | norm | 0.13794 | 0.09509 | 0.68936 |
| pedal_changes | raw | 1.68290 | 0.24439 | 0.14522 |
| pedal_changes | relative | 0.26183 | 0.24439 | 0.93338 |
| pedal_changes | norm | 0.06314 | 0.05707 | 0.90399 |

| checked_performance_pairs | works | maximum_pair_residual_error | pair_errors_above_1e_9 |
| --- | --- | --- | --- |
| 2458 | 233 | 0.00000 | 0 |

공통 패턴을 뺀 뒤에도 연주 간 원래 차이가 유지되는지 검사했다. 전체 pair 검사는 [pair_difference_preservation.csv](pair_difference_preservation.csv)에 있다.

## 수치 무결성·극단값

유효 feature cell 11,178,020개를 검사했다. 극단값은 삭제하지 않고 검수 queue와 별도 순위 민감도 분석에 사용했다.

| channel | valid_cells | missing_cells | over_6 | over_10 | max_abs |
| --- | --- | --- | --- | --- | --- |
| articulation | 1519875 | 107608 | 17502 | 4503 | 286.47879 |
| dynamics | 1524006 | 103477 | 2023 | 717 | 51.99200 |
| pedal_changes | 1627483 | 0 | 1751 | 484 | 53.42444 |
| pedal_depth | 1627483 | 0 | 602 | 0 | 9.95448 |
| pedal_down_ratio | 1627483 | 0 | 129 | 0 | 9.25488 |
| rubato | 1625845 | 1638 | 27284 | 10434 | 1636.73787 |
| tempo | 1625845 | 1638 | 16390 | 7126 | 1922.59900 |

검수 queue: **1152개 연주 / 1949개 채널 행**. max |x|>20 또는 |x|>6의 비율>2%인 채널을 표시했으며 오류 확정 label은 아니다. 작은 작품별 scale, 실제 긴 pause, 전사/정렬 오류가 극단값의 원인일 수 있다. 자동 정렬기 교체만으로 모든 극단값이 정상화된다고 가정하지 않는다.

`|normalized|>6` cell 제외 시 추천 순위 변화:

| test | mean | low | high | works |
| --- | --- | --- | --- | --- |
| rank_spearman | 0.90067 | 0.88605 | 0.91532 | 180 |
| top1_unchanged | 0.83046 | 0.80874 | 0.85260 | 180 |

수치 예외: [invalid_feature_values.csv](invalid_feature_values.csv). 극단값 사례: [extreme_beat_examples.csv](extreme_beat_examples.csv), [feature_review_queue.csv](feature_review_queue.csv).

## 시대·작곡가 효과와 metadata와의 차이

작품 단위로 분리한 5-fold composer/era probe 결과:

| label | layer | model | fold | balanced_accuracy | test_works | test_performances | classes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| composer | raw | feature | 0 | 0.61180 | 43 | 528 | 9 |
| composer | raw | chance | 0 | 0.09352 | 43 | 528 | 9 |
| composer | raw | feature | 1 | 0.66154 | 45 | 533 | 9 |
| composer | raw | chance | 1 | 0.11955 | 45 | 533 | 9 |
| composer | raw | feature | 2 | 0.45393 | 45 | 526 | 9 |
| composer | raw | chance | 2 | 0.09508 | 45 | 526 | 9 |
| composer | raw | feature | 3 | 0.38866 | 47 | 541 | 9 |
| composer | raw | chance | 3 | 0.10446 | 47 | 541 | 9 |
| composer | raw | feature | 4 | 0.35914 | 45 | 536 | 9 |
| composer | raw | chance | 4 | 0.09762 | 45 | 536 | 9 |
| composer | relative | feature | 0 | 0.43054 | 43 | 528 | 9 |
| composer | relative | chance | 0 | 0.09352 | 43 | 528 | 9 |
| composer | relative | feature | 1 | 0.46306 | 45 | 533 | 9 |
| composer | relative | chance | 1 | 0.11955 | 45 | 533 | 9 |
| composer | relative | feature | 2 | 0.38523 | 45 | 526 | 9 |
| composer | relative | chance | 2 | 0.09508 | 45 | 526 | 9 |
| composer | relative | feature | 3 | 0.37881 | 47 | 541 | 9 |
| composer | relative | chance | 3 | 0.10446 | 47 | 541 | 9 |
| composer | relative | feature | 4 | 0.29355 | 45 | 536 | 9 |
| composer | relative | chance | 4 | 0.09762 | 45 | 536 | 9 |
| composer | norm | feature | 0 | 0.34622 | 43 | 528 | 9 |
| composer | norm | chance | 0 | 0.09352 | 43 | 528 | 9 |
| composer | norm | feature | 1 | 0.34045 | 45 | 533 | 9 |
| composer | norm | chance | 1 | 0.11955 | 45 | 533 | 9 |
| composer | norm | feature | 2 | 0.25610 | 45 | 526 | 9 |
| composer | norm | chance | 2 | 0.09508 | 45 | 526 | 9 |
| composer | norm | feature | 3 | 0.30894 | 47 | 541 | 9 |
| composer | norm | chance | 3 | 0.10446 | 47 | 541 | 9 |
| composer | norm | feature | 4 | 0.26430 | 45 | 536 | 9 |
| composer | norm | chance | 4 | 0.09762 | 45 | 536 | 9 |
| era | raw | feature | 0 | 0.53542 | 47 | 539 | 6 |
| era | raw | chance | 0 | 0.18811 | 47 | 539 | 6 |
| era | raw | feature | 1 | 0.65796 | 45 | 538 | 6 |
| era | raw | chance | 1 | 0.15447 | 45 | 538 | 6 |
| era | raw | feature | 2 | 0.56108 | 47 | 539 | 6 |
| era | raw | chance | 2 | 0.15001 | 47 | 539 | 6 |
| era | raw | feature | 3 | 0.63148 | 47 | 537 | 6 |
| era | raw | chance | 3 | 0.14991 | 47 | 537 | 6 |
| era | raw | feature | 4 | 0.49095 | 47 | 538 | 6 |
| era | raw | chance | 4 | 0.16389 | 47 | 538 | 6 |
| era | relative | feature | 0 | 0.48373 | 47 | 539 | 6 |
| era | relative | chance | 0 | 0.18811 | 47 | 539 | 6 |
| era | relative | feature | 1 | 0.61445 | 45 | 538 | 6 |
| era | relative | chance | 1 | 0.15447 | 45 | 538 | 6 |
| era | relative | feature | 2 | 0.47019 | 47 | 539 | 6 |
| era | relative | chance | 2 | 0.15001 | 47 | 539 | 6 |
| era | relative | feature | 3 | 0.55914 | 47 | 537 | 6 |
| era | relative | chance | 3 | 0.14991 | 47 | 537 | 6 |
| era | relative | feature | 4 | 0.44450 | 47 | 538 | 6 |
| era | relative | chance | 4 | 0.16389 | 47 | 538 | 6 |
| era | norm | feature | 0 | 0.48615 | 47 | 539 | 6 |
| era | norm | chance | 0 | 0.18811 | 47 | 539 | 6 |
| era | norm | feature | 1 | 0.51964 | 45 | 538 | 6 |
| era | norm | chance | 1 | 0.15447 | 45 | 538 | 6 |
| era | norm | feature | 2 | 0.46190 | 47 | 539 | 6 |
| era | norm | chance | 2 | 0.15001 | 47 | 539 | 6 |
| era | norm | feature | 3 | 0.48843 | 47 | 537 | 6 |
| era | norm | chance | 3 | 0.14991 | 47 | 537 | 6 |
| era | norm | feature | 4 | 0.47337 | 47 | 538 | 6 |
| era | norm | chance | 4 | 0.16389 | 47 | 538 | 6 |

raw → 공통 패턴 제거 → 정규화 각 단계의 metadata 예측성을 비교한다. 작품별 permutation effect size/BH 보정은 [metadata_effect_sizes.csv](metadata_effect_sizes.csv), 시대/작곡가별 평균·범위는 [era_composer_feature_comparison.csv](era_composer_feature_comparison.csv)다. era는 작곡가 기반의 대략적인 label이며 작품별 작곡 연대 정답이 아니다.

다른 작품의 feature 최근접 이웃이 metadata 그룹을 얼마나 유지하는지, metadata 점수와 feature 순위의 Spearman은 [cross_work_metadata_neighborhoods.csv](cross_work_metadata_neighborhoods.csv)에 기록한다. 동일 작품 안에서는 작곡가/시대가 같기 때문에 이 metadata만으로 연주를 구별할 수 없다.

| metadata | queries | mean_top10_same_fraction | mean_pool_same_fraction | median_score_metadata_spearman |
| --- | --- | --- | --- | --- |
| artist | 500 | 0.15820 | 0.06076 | 0.05154 |
| composer | 500 | 0.47540 | 0.35438 | 0.12882 |
| era | 500 | 0.50500 | 0.36956 | 0.14632 |

위 요약은 query 수에 가중된 기술적 통계다. 같은 label의 후보가 원래 얼마나 많은지(pool 비율)와 함께 읽으며, 순위가 metadata와 다르다는 사실만으로 사용자 취향에 더 적합하다고 결론짓지 않는다.

## 동일 작품의 연주 차이와 피처 ablation

같은 연주의 겹치지 않는 beat 절반 두 집합으로 self-retrieval을 수행한다. 후보가 공통으로 관측한 beat를 사용하며 작품 단위 CI를 구한다. **연주 identity 구별력이며 취향 정확도가 아니다.**

| config | Top1 | low | high | chance | MRR | works |
| --- | --- | --- | --- | --- | --- | --- |
| all | 0.78339 | 0.75929 | 0.80628 | 0.13137 | 0.87046 | 201 |
| without_articulation | 0.78315 | 0.75954 | 0.80809 | 0.13137 | 0.87152 | 201 |
| without_dynamics | 0.69483 | 0.66697 | 0.72222 | 0.13137 | 0.81080 | 201 |
| without_pedaling | 0.72856 | 0.70241 | 0.75453 | 0.13137 | 0.83420 | 201 |
| without_rubato | 0.80426 | 0.78130 | 0.82639 | 0.13137 | 0.88461 | 201 |
| without_tempo | 0.73216 | 0.70545 | 0.75620 | 0.13137 | 0.83782 | 201 |

피처 제거 전후의 동일 작품 paired 차이:

| comparison | mean | low | high | works |
| --- | --- | --- | --- | --- |
| without_articulation_minus_all | -0.00024 | -0.01406 | 0.01463 | 201 |
| without_dynamics_minus_all | -0.08857 | -0.10205 | -0.07509 | 201 |
| without_pedaling_minus_all | -0.05484 | -0.06890 | -0.04233 | 201 |
| without_rubato_minus_all | 0.02087 | 0.00898 | 0.03258 | 201 |
| without_tempo_minus_all | -0.05123 | -0.06469 | -0.03875 | 201 |

시간상 전반부→후반부 retrieval: `{"hit1": {"mean": 0.4973720852400722, "low": 0.464113786994999, "high": 0.528396820995176, "works": 201}, "mrr": {"mean": 0.656487786360103, "low": 0.6298166786113917, "high": 0.6804224895278074, "works": 201}, "chance_hit1": {"mean": 0.1313699386468524, "low": 0.11850257127267101, "high": 0.14366193802630914, "works": 201}}`.

## 사용자 취향 추천을 위한 대리 검증

대상 작품을 포함하지 않는 3~5개 다른 작품의 동일 연주자 연주로 profile을 구성하고 대상 작품에서 해당 연주자의 연주를 찾는다. feature와 composer/era metadata, artist metadata를 비교한다. **동일 artist 선호를 가정한 proxy이며 실제 사용자 좋아요 정확도나 미관측 연주자 일반화가 아니다.**

| model | Top1 | low | high | MRR | works |
| --- | --- | --- | --- | --- | --- |
| artist_metadata | 0.65636 | 0.62769 | 0.68513 | 0.80226 | 201 |
| composer_era_metadata | 0.13137 | 0.11850 | 0.14366 | 0.32341 | 201 |
| feature | 0.17444 | 0.15232 | 0.19717 | 0.37189 | 201 |

동일 query·동일 작품에서 metadata 기준과의 paired 비교:

| comparison | mean | low | high | works |
| --- | --- | --- | --- | --- |
| feature_minus_composer_era | 0.04307 | 0.02663 | 0.06093 | 201 |
| feature_minus_artist | -0.48193 | -0.50753 | -0.45521 | 201 |

profile beat dropout에 대한 순위 안정성: `{"all": {"rank_spearman": {"mean": 0.98950116342167, "low": 0.9868621925604152, "high": 0.9917043230012088, "works": 201}, "top1_unchanged": {"mean": 0.9281923714759536, "low": 0.9074585406301825, "high": 0.9480099502487562, "works": 201}}, "without_articulation": {"rank_spearman": {"mean": 0.9909444959504327, "low": 0.9866494513533658, "high": 0.9937392956247009, "works": 201}, "top1_unchanged": {"mean": 0.9307628524046434, "low": 0.9091894693200664, "high": 0.95, "works": 201}}, "without_dynamics": {"rank_spearman": {"mean": 0.9887699361882789, "low": 0.9865489378279656, "high": 0.9906010470288474, "works": 201}, "top1_unchanged": {"mean": 0.9246268656716419, "low": 0.9044651741293532, "high": 0.9432898009950249, "works": 201}}, "without_pedaling": {"rank_spearman": {"mean": 0.9875357819463327, "low": 0.9851512693135656, "high": 0.9896236480353628, "works": 201}, "top1_unchanged": {"mean": 0.932587064676617, "low": 0.9149191542288556, "high": 0.9490049751243781, "works": 201}}, "without_rubato": {"rank_spearman": {"mean": 0.9896059425896196, "low": 0.9868057460653441, "high": 0.9919320275702197, "works": 201}, "top1_unchanged": {"mean": 0.9353233830845771, "low": 0.917412935323383, "high": 0.9519900497512438, "works": 201}}, "without_tempo": {"rank_spearman": {"mean": 0.9858824570520544, "low": 0.9826219611065012, "high": 0.9888690021821046, "works": 201}, "top1_unchanged": {"mean": 0.9378938640132669, "low": 0.9196475953565507, "high": 0.9555555555555555, "works": 201}}}`.

페달의 세 하위 채널은 하나의 피처군으로 동일 가중치를 갖는다. 후보 결측이 거리를 유리하게 만들지 않도록 query와 후보의 공통 완전한 피처군 조건을 유지한다.

## 이전 정렬과의 비교

| alignment | normalized | works | invalid_issues | identity_Top1 | preference_proxy_Top1 | outlier_top1_unchanged |
| --- | --- | --- | --- | --- | --- | --- |
| heuristic | 2366 | 225 | 0 | 0.60480 | 0.17455 | 0.64350 |
| dual_dtw | 2692 | 233 | 0 | 0.78339 | 0.17444 | 0.83046 |

정렬 gate 통과 작품/연주가 달라지므로 두 전체 cohort의 추천 지표 차이를 정렬기의 인과적 개선으로 해석하지 않는다. 공통 2284개 연주의 피처 요약 차이는 [previous_alignment_feature_differences.csv](previous_alignment_feature_differences.csv)에 있다. 이 차이에는 각 정렬의 beat 구간 및 normalization pool 변화도 포함된다.

## 활용 판단

동일 작품에서 연주 identity Top1은 **78.339%**, 후보 수에 따른 무작위 기준은 **13.137%**다. 이는 해석 차이에 대한 구별력의 기술적 지표다.

동일 연주자 선호라는 대리 과제에서는 작곡가·시대 metadata보다 추가 구별력이 관측됐다. Feature − composer/era Top1 차이는 **0.04307 (95% CI 0.02663 ~ 0.06093; 201개 작품)**다. 사용자 취향을 동일 연주자 선호로 대체한 결과이므로 실제 사용자 좋아요의 정확도로 해석하지 않는다. artist metadata와의 별도 비교도 위 표에 제공한다.

극단값 cell을 제외했을 때 Top1이 바뀐 비율은 **16.954%**다. 따라서 수치 무결성 통과와 추천 순위의 신뢰성을 구분해야 한다. 검수 queue를 확인하고, 검수 전후 순위 변화와 피처 ablation 결과를 바탕으로 사용할 피처를 정해야 한다.

현재 결과는 실제 사용자 평가에 투입할 후보 피처와 검수 대상을 결정하는 근거다. 자동 전사 정보 손실·자동 정렬·극단값 민감도·실제 사용자 label 부재 때문에 최종 취향 추천 정확도를 확정하지 않는다. 다음 검증은 사용자별 선호 연주로 profile을 만든 뒤, profile에 넣지 않은 동일 작품의 연주에 대한 쌍별 선호/순위를 예측하고 metadata 기준과 비교하는 것이다. 평가 대상 사용자·작품을 분리하고 학습/평가 경계를 넘는 공통 패턴 추정을 막아야 한다.

## 재현

저장소 루트에서:

```powershell
.venv/Scripts/python.exe classicfy-ai/scripts/extend_atepp.py --aligner selected --output classicfy-ai/analysis/atepp_research --workers 2 --resume
.venv/Scripts/python.exe classicfy-ai/scripts/validate_atepp_recommendation.py --output classicfy-ai/analysis/atepp_research
.venv/Scripts/python.exe classicfy-ai/scripts/write_research_alignment_reports.py
```

전체 순차 실행은 `run_research_alignment_pipeline.py`로 제공한다. `selection.json`에서 정렬기를 읽으므로 선택 결과가 바뀌면 후속 단계에도 반영된다. 기존 피처 수식과 기존 검증 함수를 재사용하며 동일한 검증 조건을 유지한다.
