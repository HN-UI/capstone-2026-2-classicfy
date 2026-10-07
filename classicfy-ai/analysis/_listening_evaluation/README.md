# 청취 평가 준비 기록 — 진행자 전용

**평가 전에 참가자에게 보여주지 않는다.** 참가자에게는 음원과 X/Y 화면만 담은 ZIP을 전달한다.
[실행·응답 분석 안내](../../tools/listening_evaluation/README.md)에서 사용 방법을 확인한다.

- Study ID: `370841fbeedfb5988cf6` / seed: `20261005`
- 11문항, 실제 연주 18개, WAV 36개.
- 발췌 길이: 15.06–18.77초.
- 음원은 저장소 밖 `Classicfy/datasets/listening_evaluation/pack/`, 공유 ZIP은 그 상위에 있다.
- 사람의 응답은 아직 수집하지 않았다. 이 폴더는 선정/준비 자료이며 청취 성공률을 담지 않는다.

## 합리적인 기준을 고정한 방법

기존 ASAP/nASAP 분석의 570개 연주·14차원 임베딩을 그대로 재사용한다. 원래 cohort는 같은
작품·동일 beat grid·최소 5연주·32개 공통 유효 beat·50% coverage 조건이다. 원본 캐시 SHA256은
`79ab737a08335c7fd7e9a912159de65b7c26795e1ef22e45114e5a8c6eb8ad2e`다. 기존 embeddings.csv와 키 순서와 값까지 일치하는지 확인했다.

공식 MAESTRO WAV 매핑이 있고 같은 cohort에서 음원이 3개 이상인 후보만 비교한다.
feature별로 대상 feature의 거리가 작품 내 쌍 거리의 Q75 이상이고 나머지 네 feature 거리가
중앙값 이하인 쌍을 남긴다. `abs(target/Q75 - 1) + other/max(median, 1e-12)`가 가장 작은
후보를 고르며 동점은 키 순서로 결정하고, feature마다 서로 다른 작품을 사용한다.
극단적 최대 거리만 고르지 않고 기준과 가까운 적격 사례를 선택한다. 후보 전체와 제외되지 않은
선정 비용·분위수는 `selection_pool.csv`에 저장한다. 최종 선택은 청취 전에 고정했다.

S06/S07은 같은 기준 연주의 최근접과 작품 내 Q75 이상 첫 후보다. S08은 동일 음원,
S09는 S01의 X/Y를 뒤집은 반복이다. C01/C02는 기존 다른 작품 검색의 anchor/typical 사례를
다시 고르지 않고 사용한다. 전곡 기준 B가 가깝지만, 청취에서 B를 정답으로 취급하지 않는다.

## 실제 연주와 거리

아래 거리는 다섯 feature block을 동일 가중한 전체 거리다. pair는 첫째–둘째, cross는 A–B / A–C 순이다.
공개 X/Y 위치는 참가자별로 다르며 `organizer.json`의 schedules로 복원한다.

| 문항 | 목적 | 실제 연주 ID | 전곡 거리 | 발췌 거리 |
|---|---|---|---:|---:|
| S01 | contrast · Tempo | 첫 연주/둘째 연주: `Bach/Prelude/bwv_860/YoungS01M.mid`<br>`Bach/Prelude/bwv_860/ZhangH04M.mid` | 0.678 | 0.773 |
| S02 | contrast · Rubato | 첫 연주/둘째 연주: `Bach/Prelude/bwv_848/Lin04M.mid`<br>`Bach/Prelude/bwv_848/Lou01M.mid` | 0.317 | 0.317 |
| S03 | contrast · Dynamics | 첫 연주/둘째 연주: `Chopin/Etudes_op_25/11/KyykhynenT10M.mid`<br>`Chopin/Etudes_op_25/11/MiyashitaM03M.mid` | 0.478 | 0.668 |
| S04 | contrast · Articulation | 첫 연주/둘째 연주: `Bach/Prelude/bwv_885/Huang04M.mid`<br>`Bach/Prelude/bwv_885/SINKEV06.mid` | 0.855 | 1.152 |
| S05 | contrast · Pedaling | 첫 연주/둘째 연주: `Chopin/Scherzos/31/TongB04M.mid`<br>`Chopin/Scherzos/31/ZhangW04M.mid` | 0.326 | 0.532 |
| S06 | near · all | 첫 연주/둘째 연주: `Chopin/Scherzos/31/TongB04M.mid`<br>`Chopin/Scherzos/31/TuanS05M.mid` | 0.195 | 0.563 |
| S07 | far · all | 첫 연주/둘째 연주: `Chopin/Scherzos/31/TongB04M.mid`<br>`Chopin/Scherzos/31/LeeN04M.mid` | 0.484 | 0.618 |
| S08 | identical · check | 첫 연주/둘째 연주: `Bach/Prelude/bwv_848/Lin04M.mid`<br>`Bach/Prelude/bwv_848/Lin04M.mid` | 0.000 | 0.000 |
| S09 | repeat · Tempo | 첫 연주/둘째 연주: `Bach/Prelude/bwv_860/YoungS01M.mid`<br>`Bach/Prelude/bwv_860/ZhangH04M.mid` | 0.678 | 0.773 |
| C01 | cross · all | A/B/C: `Bach/Fugue/bwv_848/Lou01M.mid`<br>`Bach/Prelude/bwv_857/ToA01M.mid`<br>`Bach/Prelude/bwv_857/WangA01M.mid` | 0.544 / 0.858 | 1.258 / 1.036 |
| C02 | cross · all | A/B/C: `Bach/Fugue/bwv_860/Nikiforov05M.mid`<br>`Schubert/Impromptu_op.90_D.899/3/Ko08M.mid`<br>`Schubert/Impromptu_op.90_D.899/3/Hou06M.mid` | 0.490 / 0.811 | 0.811 / 1.172 |

## 발췌를 해석할 때

작품의 25%·60% score interval에서 시작한다. 끝 위치는 원래 분석 cohort 전체 연주의
중앙 재생 시간이 약 16초가 되는 score beat로 결정한다. feature 차이를 탐색해 구간을 이동하지 않는다.
X/Y의 score 구간은 같고 실제 재생 시간은 다를 수 있다. MAESTRO 원본에는 ASAP 시작 offset을
더하며, 이미 잘린 ASAP WAV에는 이 offset을 다시 더하지 않는다.

공통 패턴과 scale은 기존 전체 cohort에서 계산한 것을 유지하고, 두 구간의 공통 유효 beat를
이어 붙여 14차원 요약을 다시 계산한다. 해당 두 구간은 청취용 발췌이며 전곡 전체를 대표한다고
보장하지 않는다. **C01은 발췌에서 A–C가 더 가깝다.** S06/S07도 발췌에서는 거리 차이가
전곡보다 작아진다. 청취 결과를 전곡 최근접에만 맞춰 해석하지 않고 두 기준을 따로 비교한다.

## 파일과 재현

| 파일 | 내용 |
|---|---|
| `selection_pool.csv` | 모든 적격 쌍과 선정 비용·원래 분위수·selected 여부 |
| `selection.json` | 고정 문항·private 순서·전곡/발췌 벡터·source provenance |
| `clip_manifest.csv` | 실제 연주 ID·score beat·ASAP/원본 오디오 시간·opaque 파일명 |
| `audio_sources.csv` | 공식 WAV entry와 직접 받은 음원 경로 |
| `audio_quality.csv` | 정확한 PCM frame·길이·샘플레이트·peak·RMS·SHA256 |
| `organizer.json` | study ID·공개 화면 자료·비공개 키·기준·scale·무결성 기록 |

재현 명령은 저장소 루트에서 `.venv/bin/python classicfy-ai/scripts/prepare_listening_study.py --download`다.
기본 실행은 다운로드하지 않으며 `--audio-root`로 로컬 음원을 받을 수 있다. 전체 ZIP을 받지 않고
선택 음원의 prefix만 가져온다. 같은 WAV에서 추출한 두 ASAP 작품도 별도 시간 구간으로 처리한다.
발췌는 native 16-bit PCM stereo이며 gain·속도·샘플레이트를 바꾸지 않는다.
원본 prefix는 전체 entry CRC를 검증할 수 없으므로 고정 ETag/HTTPS/발췌 SHA256로 검증 범위를 명시한다.
MAESTRO Google LLC / International Piano-e-Competition 출처와 CC BY-NC-SA 4.0 조건은 공개 pack의 LICENSE에 있다.

3인·소수 선정 사례이므로 전곡/전체 작품의 정확도·선호 추천 성능·모든 feature의 필수성을 주장하지 않는다.
MIDI의 작품 내 상대적 특징과 귀로 들리는 절대적 빠르기/음량은 다를 수 있고 녹음 gain·음색·잔향도 섞인다.
동일/반복 문항은 진단 기록이며 응답자를 자동 제외하는 필터가 아니다.
