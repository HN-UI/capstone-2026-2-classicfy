# 연주 임베딩 청취 평가 도구

임베딩에서 나타난 연주 차이가 실제로도 들리는지 팀원 세 명이 독립적으로 확인한다.
실제 녹음을 X/Y로 듣고 응답을 저장하는 정적 웹 화면이며, 로그인이나 외부 응답 서버는 없다.

## 준비된 자료로 평가하기

기본 자료는 저장소 밖 `Classicfy/datasets/listening_evaluation/pack/`에 생성된다.
`pack/index.html`을 브라우저에서 열거나, 저장소 루트에서 다음 명령을 실행한다.

```bash
.venv/bin/python classicfy-ai/tools/listening_evaluation/serve.py
```

화면 주소는 `http://127.0.0.1:8765/`이다. 로컬 서버는 오디오 구간 이동을 위한 HTTP Range를 지원한다.
서버 종료는 `Ctrl+C`다. 참가자 화면은 HTML·JS·CSS와 WAV만 사용한다.

1. `datasets/listening_evaluation/listening_participant_pack.zip`을 팀원에게 전달한다. 압축을 풀고 `index.html`을 연다.
2. 서로 다른 P1/P2/P3 번호를 정한다. 같은 헤드폰·전체 볼륨·1배 속도를 유지하고, 답은 상의하지 않는다.
3. 11문항을 평가한 뒤 응답 JSON을 저장해서 진행자에게 전달한다. CSV는 사람이 확인하기 위한 사본이다.

약 25–35분이 예상되며, 중간 JSON 저장과 브라우저 자동 저장으로 이어서 평가할 수 있다.
브라우저 데이터를 지우거나 시크릿 모드를 종료하면 임시 응답이 사라질 수 있으므로 JSON을 저장한다.
다른 컴퓨터에서는 첫 화면의 ‘다른 컴퓨터에서 이어하기’로 JSON을 불러온다.

참가자에게 **`analysis/_listening_evaluation/`은 보내지 않는다.** 해당 폴더는 실제 연주 ID,
선정 기준과 임베딩 거리를 담은 진행자 자료다. 공개 pack에는 이 정보가 포함되지 않는다.

## 무엇을 평가하는가

| 구성 | 확인하는 질문 |
|---|---|
| 5개 feature별 같은 작품 비교 | 해당 특징의 차이가 실제로도 들리는가? |
| 같은 기준 연주의 가까운·먼 후보 | 더 먼 후보가 더 다르게 들리는가? |
| 동일 음원·반복 비교 | 같은 음원이 다르다고 들리는지, 반복 판단이 얼마나 달라지는지 |
| 다른 작품의 A와 후보 X/Y 두 사례 | 특징상 가까운 후보가 실제로도 더 비슷한 경향으로 들리는가? |

각 연주는 작품의 25%·60% 악보 위치에서 시작하는 두 구간을 사용한다. 끝 위치는 작품의
원래 분석 cohort 전체 연주에서 약 16초가 되는 악보 위치로 정한다. 따라서 X/Y는 같은
악보 구간이지만 연주 속도에 따라 길이가 다르다. 특징 차이가 잘 보이는 구간을 찾아 이동하지 않는다.

같은 작품은 1–5 차이 척도와 ‘판단 어려움’, 다른 작품은 X/Y·‘둘이 비슷함’·‘판단 어려움’을 사용한다.
모든 문항에서 다섯 feature와 전체 판단, 확신, 메모를 남긴다. 대상 feature와 거리는 평가 전에 숨긴다.
참가자별 문항 순서를 바꾸고 X/Y를 교대한다. 반복 비교는 X/Y를 뒤집고 세 문항 이상 떨어뜨린다.

## 세 명의 응답 분석

저장소 루트에서 실제로 받은 세 JSON 경로를 지정한다.

```bash
.venv/bin/python classicfy-ai/scripts/analyze_listening_study.py \
  /path/to/P1.json /path/to/P2.json /path/to/P3.json
```

기본 결과 위치는 `Classicfy/datasets/listening_evaluation/results/`다. 참가자 중복, 다른
study ID, 바뀐 문항 순서, 누락된 평가를 검증한다. 세 명의 완료 응답이 기본 조건이다.
`--allow-partial`을 명시하면 일부 참가자·완료 문항만 분석하고 보고서에 부분 분석임을 표시한다.

| 결과 | 설명 |
|---|---|
| `responses.csv` | 완료 문항별 원 응답 |
| `pair_summary.csv` | 차이 척도의 중앙값·수치 응답 수·판단 어려움 수 |
| `cross_choices.csv` | X/Y를 실제 B/C로 복원한 선택 수·동점·판단 어려움·확정 선택 분모 |
| `consistency.csv` | 동일 음원 점수·반복 비교의 점수 차이 |
| `results.json`, `README.md` | 결과와 해석 범위 |
| `01_feature_contrasts.png` | 각 feature 차이에 대한 개인 점·중앙값 |
| `02_near_far.png` | 같은 기준 연주의 가까운·먼 후보에 대한 청취 차이 |
| `03_distance_vs_hearing.png` | 실제 들은 발췌 구간 거리와 청취 차이 |
| `04_cross_work_choices.png` | 다른 작품에서 더 비슷하다고 고른 후보 |

그림은 파일당 하나다. 세 사람의 개별 점을 보존하고, 동점·판단 어려움을 오답으로 바꾸지 않는다.
전체 곡과 발췌 구간의 임베딩 최근접을 **각각** 비교한다. 동일·반복 확인 문항으로 참가자를 자동 제외하지 않는다.
현재는 도구와 평가 자료만 준비됐으며, 실제 사람의 응답이나 청취 결과는 아직 없다.

## 자료 재생성

이미 분석한 ASAP/nASAP와 `feature_normalization_raw.npz` 캐시를 사용한다.
기존 570개 연주·14차원 임베딩과 값이 같은지 검증하며 feature 추출이나 정규화 로직은 변경하지 않는다.
필요한 Python 패키지는 기존 `requirements.txt`에 있고 추가 의존성은 없다.

```bash
# 선정 목록만 생성. 음원 다운로드 없음.
.venv/bin/python classicfy-ai/scripts/prepare_listening_study.py --plan-only

# 직접 받은 원본 MAESTRO WAV 또는 잘린 ASAP WAV를 사용.
.venv/bin/python classicfy-ai/scripts/prepare_listening_study.py \
  --audio-root /path/to/audio

# 선택한 음원의 발췌에 필요한 앞부분만 명시적으로 다운로드.
.venv/bin/python classicfy-ai/scripts/prepare_listening_study.py --download
```

기본 실행은 자동 다운로드하지 않는다. `--download`도 MAESTRO 전체 ZIP을 받지 않는다.
공식 ZIP의 목차와 선택한 WAV의 필요한 압축 prefix를 HTTP Range로 가져온다.
다시 준비할 때는 SHA256이 일치하는 발췌를 재사용한다. 원본 WAV는 저장하지 않고,
이미 마련된 백엔드 샘플의 원본은 재사용한다.

`--asap-root`, `--nasap-root`, `--cache`, `--out`, `--pack`으로 경로를 바꿀 수 있다.
진행자 `organizer.json`과 공개 pack의 study ID가 같아야 분석할 수 있다. 문항·음원·순서가
바뀌면 study ID도 바뀌므로 이미 평가를 시작한 pack은 재생성하지 않고 보관한다.

음원은 [MAESTRO v2.0.0](https://magenta.withgoogle.com/datasets/maestro)의 실제 피아노 녹음이며
[ASAP](https://github.com/fosfrancesco/asap-dataset) 메타데이터로 MIDI와 연결한다.
Google LLC / International Piano-e-Competition 출처를 pack의 `LICENSE.txt`에 포함한다.
CC BY-NC-SA 4.0 조건으로 비상업적 사용·출처 표기·동일 조건 공유를 유지한다.
발췌 외 gain·속도·샘플레이트를 바꾸지 않는다. 다운로드 prefix는 전체 원본 ZIP entry CRC를
검증할 수 없으므로 HTTPS·고정 ETag·발췌 SHA256 검증으로 기록하고, 전체 CRC 검증과 구분한다.

## 결과의 한계

이 평가는 선별된 11문항·3인의 탐색적 검증이다. 전곡·모든 작품에 대한 정확도나 추천 성능을
증명하지 않는다. feature별 쌍은 그 feature의 전곡 거리가 작품 내 상위 25%에 들고, 나머지 네
feature 거리가 중앙값 이하인 후보에서 결정한 사례다. feature마다 다른 작품을 사용한다.
같은 기준 연주의 가까운·먼 후보와 기존 다른 작품 검색 사례도 포함한다.

정규화된 MIDI 특징은 작품 안에서의 상대적 성향이지만 청자는 절대 빠르기·음량을 듣는다.
특히 다른 작품 비교에는 악보 차이와 녹음 음색·gain·잔향이 섞인다. Dynamics·Pedaling의
차이를 오직 연주 해석으로 단정하지 않는다. 전곡과 발췌에서 순서가 바뀌는 사례도 결과에 남긴다.
어떤 feature가 모든 검색에 필수인지, 추천 만족도가 높아지는지는 별도 실험이 필요하다.

## 검증

```bash
cd classicfy-ai
PYTHONPATH=src ../.venv/bin/python -m unittest tests.unit.tools.test_listening_study -v
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -v
```

테스트는 PCM 구간·원본/잘린 음원 offset·ZIP64·캐시 검증·비공개 순서·응답 입력 검증·
동점/불확실 분모·HTTP Range와 pack 밖 파일 차단을 확인한다.

현재 환경에서 전체 163개 테스트(skip 0), Chrome의 실제 재생·seek·모바일 배치·JSON/CSV 다운로드·
이어하기·불러오기와 `file://` 직접 실행을 확인했다. 분석 생성 검증은 자동 응답 fixture로만
수행했으며 그 결과는 저장소에 사람의 청취 결과로 포함하지 않는다.
