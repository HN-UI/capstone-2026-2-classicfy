# `midi_loader.py` 구현 계획

## 목적과 범위

`classicfy-ai/src/preprocessing/midi_loader.py`는 score MIDI와 performance MIDI를 같은 방식으로 읽어, 후속 feature extractor가 사용할 원본 note·sustain pedal 이벤트를 반환한다. 구현 근거는 루트의 `PROJECT_CONTEXT.md`, 특히 20~21절과 24~27절이다.

ASAP의 `metadata.csv`, `asap_annotations.json`, score–performance 짝과 beat alignment는 향후 `asap_loader.py`가 담당한다. 
이 모듈에서는 tempo, rubato, dynamics, articulation, pedaling 점수를 계산하지 않고 score note와 performance note를 순서대로 매칭하지 않는다.

## 공개 인터페이스

단일 진입점 `load_midi(path: str | Path) -> MidiData`를 제공한다. 호출자가 MIDI 파일 경로를 전달하며, 이 모듈은 데이터셋 위치나 현재 작업 디렉터리에 대한 기본값을 갖지 않는다. 동일한 함수로 score와 performance를 읽는다.

`dataclass` 세 개를 같은 파일에 정의한다.

| 타입 | 필드 | 의미 |
| --- | --- | --- |
| `NoteEvent` | `pitch: int`, `start: float`, `end: float`, `velocity: int`, `instrument_idx: int`, `program: int` | 원본 note 값과 원래 instrument 위치·MIDI program |
| `PedalEvent` | `time: float`, `value: int`, `instrument_idx: int` | CC64 시각·원본 0~127 값·원래 instrument 위치 |
| `MidiData` | `notes: list[NoteEvent]`, `pedals: list[PedalEvent]`, `duration: float` | 정렬된 이벤트와 `PrettyMIDI.get_end_time()`의 초 단위 길이 |

`NoteEvent.duration`은 필요할 때 `end - start`를 반환하는 읽기 전용 속성으로 둔다. CC64 value는 임계값 적용이나 정규화 없이 보존한다. `instrument_idx`는 `midi.instruments`의 원래 인덱스이므로, 여러 score part를 합친 뒤에도 출처를 추적할 수 있다. `program`이나 instrument 개수로 오른손·왼손을 추정하지 않는다.

## 구현 순서

1. `path`를 `Path`로 변환한다. 존재하지 않으면 경로가 포함된 `FileNotFoundError`, 디렉터리 등 일반 파일이 아니면 `ValueError`를 발생시킨다.
2. `pretty_midi.PrettyMIDI(str(path))`로 파싱한다. 파싱 실패에는 원래 예외를 연결해 경로가 포함된 `ValueError`를 발생시킨다. `pretty_midi`의 warning은 숨기지 않는다.
3. `midi.instruments`를 원래 순서로 순회한다. drum instrument의 note와 CC는 제외하고, 나머지 모든 instrument의 note를 `NoteEvent`로 복사한다. score가 여러 instrument로 나뉘어 있어도 하나만 선택하지 않는다.
4. 같은 instrument의 `control_changes` 중 `number == 64`인 이벤트만 `PedalEvent`로 복사한다. 중간 pedal value와 같은 시각의 여러 이벤트도 버리지 않는다.
5. note는 `(start, pitch, instrument_idx)`, pedal은 `(time, instrument_idx)` 순서로 안정적으로 정렬한다. `midi.get_end_time()`을 duration으로 넣어 `MidiData`를 반환한다. note나 pedal이 없는 유효한 MIDI는 빈 목록으로 반환한다.

시간 단위는 `pretty_midi`가 제공하는 초를 그대로 사용한다. score와 performance의 절대 시간축이 같다고 가정하지 않는다. 일부 ASAP MIDI의 비표준 tempo 관련 warning이 알려져 있으므로 내부 tempo 이벤트로 feature를 계산하지 않는다.

## 검증과 완료 기준

- 작은 임시 MIDI를 생성하는 단위 테스트에서 다중 non-drum instrument의 note가 모두 포함되고 drum note가 제외되는지 확인한다. 각 note의 pitch·start·end·velocity·instrument index·program 및 정렬 순서를 비교한다.
- CC64와 다른 CC를 함께 넣어 CC64만 수집되는지 확인한다. 0, 127뿐 아니라 중간 value가 원형대로 유지되고, 여러 instrument의 pedal 출처와 시간 순서가 보존되는지 확인한다.
- note가 없는 유효한 MIDI, CC64가 없는 MIDI, 존재하지 않는 경로, 디렉터리 경로, 손상된 MIDI를 각각 확인한다. 유효한 빈 입력은 빈 목록을 반환하고 잘못된 경로·파일은 명확한 오류를 낸다.
- 로컬 ASAP의 `Bach/Fugue/bwv_846/midi_score.mid`와 `Bach/Fugue/bwv_846/Shi05M.mid`를 읽는 smoke test를 수행한다. 두 파일의 note가 비어 있지 않고 `0 <= pitch, velocity <= 127`, `end >= start`, `0 <= pedal value <= 127`인지 확인한다. 참고로 `PROJECT_CONTEXT.md`에서 관찰한 note 수는 score 755개, performance 754개다.
- ASAP 전체 호환성 확인은 별도 통합 검증으로 수행한다. 로컬 `metadata.csv`의 1,067행에서 `midi_score`와 `midi_performance` 경로를 읽고, 중복 score 경로는 한 번만 파싱한다. 각 MIDI에 `load_midi`를 적용해 파싱 성공 여부, note·pedal 개수, 값 범위와 시간 순서를 검사하고 실패 경로와 warning을 요약한다. 작품별로 note 수나 pedal 이벤트 수가 같아야 한다고 가정하지 않는다. 이 전수 검증은 데이터셋이 필요한 수동 또는 선택적 테스트로 두고 일반 단위 테스트의 필수 조건으로 만들지 않는다.
- 테스트에는 개인의 절대 경로를 저장하지 않는다. 데이터셋 경로를 인자로 받거나 저장소 상위의 `datasets/ASAP`을 테스트 실행 시에만 찾아, 데이터셋이 없는 환경에서는 smoke test를 건너뛴다.

현재 `load_midi`와 세 데이터 모델을 구현했으며, 임시 MIDI와 위 ASAP score·performance 파일로 기본 동작을 확인했다. ASAP 전수 검증은 아직 수행하지 않았다. 후속 `asap_loader.py` 및 feature 모듈에서 공개 함수와 데이터 모델을 재사용할 수 있는지는 해당 모듈 구현 시 확인한다.
