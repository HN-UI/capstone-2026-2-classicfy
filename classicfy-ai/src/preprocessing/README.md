# Classicfy AI preprocessing

`preprocessing`은 MIDI 파일의 원본 이벤트와 ASAP 데이터셋의 악보-연주 관계를 읽는다. 연주 해석 feature와 score-performance alignment를 새로 계산하지 않는다.

## 공개 인터페이스

- `load_midi(path: str | Path) -> MidiData`: MIDI 파일 하나에서 비드럼 파트의 음표와 원본 CC64 이벤트를 읽는다. 시간 단위는 초다.
- `ASAPLoader(root: str | Path)`: 전달받은 ASAP 루트의 `metadata.csv`와 `asap_annotations.json`을 읽는다.
- `get_sample(performance_key: str) -> AsapSample`: `metadata.csv`의 상대 연주 MIDI 경로를 키로 받아 악보·연주 파일 경로와 annotation을 반환한다.
- `iter_samples(aligned_only: bool = False)`: 메타데이터 순서로 샘플을 순회한다. `aligned_only=True`면 정렬되지 않은 샘플을 제외한다.

`AsapSample`은 `score_path`, `performance_path`, `aligned`, 양쪽의 beat·downbeat 목록과 원본 박자표 정보를 제공한다. `aligned=False`인 샘플은 양쪽 beat 목록의 길이가 다를 수 있으므로 인덱스로 짝짓지 않는다. `MidiData`는 `notes`, `pedals`, `duration`을 제공한다.

공개 이름은 `preprocessing` 패키지에서 가져온다. `AsapSample`, `MidiData`, `NoteEvent`, `PedalEvent`도 같은 경로로 사용할 수 있다.

```python
from preprocessing import ASAPLoader, load_midi

loader = ASAPLoader("/path/to/ASAP")
sample = loader.get_sample("Bach/Fugue/bwv_846/Shi05M.mid")
score = load_midi(sample.score_path)
performance = load_midi(sample.performance_path)
```

존재하지 않는 파일은 `FileNotFoundError`, 잘못된 파일·데이터 형식은 `ValueError`, 알 수 없는 연주 키나 누락된 annotation은 `KeyError`로 처리한다. 경로는 호출자가 전달하며 저장소나 현재 작업 디렉터리의 고정 위치에 의존하지 않는다.

## 테스트

`classicfy-ai` 디렉터리에서 실행한다.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

실제 ASAP 샘플 테스트는 저장소 상위의 `datasets/ASAP`을 찾고, 데이터셋이 없으면 건너뛴다. 다른 위치에 있다면 `ASAP_ROOT` 환경 변수를 지정한다. 전체 데이터셋 순회 검증은 별도로 수행한다.
