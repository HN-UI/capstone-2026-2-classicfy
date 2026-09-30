"""(n)ASAP의 match 파일에서 note 단위 악보-연주 정렬을 읽는다.

match 파일은 줄마다 Prolog 사실 하나를 담는다. 이 모듈은 아래 네 종류만 해석하고
나머지(``sustain``, ``soft``, ``meta``, ``scoreprop`` 등)는 무시한다.

- ``info(midiClockUnits,U).``, ``info(midiClockRate,R).``: tick을 초로 바꾸는 값
- ``snote(...)-note(...).``: 악보 음과 연주 음이 짝지어진 경우
- ``snote(...)-deletion.``: 연주에서 빠진 악보 음
- ``insertion-note(...).``: 악보에 없는 연주 음

(n)ASAP에는 ``note`` 형식이 다른 두 버전이 섞여 있다.

- 1.0.0: ``note(Id,MidiPitch,Onset,Offset,Velocity,Channel,Track)``
- 5.0: ``note(Id,[Step,Alter],Octave,Onset,Offset,AdjOffset,Velocity)``

두 버전 모두 ``Offset``은 건반을 뗀 시각(MIDI note-off)이다. 5.0의 ``AdjOffset``은
페달로 늘어난 소리의 끝이라 쓰지 않는다.
"""

from dataclasses import dataclass
from pathlib import Path

_STEP_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ALTERS = {"n": 0, "#": 1, "b": -1, "x": 2, "##": 2, "bb": -2}
_MICROSECONDS_PER_SECOND = 1_000_000


@dataclass(frozen=True)
class ScoreNote:
    """match 파일의 악보 음 하나. 시각은 악보 beat 단위다."""

    note_id: str
    pitch: int
    onset_beats: float
    offset_beats: float
    attributes: tuple[str, ...]

    @property
    def duration_beats(self) -> float:
        return self.offset_beats - self.onset_beats

    @property
    def is_grace(self) -> bool:
        return "grace" in self.attributes or self.duration_beats <= 0


@dataclass(frozen=True)
class PerformedNote:
    """match 파일의 연주 음 하나. 시각은 연주 MIDI 기준 초 단위다."""

    note_id: str
    pitch: int
    onset: float
    offset: float
    velocity: int

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass(frozen=True)
class NoteAlignment:
    """한 연주의 note 단위 정렬 결과."""

    matches: tuple[tuple[ScoreNote, PerformedNote], ...]
    deletions: tuple[ScoreNote, ...]
    insertions: tuple[PerformedNote, ...]


def _split_arguments(text: str) -> list[str]:
    """괄호 깊이 0에 있는 쉼표로만 인자를 나눈다."""
    arguments = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            arguments.append(text[start:index])
            start = index + 1
    arguments.append(text[start:])
    return [argument.strip() for argument in arguments]


def _call_arguments(text: str, name: str) -> list[str]:
    """``name(...)`` 형태의 문자열에서 인자 목록을 꺼낸다."""
    if not text.startswith(f"{name}(") or not text.endswith(")"):
        raise ValueError(f"Expected {name}(...): {text}")
    return _split_arguments(text[len(name) + 1 : -1])


def _list_items(text: str) -> list[str]:
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"Expected a list: {text}")
    inner = text[1:-1].strip()
    return _split_arguments(inner) if inner else []


def _spelled_pitch(spelling: str, octave: str) -> int:
    step, alter = _list_items(spelling)
    if step not in _STEP_PITCH_CLASSES or alter not in _ALTERS:
        raise ValueError(f"Unknown pitch spelling: {spelling}")
    return 12 * (int(octave) + 1) + _STEP_PITCH_CLASSES[step] + _ALTERS[alter]


def _parse_score_note(text: str) -> ScoreNote:
    # snote(Id,[Step,Alter],Octave,Bar:Beat,Offset,Duration,OnsetInBeats,OffsetInBeats,[Attributes])
    arguments = _call_arguments(text, "snote")
    if len(arguments) != 9:
        raise ValueError(f"Unexpected snote format: {text}")
    return ScoreNote(
        note_id=arguments[0],
        pitch=_spelled_pitch(arguments[1], arguments[2]),
        onset_beats=float(arguments[6]),
        offset_beats=float(arguments[7]),
        attributes=tuple(_list_items(arguments[8])),
    )


def _parse_performed_note(text: str, name: str, seconds_per_tick: float) -> PerformedNote:
    arguments = _call_arguments(text, name)
    if len(arguments) == 7 and arguments[1].startswith("["):
        # 5.0: note(Id,[Step,Alter],Octave,Onset,Offset,AdjOffset,Velocity)
        pitch = _spelled_pitch(arguments[1], arguments[2])
        onset, offset, velocity = arguments[3], arguments[4], arguments[6]
    elif len(arguments) == 7:
        # 1.0.0: note(Id,MidiPitch,Onset,Offset,Velocity,Channel,Track)
        pitch = int(arguments[1])
        onset, offset, velocity = arguments[2], arguments[3], arguments[4]
    else:
        raise ValueError(f"Unexpected note format: {text}")
    return PerformedNote(
        note_id=arguments[0],
        pitch=pitch,
        onset=int(onset) * seconds_per_tick,
        offset=int(offset) * seconds_per_tick,
        velocity=int(velocity),
    )


def _read_clock(lines: list[str], path: Path) -> float:
    values = {}
    for line in lines:
        if line.startswith("info(midiClock"):
            key, value = _call_arguments(line[:-1], "info")
            values[key] = int(value)
    if "midiClockUnits" not in values or "midiClockRate" not in values:
        raise ValueError(f"Missing MIDI clock info in {path}")
    if values["midiClockUnits"] <= 0 or values["midiClockRate"] <= 0:
        raise ValueError(f"Invalid MIDI clock info in {path}")
    # midiClockRate는 4분음표 하나의 길이(µs), midiClockUnits는 4분음표 하나의 tick 수다.
    return values["midiClockRate"] / _MICROSECONDS_PER_SECOND / values["midiClockUnits"]


def load_match(path: str | Path) -> NoteAlignment:
    """match 파일 하나를 읽어 정렬된 음, 빠진 악보 음, 추가된 연주 음으로 나눈다."""
    match_path = Path(path)
    if not match_path.exists():
        raise FileNotFoundError(match_path)
    if not match_path.is_file():
        raise ValueError(f"Not a match file: {match_path}")

    lines = [line.strip() for line in match_path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    seconds_per_tick = _read_clock(lines, match_path)

    matches = []
    deletions = []
    insertions = []
    for line in lines:
        if not line.endswith("."):
            raise ValueError(f"Invalid match line in {match_path}: {line}")
        body = line[:-1]
        try:
            if body.startswith("snote("):
                score_text, separator, performance_text = body.rpartition(")-")
                if not separator:
                    raise ValueError(f"Unexpected snote line: {line}")
                score_note = _parse_score_note(score_text + ")")
                if performance_text == "deletion":
                    deletions.append(score_note)
                else:
                    matches.append(
                        (
                            score_note,
                            _parse_performed_note(performance_text, "note", seconds_per_tick),
                        )
                    )
            elif body.startswith("insertion-note("):
                insertions.append(
                    _parse_performed_note(
                        body.removeprefix("insertion-"), "note", seconds_per_tick
                    )
                )
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid match line in {match_path}: {line}") from exc

    return NoteAlignment(
        matches=tuple(matches), deletions=tuple(deletions), insertions=tuple(insertions)
    )
