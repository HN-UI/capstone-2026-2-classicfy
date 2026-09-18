"""MIDI 파일에서 원본 음표와 서스테인 페달 이벤트를 읽는다."""

from dataclasses import dataclass
from pathlib import Path
import pretty_midi


@dataclass
class NoteEvent:
    """음표 하나와 원본 MIDI 파트 및 악기 프로그램 정보를 담는다."""
    pitch: int
    start: float
    end: float
    velocity: int
    instrument_idx: int
    program: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PedalEvent:
    """서스테인 페달 이벤트 하나와 원본 CC64 값을 담는다."""
    time: float
    value: int
    instrument_idx: int


@dataclass
class MidiData:
    """MIDI 파일 하나의 음표, 페달 이벤트, 전체 길이를 담는다."""
    notes: list[NoteEvent]
    pedals: list[PedalEvent]
    duration: float


def load_midi(path: str | Path) -> MidiData:
    """드럼을 제외한 모든 MIDI 파트에서 음표와 원본 CC64 이벤트를 읽는다."""
    midi_path = Path(path)
    if not midi_path.exists():
        raise FileNotFoundError(midi_path)
    if not midi_path.is_file():
        raise ValueError(f"Not a MIDI file: {midi_path}")

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        raise ValueError(f"Could not parse MIDI file: {midi_path}") from exc

    notes = []
    pedals = []
    # 피아노 한 곡의 악보도 여러 MIDI 파트로 나뉘어 있을 수 있다.
    for instrument_idx, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            notes.append(
                NoteEvent(
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                    velocity=int(note.velocity),
                    instrument_idx=instrument_idx,
                    program=int(instrument.program),
                )
            )

        for control in instrument.control_changes:
            if control.number == 64:
                pedals.append(
                    PedalEvent(
                        time=float(control.time),
                        value=int(control.value),
                        instrument_idx=instrument_idx,
                    )
                )

    notes.sort(key=lambda note: (note.start, note.pitch, note.instrument_idx))
    pedals.sort(key=lambda pedal: (pedal.time, pedal.instrument_idx))
    return MidiData(notes=notes, pedals=pedals, duration=float(midi.get_end_time()))
