import tempfile
import unittest
from pathlib import Path

import pretty_midi

from preprocessing.midi_loader import load_midi


class MidiLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.path = self.root / "sample.mid"

    def test_loads_all_piano_parts_and_only_cc64(self) -> None:
        midi = pretty_midi.PrettyMIDI()
        first = pretty_midi.Instrument(program=0)
        first.notes.extend(
            [
                pretty_midi.Note(velocity=90, pitch=60, start=1.0, end=1.5),
                pretty_midi.Note(velocity=80, pitch=72, start=0.5, end=0.75),
            ]
        )
        first.control_changes.extend(
            [
                pretty_midi.ControlChange(number=64, value=63, time=0.1),
                pretty_midi.ControlChange(number=1, value=42, time=0.3),
                pretty_midi.ControlChange(number=64, value=0, time=2.0),
            ]
        )
        second = pretty_midi.Instrument(program=1)
        second.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=1.0, end=1.25))
        second.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0.1))
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        drum.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.2, end=0.4))
        drum.control_changes.append(pretty_midi.ControlChange(number=64, value=80, time=0.05))
        midi.instruments.extend([first, second, drum])
        midi.write(str(self.path))

        data = load_midi(self.path)

        self.assertEqual(
            [(note.pitch, note.velocity, note.instrument_idx, note.program) for note in data.notes],
            [(72, 80, 0, 0), (60, 90, 0, 0), (60, 70, 1, 1)],
        )
        self.assertAlmostEqual(data.notes[0].start, 0.5, places=2)
        self.assertAlmostEqual(data.notes[0].end, 0.75, places=2)
        self.assertAlmostEqual(data.notes[0].duration, 0.25, places=2)
        self.assertEqual(
            [(pedal.value, pedal.instrument_idx) for pedal in data.pedals],
            [(63, 0), (127, 1), (0, 0)],
        )
        self.assertAlmostEqual(data.pedals[0].time, 0.1, places=2)
        self.assertAlmostEqual(data.duration, 2.0, places=2)

    def test_valid_empty_midi_returns_empty_events(self) -> None:
        pretty_midi.PrettyMIDI().write(str(self.path))

        data = load_midi(self.path)

        self.assertEqual(data.notes, [])
        self.assertEqual(data.pedals, [])

    def test_invalid_paths_and_corrupt_midi(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_midi(self.path)
        with self.assertRaises(ValueError):
            load_midi(self.root)
        self.path.write_bytes(b"not a MIDI file")
        with self.assertRaisesRegex(ValueError, "Could not parse MIDI file"):
            load_midi(self.path)


if __name__ == "__main__":
    unittest.main()
