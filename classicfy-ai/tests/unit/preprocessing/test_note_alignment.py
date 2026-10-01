import tempfile
import unittest
from pathlib import Path

from preprocessing import load_match

# 480 tick = 4분음표 하나 = 0.5초이므로 1 tick = 1/960초다.
CLOCK = "info(midiClockUnits,480).\ninfo(midiClockRate,500000).\n"

VERSION_1 = CLOCK + """info(matchFileVersion,1.0.0).
scoreprop(timeSignature,4/4,1:1,0,0.0000).
snote(n1-1,[C,n],4,1:1,0,1/4,0.0000,1.0000,[v1,staff1,staccato])-note(n0,60,960,1200,64,0,0).
snote(n2-1,[C,#],6,1:2,0,0,1.0000,1.0000,[v1,staff1,grace])-note(n1,85,1400,1440,50,0,0).
snote(n3-1,[E,n],4,1:3,0,1/4,2.0000,3.0000,[v2,staff2])-deletion.
insertion-note(n2,67,2000,2100,40,0,0).
sustain(0,127).
"""

VERSION_5 = CLOCK + """info(matchFileVersion,5.0).
snote(n1-1,[F,x],3,1:1,0,1/8,0.0,0.5,[])-note(n0,[F,x],3,480,720,1440,70).
snote(n2-1,[B,bb],3,1:1,1/8,1/8,0.5,1.0,[])-note(n1,[B,bb],3,960,1100,1440,71).
"""


class LoadMatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _write(self, text: str) -> Path:
        path = Path(self.temp_dir.name) / "take.match"
        path.write_text(text, encoding="utf-8")
        return path

    def test_version_1_matches_deletions_and_insertions(self) -> None:
        alignment = load_match(self._write(VERSION_1))

        self.assertEqual(len(alignment.matches), 2)
        score, performed = alignment.matches[0]
        self.assertEqual(score.note_id, "n1-1")
        self.assertEqual(score.pitch, 60)
        self.assertEqual((score.onset_beats, score.offset_beats), (0.0, 1.0))
        self.assertEqual(score.attributes, ("v1", "staff1", "staccato"))
        self.assertFalse(score.is_grace)
        self.assertEqual(performed.note_id, "n0")
        self.assertEqual(performed.pitch, 60)
        self.assertAlmostEqual(performed.onset, 1.0)
        self.assertAlmostEqual(performed.offset, 1.25)
        self.assertAlmostEqual(performed.duration, 0.25)
        self.assertEqual(performed.velocity, 64)

        self.assertTrue(alignment.matches[1][0].is_grace)
        self.assertEqual(alignment.matches[1][0].pitch, 85)
        self.assertEqual([note.note_id for note in alignment.deletions], ["n3-1"])
        self.assertEqual([note.pitch for note in alignment.insertions], [67])

    def test_version_5_uses_spelled_pitch_and_key_release(self) -> None:
        alignment = load_match(self._write(VERSION_5))

        (first_score, first), (second_score, second) = alignment.matches
        self.assertEqual((first_score.pitch, first.pitch), (55, 55))  # F##3 = G3
        self.assertEqual((second_score.pitch, second.pitch), (57, 57))  # Bbb3 = A3
        # AdjOffset(1440 tick)가 아니라 Offset(720 tick)을 연주 음의 끝으로 쓴다.
        self.assertAlmostEqual(first.offset, 0.75)
        self.assertEqual(second.velocity, 71)

    def test_missing_clock_info_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing MIDI clock"):
            load_match(self._write(VERSION_1.replace("info(midiClockRate,500000).\n", "")))

    def test_malformed_note_line_raises(self) -> None:
        text = CLOCK + "snote(n1-1,[C,n],4,1:1,0,1/4,0.0,1.0,[])-note(n0,60,960).\n"
        with self.assertRaisesRegex(ValueError, "Invalid match line"):
            load_match(self._write(text))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_match(Path(self.temp_dir.name) / "missing.match")


if __name__ == "__main__":
    unittest.main()
