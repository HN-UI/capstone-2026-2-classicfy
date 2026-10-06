import unittest
from pathlib import Path
import tempfile
import mido
import numpy as np
from preprocessing import MidiData,NoteEvent,ScoreNote
from preprocessing.atepp_alignment import load_score_midi,align_score_performance,canonical_path


class ATEPPAlignmentTests(unittest.TestCase):
    def test_quarter_positions_ignore_score_tempo(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'score.mid'; midi=mido.MidiFile(ticks_per_beat=480)
            track=mido.MidiTrack(); midi.tracks.append(track)
            track.extend([mido.MetaMessage('set_tempo',tempo=1000000),mido.Message('note_on',note=60,velocity=64),
                          mido.Message('note_off',note=60,time=480)])
            midi.save(path); score=load_score_midi(path)
            self.assertEqual(score[0].onset_beats,0)
            self.assertEqual(score[0].duration_beats,1)

    def test_constant_tempo_synthetic_pair_has_correct_map(self):
        pitches=np.random.default_rng(12).integers(48,84,size=100)
        score=tuple(ScoreNote(str(i),int(p),float(i),i+.8,()) for i,p in enumerate(pitches))
        midi=MidiData([NoteEvent(int(p),i*.5+.25,(i+.8)*.5+.25,80,0,0)
                       for i,p in enumerate(pitches)],[],51.)
        alignment,grid,times,metrics=align_score_performance(score,midi)
        self.assertTrue(metrics['accepted'],metrics)
        self.assertGreater(metrics['score_recall'],.95)
        np.testing.assert_allclose(times,grid*.5+.25,atol=.03)

    def test_unrelated_pitches_rejected(self):
        score=tuple(ScoreNote(str(i),60+(i%5),float(i),i+.8,()) for i in range(60))
        midi=MidiData([NoteEvent(80+(i%5),i*.5,(i+.8)*.5,80,0,0) for i in range(60)],[],31.)
        with self.assertRaisesRegex(ValueError,'anchors'):
            align_score_performance(score,midi)

    def test_unicode_and_zip_mojibake_resolve_same_score(self):
        self.assertEqual(canonical_path('Pre╠ülude.mxl.midi'),canonical_path('Prélude.mxl.midi'))


if __name__=='__main__': unittest.main()
