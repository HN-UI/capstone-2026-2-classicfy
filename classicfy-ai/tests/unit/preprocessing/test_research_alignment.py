"""Check evaluation isolation, identity ambiguity, and a known synthetic mapping."""
import unittest
from pathlib import Path
import sys
import numpy as np
from preprocessing import MidiData, NoteEvent, ScoreNote
from preprocessing.research_alignment import alignment_from_predictions, align_research, note_arrays
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from benchmark_research_alignment import identity_map
from preprocessing.exact_dtw_kernel import membership_cumulative, pairwise_cumulative


class ResearchAlignmentTests(unittest.TestCase):
    def setUp(self):
        pitches=np.random.default_rng(12).integers(48,84,size=64)
        self.score=tuple(ScoreNote(str(i),int(p),float(i),i+.8,()) for i,p in enumerate(pitches))
        self.midi=MidiData([NoteEvent(int(p),i*.5+.25,(i+.8)*.5+.25,80,0,0)
                           for i,p in enumerate(pitches)],[],33.)

    def test_both_official_matchers_recover_known_tempo(self):
        for model in ('dual_dtw','gluenote'):
            with self.subTest(model=model):
                alignment,q,t,quality=align_research(self.score,self.midi,model)
                self.assertGreater(len(alignment.matches),55)
                np.testing.assert_allclose(t,q*.5+.25,atol=.03)

    def test_conflicting_correspondences_are_all_removed_without_guessing(self):
        predictions=[dict(label='match',score_id='0',performance_id='0'),
                     dict(label='match',score_id='1',performance_id='0')]
        alignment=alignment_from_predictions(predictions,self.score,self.midi)
        self.assertEqual(len(alignment.matches),0)
        self.assertEqual(len(alignment.deletions),len(self.score))

    def test_equal_pitch_onset_reference_identity_is_ambiguous(self):
        reference=[ScoreNote('reference',60,0.,1.,())]
        target=[ScoreNote('a',60,0.,1.,()),ScoreNote('b',60,0.,2.,())]
        mapping=identity_map(reference,target,lambda n:n.onset_beats,lambda n:n.onset_beats,.002)
        self.assertEqual(mapping,{})

    def test_matcher_inputs_contain_no_reference_correspondence(self):
        s,p=note_arrays(self.score,self.midi)
        self.assertNotIn('performance_id',s.dtype.names)
        self.assertNotIn('score_id',p.dtype.names)
        self.assertEqual(s[3]['onset_quarter'],3.)
        self.assertEqual(p[3]['onset_sec'],1.75)

    def test_compiled_cost_matrix_is_exactly_equal_to_published_python(self):
        import parangonar.dp.dtw as module
        from parangonar.dp.metrics import element_of_set_metric
        original=getattr(module.cdist_dtw_single_loop,'_classicfy_original',module.cdist_dtw_single_loop)
        rng=np.random.default_rng(13)
        pitches=rng.integers(48,84,size=40)
        sets=[set(map(int,rng.integers(48,84,size=4))) for _ in range(35)]
        masks=np.zeros((len(sets),128),bool)
        for i,s in enumerate(sets): masks[i,list(s)]=True
        expected=original(pitches,sets,element_of_set_metric)
        np.testing.assert_array_equal(membership_cumulative(pitches,masks),expected)

    def test_pitchwise_scalar_costs_and_recurrence_are_exact(self):
        import parangonar.dp.dtw as module
        from scipy.spatial.distance import cdist,euclidean
        rng=np.random.default_rng(56)
        x=rng.normal(size=(31,1)); y=rng.normal(size=(27,1))
        distances=cdist(x,y,euclidean)
        np.testing.assert_array_equal(np.abs(x[:,0,None]-y[None,:,0]),distances)
        original=getattr(module.dtw_dmatrix_from_pairwise_dmatrix,'_classicfy_original',
                         module.dtw_dmatrix_from_pairwise_dmatrix)
        np.testing.assert_array_equal(pairwise_cumulative(distances),original(distances))


if __name__=='__main__': unittest.main()
