import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from validate_atepp_recommendation import rank_metrics,distance,vector


class ATEPPEvaluationTests(unittest.TestCase):
    def test_metadata_ties_receive_random_expected_mrr(self):
        m=rank_metrics(np.zeros(4),2)
        self.assertEqual(m['hit1'],.25)
        self.assertAlmostEqual(m['mrr'],np.mean(1/np.arange(1,5)))

    def test_pedal_subchannels_do_not_overweight_pedaling(self):
        q=np.zeros(14); c=np.zeros((2,14)); c[0,0:2]=1; c[1,8:14]=1
        np.testing.assert_allclose(distance(q,c),[np.sqrt(.2),np.sqrt(.2)])

    def test_missing_feature_stays_missing(self):
        x=np.ones((20,7)); x[:,3]=np.nan
        v=vector(x); self.assertTrue(np.isnan(v[6:8]).all())

    def test_missing_candidate_block_cannot_improve_its_rank(self):
        q=np.zeros(14); c=np.ones((2,14)); c[1,8:14]=np.nan
        scores=distance(q,c)
        self.assertEqual(scores[0],1.)
        self.assertTrue(np.isinf(scores[1]))


if __name__=='__main__': unittest.main()
