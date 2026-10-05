"""Safeguards for cross-piece example selection and displayed percentiles."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts"))
from validate_interpretation_examples import cross_selection, percentile_profile


class InterpretationExamplesTest(unittest.TestCase):
    def test_nearest_excludes_query_piece_and_contrast_stays_in_target_piece(self):
        vectors=np.zeros((5,14))
        vectors[:,0]=[0,.001,1,3,8]
        pieces=np.array(["a","a","b","b","c"])
        b,c,distances=cross_selection(vectors,pieces,0)
        self.assertEqual(b,2)
        self.assertEqual(c,3)
        self.assertNotEqual(pieces[b],pieces[0])
        self.assertEqual(pieces[b],pieces[c])
        self.assertLess(distances[b],distances[c])

    def test_missing_other_piece_is_an_error(self):
        with self.assertRaises(ValueError):
            cross_selection(np.zeros((3,14)),np.array(["a"]*3),0)

    def test_equal_contrast_is_distinct_and_not_falsely_claimed_further(self):
        b,c,d=cross_selection(np.zeros((4,14)),np.array(["a","a","b","b"]),0)
        self.assertNotEqual(b,c)
        self.assertEqual(d[b],d[c])

    def test_midrank_ties_and_input_preservation(self):
        cohort=np.array([[0,4],[1,4],[2,4]],dtype=float)
        copy=cohort.copy()
        np.testing.assert_allclose(percentile_profile(cohort[0],cohort),[100/6,50])
        np.testing.assert_allclose(percentile_profile(cohort[2],cohort),[500/6,50])
        np.testing.assert_array_equal(cohort,copy)


if __name__=="__main__":unittest.main()
