"""Protect ranking ties, independent-feature checks and work-balanced aggregation."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts"))
from validate_feature_search_roles import (aggregate, block_components, cohort_ids, evaluate,
                                          heldout_score, median_contrast, midrank, topk_membership)


class FeatureSearchRolesTest(unittest.TestCase):
    def test_tied_rank_and_topk_ignore_ineligible_candidates(self):
        distances=np.array([np.inf,.2,.2,.2,.8])
        self.assertEqual(midrank(distances,1),2)
        weights=topk_membership(distances,2)
        np.testing.assert_allclose(weights,[0,2/3,2/3,2/3,0])
        self.assertAlmostEqual(weights.sum(),2)
        with self.assertRaises(ValueError):midrank(distances,0)

    def test_pedal_block_has_same_weight_despite_more_coordinates(self):
        vectors=np.zeros((2,14));vectors[1,[0,1]]=2;vectors[1,8:]=2
        comp=block_components(vectors)
        np.testing.assert_allclose(comp[:,0,1],[4,0,0,0,4])
        with self.assertRaises(ValueError):block_components(np.full((2,14),np.nan))

    def test_ties_are_half_points_and_only_same_work_is_comparator(self):
        pieces=np.array(['query','target','target','target','other'])
        score,ties,count=heldout_score(np.array([0,1,1,2,.01]),pieces,1)
        self.assertEqual(score,.75);self.assertEqual(ties,.5);self.assertEqual(count,2)
        equal=heldout_score(np.ones(5),pieces,1)
        self.assertEqual(equal[0],.5)

    def test_random_same_work_candidate_has_half_expected_agreement_with_ties(self):
        pieces=np.array(['query','target','target','target','target'])
        distances=np.array([0,1,1,2,4.])
        scores=[heldout_score(distances,pieces,b)[0] for b in range(1,5)]
        self.assertAlmostEqual(np.mean(scores),.5)

    def test_comparators_exclude_different_grid_of_same_work(self):
        rows=[{'piece':'same','grid':g} for g in ['a','a','b','b']]
        cohorts=cohort_ids(rows)
        score,ties,count=heldout_score(np.array([1,2,0,0.]),cohorts,0)
        self.assertEqual(count,1);self.assertEqual(score,1);self.assertEqual(ties,0)

    def test_search_does_not_use_heldout_feature(self):
        rows=[{'key':str(i),'piece':'abc'[i//3],'grid':'one'} for i in range(9)]
        vectors=np.zeros((9,14));vectors[3:,0]=[1,2,3,4,5,6]
        original=vectors.copy()
        metrics,held,*_=evaluate(vectors,rows)
        first=next(r for r in held if r['query_key']=='0' and r['feature']=='Tempo')
        self.assertEqual(first['selected_key'],'3');self.assertEqual(first['pairwise_agreement'],1)
        np.testing.assert_array_equal(vectors,original)
        vectors[3,0]=10
        _,held_changed,*_=evaluate(vectors,rows)
        second=next(r for r in held_changed if r['query_key']=='0' and r['feature']=='Tempo')
        self.assertEqual(second['selected_key'],first['selected_key'])
        self.assertEqual(second['pairwise_agreement'],0)
        self.assertTrue(all(r['piece']!=r['selected_piece'] for r in metrics))
        self.assertTrue(all(r['top5_retained']==1 for r in metrics if r['config']=='all'))

    def test_contrast_is_middle_distance_not_farthest(self):
        pieces=np.array(['a','b','b','b','b','b'])
        distances=np.array([[np.inf,.1,.2,.3,.4,.5]])
        self.assertEqual(median_contrast(0,1,pieces,distances),3)

    def test_work_macro_does_not_give_more_weight_to_larger_work(self):
        metrics=[];held=[]
        for piece,count,value in [('small',2,0.),('large',8,1.)]:
            for _ in range(count):
                metrics.append({'mode':'full','config':'all','piece':piece,'grid':'one',
                                'neighbor_changed':value,'baseline_rank':1,'baseline_rank_fraction':value,'top5_retained':value})
                held.append({'mode':'full','feature':'Tempo','piece':piece,'grid':'one',
                             'pairwise_agreement':value,'pairwise_ties':0.})
        _,_,summary=aggregate(metrics,held,seed=1,samples=100)
        for row in summary:
            if row['metric']!='pairwise_ties':self.assertEqual(row['macro_value'],.5)


if __name__=='__main__':unittest.main()
