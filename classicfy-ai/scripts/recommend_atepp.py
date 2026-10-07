"""Feature-only baseline: aggregate liked performances and rank one work.

Example: python classicfy-ai/scripts/recommend_atepp.py --favorites 00025 00048
         --work 15 --output classicfy-ai/analysis/atepp/example_ranking.csv
This baseline uses candidate-pool normalization; it is not a learned encoder.
"""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from validate_atepp_recommendation import vector,distance,BLOCKS


def recommend(catalog, favorite_ids, composition_id):
    favorites=catalog[catalog.perf_id.isin(favorite_ids)]
    missing=set(favorite_ids)-set(favorites.perf_id)
    if missing: raise ValueError(f'Unknown or ineligible favorite IDs: {sorted(missing)}')
    candidates=catalog[(catalog.composition_id.astype(str)==str(composition_id)) & ~catalog.perf_id.isin(favorite_ids)].copy()
    if 'sha256' in catalog:
        candidates=candidates[~candidates.sha256.isin(favorites.sha256)].drop_duplicates('sha256')
    if candidates.empty: raise ValueError('No eligible unseen candidates for this work')
    def read(path):
        with np.load(path) as z: return vector(z['normalized'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        profile=np.nanmean(np.array([read(p) for p in favorites.normalized_cache]),axis=0)
    gallery=np.array([read(p) for p in candidates.normalized_cache])
    candidates['feature_distance']=distance(profile,gallery)
    for name,cols in BLOCKS.items():
        delta=(gallery[:,cols]-profile[cols])**2
        support=np.isfinite(delta).sum(axis=1)
        candidates[name+'_distance']=np.sqrt(np.divide(np.nansum(delta,axis=1),support,
                                                      out=np.full(len(gallery),np.nan),where=support>0))
    candidates=candidates[np.isfinite(candidates.feature_distance)].sort_values(['feature_distance','perf_id'])
    if candidates.empty: raise ValueError('Insufficient shared feature blocks')
    candidates.insert(0,'rank',candidates.feature_distance.rank(method='min').astype(int))
    return candidates[['rank','perf_id','artist','composer','track','feature_distance',
                       *[name+'_distance' for name in BLOCKS]]]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--favorites',nargs='+',required=True)
    p.add_argument('--work',required=True); p.add_argument('--catalog',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.catalog is None:
        research=Path('classicfy-ai/analysis/atepp_research/performance_summary.csv')
        args.catalog=research if research.exists() else Path('classicfy-ai/analysis/atepp/performance_summary.csv')
    catalog=pd.read_csv(args.catalog,dtype={'perf_id':str})
    result=recommend(catalog,args.favorites,args.work)
    print(result.head(10).to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(args.output,index=False,encoding='utf-8-sig')


if __name__=='__main__': main()
