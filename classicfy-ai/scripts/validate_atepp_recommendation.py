"""Evidence about representation, not a claim of measured user satisfaction.

All uncertainty intervals resample works. Composer/era probes hold out works.
Self-retrieval and performer proxies are explicitly not preference ground truth.
"""
import argparse
from pathlib import Path
import sys
import json
import warnings
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from extend_atepp import CHANNELS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLOCKS = {'tempo':[0,1],'rubato':[2,3],'dynamics':[4,5],'articulation':[6,7],
          'pedaling':list(range(8,14))}


def vector(x):
    result=[]
    for c in range(7):
        v=x[:,c]; v=v[np.isfinite(v)]
        result.extend([float(np.median(np.abs(v)) if c==1 else v.mean()),
                       float(np.diff(np.percentile(v,[5,95]))[0])] if len(v) else [np.nan,np.nan])
    return np.array(result)


def distance(query, candidates, without=None):
    pieces=[]
    for name, cols in BLOCKS.items():
        if name==without: continue
        if not np.isfinite(query[cols]).all(): continue
        diff=(candidates[:,cols]-query[cols])**2
        values=np.mean(diff,axis=1)
        values[~np.isfinite(diff).all(axis=1)]=np.nan
        pieces.append(values)
    if len(pieces)<3: return np.full(len(candidates),np.inf)
    a=np.array(pieces)
    support=np.isfinite(a).sum(axis=0)
    # Every ranked candidate must expose the same complete musical blocks.
    return np.sqrt(np.divide(np.nansum(a,axis=0),support,out=np.full(len(candidates),np.inf),where=support==len(pieces)))


def rank_metrics(scores,target):
    value=scores[target]
    if not np.isfinite(value): return dict(hit1=np.nan,mrr=np.nan,ndcg=np.nan,rank=np.nan)
    ties=np.isclose(scores,value,atol=1e-10,rtol=0)
    less=int(np.sum((scores<value)&~ties)); count=int(ties.sum())
    ranks=np.arange(less+1,less+count+1)
    return dict(hit1=float(less==0)/count,mrr=float(np.mean(1/ranks)),
                ndcg=float(np.mean(1/np.log2(ranks+1))),rank=float(ranks.mean()))


def bootstrap_work(frame,column,seed=20261006):
    means=frame.groupby('composition_id')[column].mean().dropna().to_numpy()
    if not len(means): return dict(mean=None,low=None,high=None,works=0)
    rng=np.random.default_rng(seed)
    values=np.array([rng.choice(means,len(means),replace=True).mean() for _ in range(1000)])
    return dict(mean=float(means.mean()),low=float(np.percentile(values,2.5)),
                high=float(np.percentile(values,97.5)),works=len(means))


def integrity(summary,audit,out):
    rows=[]; issues=[]; total=0
    for r in summary.to_dict('records'):
        with np.load(r['normalized_cache']) as z:
            x=z['normalized']; mask=z['mask']; raw=z['raw']; rel=z['relative']
            total+=int(mask.sum())
            if x.shape!=mask.shape or x.shape[1]!=7 or x.shape[0]!=len(z['score_beats'])-1:
                issues.append(dict(perf_id=r['perf_id'],issue='shape'))
            if np.any(mask & ~np.isfinite(x)) or np.any(~mask & ~np.isnan(x)):
                issues.append(dict(perf_id=r['perf_id'],issue='mask_nonfinite'))
            if np.isinf(raw).any() or np.isinf(rel).any():
                issues.append(dict(perf_id=r['perf_id'],issue='infinite_raw_relative'))
            for c,name in enumerate(CHANNELS):
                v=x[:,c][mask[:,c]]; rv=raw[:,c]; finite=rv[np.isfinite(rv)]
                invalid=int(np.sum((finite < -1e-8)|(finite > 1+1e-8))) if name in ('dynamics','pedal_depth','pedal_down_ratio') else 0
                if name=='pedal_changes': invalid=int(np.sum((finite<0)|(finite!=np.round(finite))))
                if invalid: issues.append(dict(perf_id=r['perf_id'],issue=f'{name}_domain',count=invalid))
                rows.append(dict(perf_id=r['perf_id'],composition_id=r['composition_id'],channel=name,
                                 valid=len(v),missing=int((~mask[:,c]).sum()),
                                 over_6=int(np.sum(np.abs(v)>6)),over_10=int(np.sum(np.abs(v)>10)),
                                 maximum_abs=float(np.max(np.abs(v))) if len(v) else np.nan,
                                 raw_invalid=invalid))
    pd.DataFrame(rows).to_csv(out/'feature_quality.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(issues,columns=['perf_id','issue','count']).to_csv(out/'invalid_feature_values.csv',index=False)
    duplicates=audit[audit.duplicated('sha256',keep=False)&audit.sha256.notna()].sort_values('sha256')
    duplicates.to_csv(out/'exact_duplicate_midi.csv',index=False,encoding='utf-8-sig')
    return dict(normalized_performances=len(summary),valid_cells=total,invalid_feature_issues=len(issues),
                duplicate_metadata_ids=int(audit.perf_id.duplicated().sum()),exact_duplicate_rows=len(duplicates),
                invalid_notes=int(audit.invalid_notes.fillna(0).sum()),invalid_pedals=int(audit.invalid_pedals.fillna(0).sum()),
                feature_channels=7,missing_is_masked=True)


def write_review_queue(summary,out):
    q=pd.read_csv(out/'feature_quality.csv',dtype={'perf_id':str})
    catalog=summary.set_index('perf_id')
    q['fraction_over6']=q.over_6/q.valid.replace(0,np.nan)
    review=q[(q.maximum_abs>20)|(q.fraction_over6>.02)].copy()
    review=review.merge(catalog[['composition_id','composer','artist','track','normalized_cache']],
                        left_on='perf_id',right_index=True,suffixes=('','_catalog'))
    review.to_csv(out/'feature_review_queue.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for c,name in enumerate(CHANNELS):
        for r in q[q.channel==name].nlargest(10,'maximum_abs').to_dict('records'):
            meta=catalog.loc[r['perf_id']]
            with np.load(meta.normalized_cache) as z:
                values=z['normalized'][:,c]
                if not np.isfinite(values).any(): continue
                i=int(np.nanargmax(np.abs(values)))
                rows.append(dict(perf_id=r['perf_id'],composition_id=meta.composition_id,channel=name,artist=meta.artist,
                                 score_quarter=float(z['score_beats'][i]),raw=float(z['raw'][i,c]),
                                 relative=float(z['relative'][i,c]),normalized=float(values[i]),
                                 implied_scale=float(z['relative'][i,c]/values[i]) if values[i] else np.nan))
    pd.DataFrame(rows).to_csv(out/'extreme_beat_examples.csv',index=False,encoding='utf-8-sig')
    return dict(flagged_performances=int(review.perf_id.nunique()),flagged_channel_rows=len(review),
                policy='Review only: max abs > 20 or fraction abs > 6 exceeds 2%; not proof of corruption')


def leakage_probes(frame,out):
    results=[]; effects=[]
    for label in ['composer','era']:
        work_counts=frame.groupby(label).composition_id.nunique()
        d=frame[frame[label].isin(work_counts[work_counts>=5].index)].copy()
        y=d[label].astype(str).to_numpy(); groups=d.composition_id.to_numpy()
        if len(np.unique(y))<2: continue
        split=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(d,y,groups))
        for layer in ['raw','relative','norm']:
            cols=[f'{layer}_{n}_{stat}' for n in CHANNELS for stat in ['abs' if n=='rubato' else 'mean','range']]
            x=d[cols].to_numpy(float)
            for fold,(train,test) in enumerate(split):
                for name,clf in [('feature',LogisticRegression(max_iter=2000,C=.1,class_weight='balanced')),
                                 ('chance',DummyClassifier(strategy='stratified',random_state=fold+42))]:
                    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RobustScaler(),clf)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore'); pred=model.fit(x[train],y[train]).predict(x[test])
                    results.append(dict(label=label,layer=layer,model=name,fold=fold,
                                        balanced_accuracy=balanced_accuracy_score(y[test],pred),
                                        test_works=len(np.unique(groups[test])),test_performances=len(test),
                                        classes=len(np.unique(y))))
            # Era/composer effect at WORK level to avoid pretending recordings are independent.
            w=d.groupby('composition_id').agg({**{c:'mean' for c in cols},label:'first'})
            for col in cols:
                a=w[col].dropna(); labs=w.loc[a.index,label].to_numpy(); a=a.to_numpy()
                if len(a)<10 or np.var(a)<1e-12: continue
                def eta(labels):
                    total=np.sum((a-a.mean())**2)
                    return sum(np.sum(labels==v)*(a[labels==v].mean()-a.mean())**2 for v in np.unique(labels))/total
                observed=eta(labs); rng=np.random.default_rng(42)
                null=np.array([eta(rng.permutation(labs)) for _ in range(499)])
                effects.append(dict(label=label,layer=layer,coordinate=col,eta_squared=observed,
                                    permutation_p=(1+int(np.sum(null>=observed)))/500,works=len(a)))
    pd.DataFrame(results).to_csv(out/'metadata_leakage_probes.csv',index=False,encoding='utf-8-sig')
    effect_table=pd.DataFrame(effects)
    if len(effect_table):
        # Benjamini-Hochberg across all reported coordinate/label/layer tests.
        order=np.argsort(effect_table.permutation_p.to_numpy()); pvals=effect_table.permutation_p.to_numpy()[order]
        q=np.minimum.accumulate((pvals*len(pvals)/np.arange(1,len(pvals)+1))[::-1])[::-1]
        effect_table['bh_q']=np.nan; effect_table.loc[order,'bh_q']=np.minimum(q,1.)
    effect_table.to_csv(out/'metadata_effect_sizes.csv',index=False,encoding='utf-8-sig')
    return pd.DataFrame(results)


def variability(frame,sequences,out):
    rows=[]; variance=[]; examples=[]; preservation=[]
    for c,name in enumerate(CHANNELS):
        for layer in ['raw','relative','norm']:
            col=f'{layer}_{name}_mean'; d=frame[['composition_id',col]].dropna()
            overall=d[col].var(ddof=0)
            within=np.mean((d[col]-d.groupby('composition_id')[col].transform('mean'))**2)
            variance.append(dict(channel=name,layer=layer,total_variance=overall,
                                 within_work_variance=within,within_fraction=within/overall if overall>0 else np.nan))
    for group,d in frame.groupby('norm_group'):
        ids=d.index.to_numpy()
        if len(ids)>=2:
            first=sequences[ids[0]]
            for j in ids[1:]:
                other=sequences[j]
                delta_raw=first['raw']-other['raw']
                delta_rel=first['relative']-other['relative']
                valid=np.isfinite(delta_raw)&np.isfinite(delta_rel)
                # Rubato raw is already individually centered; all channels retain pair differences.
                error=float(np.max(np.abs(delta_raw[valid]-delta_rel[valid]))) if valid.any() else 0.
                preservation.append(dict(composition_id=d.iloc[0].composition_id,group=group,
                                         perf_a=frame.loc[ids[0],'perf_id'],perf_b=frame.loc[j,'perf_id'],
                                         max_pair_residual_error=error,valid_pairs=int(valid.sum())))
        if len(d)<3: continue
        candidates=np.array([vector(sequences[k]['normalized']) for k in ids])
        for pos,i in enumerate(ids):
            scores=distance(candidates[pos],candidates); scores[pos]=np.inf
            finite=scores[np.isfinite(scores)]
            if not len(finite): continue
            rows.append(dict(composition_id=d.iloc[0].composition_id,perf_id=frame.loc[i,'perf_id'],
                             candidates=len(d),median_distance=float(np.median(finite)),
                             nearest_distance=float(finite.min()),furthest_distance=float(finite.max()),
                             near_identical=int(np.sum(finite<1e-6))))
        if len(examples)<10 and len(d)>=5:
            order=np.argsort(candidates[:,0]); lo,hi=ids[order[0]],ids[order[-1]]
            examples.append(dict(composition_id=d.iloc[0].composition_id,track=d.iloc[0].track,
                                 slow_perf=frame.loc[lo,'perf_id'],fast_perf=frame.loc[hi,'perf_id'],
                                 slow_artist=frame.loc[lo,'artist'],fast_artist=frame.loc[hi,'artist'],
                                 tempo_residual_gap=float(candidates[order[-1],0]-candidates[order[0],0])))
    pd.DataFrame(rows).to_csv(out/'within_work_distances.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(variance).to_csv(out/'variance_decomposition.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(examples).to_csv(out/'same_work_examples.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(preservation).to_csv(out/'pair_difference_preservation.csv',index=False,encoding='utf-8-sig')
    cols=[f'norm_{n}_mean' for n in CHANNELS]
    frame[cols].corr(method='spearman').to_csv(out/'feature_redundancy.csv',encoding='utf-8-sig')
    return pd.DataFrame(rows)


def retrieval(frame,sequences,out):
    rng=np.random.default_rng(42); records=[]; contrast=[]; collection=[]
    for group,d in frame.groupby('norm_group'):
        if len(d)<3: continue
        ids=d.index.to_numpy(); gallery=[]; queries=[]
        stack=np.array([sequences[i]['normalized'] for i in ids])
        shared=np.all(np.isfinite(stack),axis=(0,2))
        collection.append(dict(group=group,composition_id=d.iloc[0].composition_id,performances=len(ids),
                               total_beats=stack.shape[1],shared_beats=int(shared.sum()),
                               shared_fraction=float(shared.mean()),eligible=bool(shared.sum()>=32 and shared.mean()>=.25)))
        if shared.sum()<32 or shared.mean()<.25: continue
        perm=rng.permutation(int(shared.sum())); half=int(shared.sum())//2
        for i in ids:
            x=sequences[i]['normalized'][shared]
            # Independent disjoint random beat subsets of the SAME performance.
            gallery.append(vector(x[perm[:half]])); queries.append(vector(x[perm[half:]]))
        gallery=np.array(gallery); queries=np.array(queries)
        for pos,i in enumerate(ids):
            for config in ['all',*['without_'+name for name in BLOCKS]]:
                scores=distance(queries[pos],gallery,config.replace('without_','') if config!='all' else None)
                metrics=rank_metrics(scores,pos)
                records.append(dict(composition_id=d.iloc[0].composition_id,perf_id=frame.loc[i,'perf_id'],
                                    config=config,candidates=len(d),chance_hit1=1/len(d),
                                    chance_mrr=float(np.mean(1/np.arange(1,len(d)+1))),**metrics))
            # Compare relative-only to metadata-constant ranking among same-work candidates.
            scores=distance(queries[pos],gallery); valid=scores[np.isfinite(scores)]
            if len(valid):
                contrast.append(dict(composition_id=d.iloc[0].composition_id,perf_id=frame.loc[i,'perf_id'],
                                     feature_score_range=float(np.ptp(valid)),metadata_score_range=0.,
                                     feature_distinct_scores=len(np.unique(np.round(valid,8))),candidates=len(d)))
    result=pd.DataFrame(records); result.to_csv(out/'disjoint_beat_retrieval.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(collection).to_csv(out/'retrieval_collection.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(contrast).to_csv(out/'metadata_vs_feature_ranking.csv',index=False,encoding='utf-8-sig')
    return result


def section_retrieval(frame,sequences,out):
    rows=[]
    for group,d in frame.groupby('norm_group'):
        if len(d)<3: continue
        ids=d.index.to_numpy(); early=[]; late=[]
        stack=np.array([sequences[i]['normalized'] for i in ids]); shared=np.all(np.isfinite(stack),axis=(0,2))
        if shared.sum()<32 or shared.mean()<.25: continue
        for i in ids:
            x=sequences[i]['normalized'][shared]; half=len(x)//2
            early.append(vector(x[:half])); late.append(vector(x[half:]))
        for pos,i in enumerate(ids):
            rows.append(dict(composition_id=d.iloc[0].composition_id,perf_id=frame.loc[i,'perf_id'],
                             candidates=len(ids),chance_hit1=1/len(ids),
                             **rank_metrics(distance(early[pos],np.array(late)),pos)))
    result=pd.DataFrame(rows); result.to_csv(out/'first_vs_second_half_retrieval.csv',index=False,encoding='utf-8-sig')
    return result


def cross_work_metadata_comparison(frame,sequences,out):
    """Does feature-nearest retrieval merely reproduce metadata neighborhoods?"""
    all_vectors=np.array([vector(sequences[i]['normalized']) for i in frame.index])
    results=[]; rng=np.random.default_rng(20261006)
    # At most 500 reproducibly selected queries; work-stratified uncertainty below.
    sample=frame.index.to_numpy()
    if len(sample)>500: sample=rng.choice(sample,500,replace=False)
    for i in sample:
        r=frame.loc[i]; eligible=frame.composition_id.to_numpy()!=r.composition_id
        scores=distance(all_vectors[i],all_vectors); scores[~eligible]=np.inf
        ids=np.flatnonzero(np.isfinite(scores))
        if len(ids)<10: continue
        top=ids[np.argsort(scores[ids])[:10]]
        for label in ['composer','era','artist']:
            equal=(frame[label].to_numpy()==r[label])
            results.append(dict(composition_id=r.composition_id,perf_id=r.perf_id,metadata=label,
                                top10_same_fraction=float(equal[top].mean()),
                                pool_same_fraction=float(equal[ids].mean()),
                                score_metadata_spearman=float(spearmanr(scores[ids],(~equal[ids]).astype(int)).statistic)
                                if np.unique(equal[ids]).size>1 else np.nan))
    d=pd.DataFrame(results); d.to_csv(out/'cross_work_metadata_neighborhoods.csv',index=False,encoding='utf-8-sig')
    return d


def paired_comparisons(retrieval_results,proxy_results,out):
    result={}
    if len(proxy_results):
        d=proxy_results.pivot(index=['composition_id','perf_id'],columns='model',values='hit1').reset_index()
        d['feature_minus_composer_era']=d.feature-d.composer_era_metadata
        d['feature_minus_artist']=d.feature-d.artist_metadata
        d.to_csv(out/'paired_proxy_comparisons.csv',index=False,encoding='utf-8-sig')
        result['proxy']={c:bootstrap_work(d,c) for c in ['feature_minus_composer_era','feature_minus_artist']}
    if len(retrieval_results):
        d=retrieval_results.pivot(index=['composition_id','perf_id'],columns='config',values='hit1').reset_index()
        for c in [x for x in d.columns if x.startswith('without_')]: d[c+'_minus_all']=d[c]-d['all']
        d.to_csv(out/'paired_ablation_comparisons.csv',index=False,encoding='utf-8-sig')
        result['ablation']={c:bootstrap_work(d,c) for c in d.columns if c.endswith('_minus_all')}
    return result


def ranking_stability(frame,sequences,out):
    """Local preference profiles from other works, with independent beat dropout."""
    rng=np.random.default_rng(19); rows=[]
    v={i:vector(z['normalized']) for i,z in sequences.items()}
    for group,d in frame.groupby('norm_group'):
        if len(d)<3: continue
        ids=d.index.to_numpy(); gallery=np.array([v[i] for i in ids])
        for i in ids[:min(5,len(ids))]:
            r=frame.loc[i]
            favorite_ids=frame[(frame.artist==r.artist)&(frame.composition_id!=r.composition_id)].drop_duplicates('composition_id').index[:5]
            if len(favorite_ids)<3: continue
            profile=np.nanmean([v[k] for k in favorite_ids],axis=0)
            noisy=[]
            for k in favorite_ids:
                x=sequences[k]['normalized']; keep=rng.choice(len(x),max(2,int(.8*len(x))),replace=False)
                noisy.append(vector(x[keep]))
            altered=np.nanmean(noisy,axis=0)
            for config in ['all',*['without_'+name for name in BLOCKS]]:
                without=config.replace('without_','') if config!='all' else None
                clean=distance(profile,gallery,without); noisy_scores=distance(altered,gallery,without)
                ok=np.isfinite(clean)&np.isfinite(noisy_scores)
                if ok.sum()<3: continue
                rows.append(dict(composition_id=r.composition_id,perf_id=r.perf_id,config=config,
                                 rank_spearman=float(spearmanr(clean[ok],noisy_scores[ok]).statistic),
                                 top1_unchanged=int(np.argmin(clean)==np.argmin(noisy_scores)),candidates=len(ids)))
    d=pd.DataFrame(rows); d.to_csv(out/'profile_dropout_ranking_stability.csv',index=False,encoding='utf-8-sig')
    return d


def outlier_sensitivity(frame,sequences,out):
    rows=[]
    for group,d in frame.groupby('norm_group'):
        if len(d)<3: continue
        ids=d.index.to_numpy(); original=np.array([vector(sequences[i]['normalized']) for i in ids]); removed=[]
        for i in ids:
            x=sequences[i]['normalized'].copy(); x[np.abs(x)>6]=np.nan
            removed.append(vector(x))
        removed=np.array(removed)
        for pos,i in enumerate(ids):
            a=distance(original[pos],original); b=distance(removed[pos],removed)
            a[pos]=np.inf; b[pos]=np.inf; ok=np.isfinite(a)&np.isfinite(b)
            if ok.sum()<3: continue
            rows.append(dict(composition_id=d.iloc[0].composition_id,perf_id=frame.loc[i,'perf_id'],
                             rank_spearman=float(spearmanr(a[ok],b[ok]).statistic),
                             top1_unchanged=int(np.argmin(a)==np.argmin(b))))
    d=pd.DataFrame(rows); d.to_csv(out/'outlier_removal_sensitivity.csv',index=False,encoding='utf-8-sig')
    return d


def performer_proxy(frame,sequences,out):
    vectors={i:vector(z['normalized']) for i,z in sequences.items()}
    results=[]; examples=[]
    for work,d in frame.groupby('norm_group'):
        if len(d)<3: continue
        ids=d.index.to_numpy(); gallery=np.array([vectors[i] for i in ids])
        for pos,i in enumerate(ids):
            r=frame.loc[i]
            favorites=frame[(frame.artist==r.artist)&(frame.composition_id!=r.composition_id)]
            favorites=favorites.sort_values('perf_id').drop_duplicates('composition_id')
            if len(favorites)<3: continue
            profile=np.nanmean(np.array([vectors[k] for k in favorites.index[:5]]),axis=0)
            scores=distance(profile,gallery)
            metadata=np.where(d.artist.to_numpy()==r.artist,0.,1.)
            for name,dist in [('feature',scores),('composer_era_metadata',np.zeros(len(d))),('artist_metadata',metadata)]:
                results.append(dict(composition_id=r.composition_id,perf_id=r.perf_id,model=name,
                                    favorite_works=min(5,len(favorites)),candidates=len(d),**rank_metrics(dist,pos)))
            if len(examples)<30 and np.isfinite(scores).all():
                order=np.argsort(scores)
                examples.append(dict(composition_id=r.composition_id,profile_artist=r.artist,target=r.perf_id,
                                     feature_top1=frame.loc[ids[order[0]],'perf_id'],
                                     feature_top1_artist=frame.loc[ids[order[0]],'artist'],
                                     favorite_ids=';'.join(favorites.perf_id.iloc[:5]),
                                     target_rank=rank_metrics(scores,pos)['rank']))
    result=pd.DataFrame(results); result.to_csv(out/'cross_work_performer_proxy.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(examples).to_csv(out/'ranking_examples.csv',index=False,encoding='utf-8-sig')
    return result


def figures(probes,distances,retr,proxy,out):
    plt.rcParams.update({'font.size':11,'figure.dpi':140})
    fig,ax=plt.subplots(figsize=(8,4))
    if len(probes):
        p=probes.groupby(['label','layer','model']).balanced_accuracy.mean().unstack('model')
        p.plot.bar(ax=ax,color=['#b0b7c1','#2563eb']); ax.set_ylim(0,1); ax.set_ylabel('Balanced accuracy')
    ax.set_title('Composer / era probes: held-out works'); fig.tight_layout(); fig.savefig(out/'metadata_leakage.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if len(retr):
        p=retr.groupby(['composition_id','config']).hit1.mean().groupby('config').mean().sort_values(ascending=False)
        p.plot.bar(ax=ax,color='#2563eb'); ax.axhline(retr[retr.config=='all'].groupby('composition_id').chance_hit1.mean().mean(),color='#f97316',label='Random'); ax.legend()
    ax.set_ylabel('Top-1 accuracy'); ax.set_ylim(0,1); ax.set_title('Disjoint-beat self-retrieval (not user preference)')
    fig.tight_layout(); fig.savefig(out/'retrieval_ablation.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if len(distances):
        values=np.sort(distances.nearest_distance.dropna().to_numpy())
        positive=values[values>0]
        if len(positive):
            ax.plot(positive,np.arange(1,len(positive)+1)/len(positive),color='#2563eb',linewidth=2)
            ax.set_xscale('log'); ax.set_ylim(0,1)
    ax.set_xlabel('Nearest other performance distance (log scale)'); ax.set_ylabel('Cumulative fraction')
    ax.set_title('Same-work interpretation differences'); fig.tight_layout(); fig.savefig(out/'same_work_distances.png'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('classicfy-ai/analysis/atepp'))
    args=p.parse_args(); out=args.output.resolve()
    audit=pd.read_csv(out/'midi_audit.csv',dtype={'perf_id':str})
    frame=pd.read_csv(out/'performance_summary.csv',dtype={'perf_id':str})
    quality=integrity(frame,audit,out); print('Integrity completed',flush=True)
    review=write_review_queue(frame,out)
    # Exclude exact duplicate MIDI globally from statistical and retrieval analyses.
    hashes=audit.set_index('perf_id').sha256
    frame['sha256']=frame.perf_id.map(hashes)
    frame=frame.drop_duplicates('sha256').reset_index(drop=True)
    seq={}
    for i,r in frame.iterrows():
        with np.load(r.normalized_cache) as z: seq[i]={k:z[k].copy() for k in ['raw','relative','normalized']}
    distances=variability(frame,seq,out); print('Within-work variation completed',flush=True)
    era_tables=[]
    for layer in ['raw','relative','norm']:
        for name in CHANNELS:
            for label in ['era','composer']:
                cols=[f'{layer}_{name}_mean',f'{layer}_{name}_range']
                works=frame.groupby('composition_id').agg({label:'first',**{c:'mean' for c in cols}})
                for group,w in works.groupby(label):
                    era_tables.append(dict(layer=layer,channel=name,label=label,group=group,works=len(w),
                                           signed_mean=float(w[cols[0]].mean()),between_work_sd=float(w[cols[0]].std()),
                                           mean_p95_p05_range=float(w[cols[1]].mean())))
    pd.DataFrame(era_tables).to_csv(out/'era_composer_feature_comparison.csv',index=False,encoding='utf-8-sig')
    probes=leakage_probes(frame,out); print('Metadata probes completed',flush=True)
    retr=retrieval(frame,seq,out); print('Disjoint retrieval completed',flush=True)
    sections=section_retrieval(frame,seq,out)
    proxy=performer_proxy(frame,seq,out); print('Cross-work performer proxy completed',flush=True)
    neighborhoods=cross_work_metadata_comparison(frame,seq,out)
    stability=ranking_stability(frame,seq,out)
    sensitivity=outlier_sensitivity(frame,seq,out)
    paired=paired_comparisons(retr,proxy,out)
    figures(probes,distances,retr,proxy,out)
    provenance=json.loads((out/'provenance.json').read_text(encoding='utf-8')) if (out/'provenance.json').exists() else {}
    stats=dict(quality=quality,review_queue=review,descriptive_pool_normalization=True,approximate_alignment=True,
               alignment_method=provenance.get('alignment','unknown'),automatic_alignment_not_ground_truth=True,
               paired_proxy_comparisons=paired.get('proxy',{}),paired_ablation_comparisons=paired.get('ablation',{}),
               actual_user_preference_labels=False,
               unique_analyzed_performances=len(frame),works=int(frame.composition_id.nunique()),
               composers=int(frame.composer.nunique()),artists=int(frame.artist.nunique()),
               statuses=audit.status.value_counts().to_dict(),
               outlier_sensitivity={metric:bootstrap_work(sensitivity,metric) for metric in ['rank_spearman','top1_unchanged']} if len(sensitivity) else {},
               section_retrieval={metric:bootstrap_work(sections,metric) for metric in ['hit1','mrr','chance_hit1']},
               cross_work_neighborhoods={label:{metric:bootstrap_work(d,metric) for metric in ['top10_same_fraction','pool_same_fraction','score_metadata_spearman']}
                                         for label,d in neighborhoods.groupby('metadata')} if len(neighborhoods) else {},
               ranking_stability={config:{metric:bootstrap_work(d,metric) for metric in ['rank_spearman','top1_unchanged']}
                                  for config,d in stability.groupby('config')} if len(stability) else {},
               retrieval={config:{metric:bootstrap_work(d,metric) for metric in ['hit1','mrr','chance_hit1','chance_mrr']}
                          for config,d in retr.groupby('config')},
               performer_proxy={model:{metric:bootstrap_work(d,metric) for metric in ['hit1','mrr','ndcg']}
                                for model,d in proxy.groupby('model')} if len(proxy) else {})
    (out/'validation_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
