"""Generate Korean comparison/re-extraction reports solely from observed results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(frame):
    if frame.empty: return '해당 조건을 만족하는 관측치가 없습니다.'
    d=frame.copy()
    for col in d:
        if pd.api.types.is_float_dtype(d[col]):
            d[col]=d[col].map(lambda v:f'{v:.5f}' if pd.notna(v) else '-')
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+ '\n'.join(
        '| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in r)+' |' for r in d.itertuples(index=False,name=None))


def interval(ci):
    if not ci or ci.get('mean') is None: return '평가 가능한 작품 없음'
    return f"{ci['mean']:.5f} (95% CI {ci['low']:.5f} ~ {ci['high']:.5f}; {ci['works']}개 작품)"


def benchmark_report(out):
    selected=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    env=json.loads((out/'environment.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(out/'per_performance.csv')
    summary=pd.read_csv(out/'model_summary.csv')
    by_model=summary.set_index('model')
    winner=by_model.loc[selected['selected_model']]
    selection_summary=f"1차 지표인 0.25박 이내 beat 정확도는 DualDTW {by_model.loc['dual_dtw','beat_within_quarter']:.3%}, TheGlueNote {by_model.loc['gluenote','beat_within_quarter']:.3%}다. 선택한 방법의 note reference F1은 {winner.note_f1:.3%}다."
    mae_model=summary.sort_values('beat_mae_seconds').iloc[0].model
    mae_caveat=f"**모든 지표에서 같은 방법이 우세한 것은 아니다.** 큰 오류의 영향을 받는 beat MAE는 {mae_model}가 더 낮다. 선택은 사전 고정한 0.25박 정확도와 paired CI에 따른 것이며, MAE와 큰 오차 사례도 함께 공개한다."
    metric_names={'attempted':'시도한 연주','completed':'완료한 연주','failures':'실행 실패',
        'works':'작품 수','beat_coverage':'Beat 평가 범위 coverage',
        'beat_within_quarter':'0.25박 이내 beat 정확도 (작품 macro)',
        'beat_within_100ms':'100ms 이내 beat 정확도 (작품 macro)',
        'beat_mae_seconds':'Beat MAE (초, 성공/범위 안)',
        'median_performance_p90_seconds':'연주별 beat p90의 중앙값 (초)',
        'beat_catastrophic_fraction':'1박 초과 오류 비율 (범위 안)',
        'median_runtime_seconds':'정렬 runtime 중앙값 (초)',
        'eligible_note_performances':'Note 평가 가능한 연주',
        'note_precision':'Note reference precision','note_recall':'Note reference recall',
        'note_f1':'Note reference F1 (작품 macro)',
        'note_onset_mae_seconds':'대응 note onset MAE (초, 비교 가능한 음)',
        'conflicting_pairs_removed':'충돌로 제외한 pair 수','accepted_fraction':'기존 quality gate 통과율'}
    comparison=summary.set_index('model').T.reset_index(names='지표')
    comparison['지표']=comparison['지표'].map(lambda x:metric_names.get(x,x))
    excluded=pd.read_csv(out/'excluded_inputs.csv')
    ok=frame[frame.status=='ok']
    reference=ok.groupby('model').agg(reference_score_mapping_median=('reference_score_mapping_fraction','median'),
        reference_perf_mapping_median=('reference_performance_mapping_fraction','median'),
        eligible_note_performances=('note_evaluation_eligible','sum'))
    difficult=ok.sort_values('beat_p90_relative',ascending=False).head(20)
    difficult[['key','model','beat_p90_seconds','beat_p90_relative','beat_coverage','accepted']].to_csv(
        out/'difficult_cases.csv',index=False,encoding='utf-8-sig')
    if selected.get('paired_only'):
        benchmark_options='--summarize-only --paired-only'
        scope=f"사용자의 요청에 따라 전수 비교를 중단하고, 두 방법의 결과가 모두 저장된 **{frame.key.nunique()}개 동일 연주 / {frame.work.nunique()}개 작품**을 비교했다. 전체 평가 가능 corpus는 {env['corpus_performances']}개 연주다. 한쪽 결과만 저장된 연주는 주 비교에서 제외하고 unpaired_observations.csv에 보존했다."
        sampling_caveat='**부분 평가의 선택 편향:** 작곡가·작품명 순으로 실행하던 중 중단한 표본이다. 무작위·시대별 층화 표본이 아니며 뒤 순서의 작곡가/작품과 처리 시간이 긴 사례가 덜 포함될 수 있다. 따라서 아래 결과는 이번 관측 subset에서의 운영상 선택이며 ASAP 전체 또는 ATEPP 전체에 대한 우월성 판정이 아니다.'
    else:
        benchmark_options='--workers 2 --torch-threads 4 --glue-backend openvino_gpu'
        scope=f"평가 대상은 ASAP의 aligned flag가 참인 **{env['corpus_performances']}개 연주 / {frame.work.nunique()}개 작품**이다. 전체 대상에 두 방법을 모두 시도했으며 실행 실패도 포함했다."
        sampling_caveat='전체 예정 corpus를 평가했다.'
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for model,d in ok.groupby('model'):
        v=np.sort(d.groupby('work').beat_p90_seconds.mean().to_numpy())
        axes[0].plot(np.maximum(v,1e-5),np.arange(1,len(v)+1)/len(v),label=model)
    axes[0].set_xscale('log'); axes[0].set_xlabel('Work mean of performance p90 beat error (seconds)')
    axes[0].set_ylabel('Cumulative fraction of works'); axes[0].legend()
    axes[1].bar(summary.model,summary.beat_within_quarter,color=['#2563eb','#f97316'])
    axes[1].set_ylim(0,1); axes[1].set_ylabel('Work-macro beat accuracy within 0.25 local beat')
    for i,v in enumerate(summary.beat_within_quarter): axes[1].text(i,v+.01,f'{v:.4f}',ha='center')
    fig.tight_layout(); fig.savefig(out/'alignment_comparison.png',dpi=160); plt.close(fig)
    md=f'''# ASAP: DualDTW와 TheGlueNote 정렬 비교

## 결과와 선택

선택한 정렬기: **{selected['selected_model']}**.

선택 근거: `{selected['selection_reason']}`. 이 선택은 이번 입력 형식·ASAP 관측 결과에 대한 선택이며 ATEPP에서의 우월성을 입증하지 않는다.

{selection_summary}

{mae_caveat}

{scope} 원본 {len(excluded)+env['corpus_performances']}개 중 {len(excluded)}개는 aligned flag 등 입력 조건으로 평가 전 제외했다.

{sampling_caveat}

포함된 작곡가별 비교 연주 수:

{table(frame.groupby('composer').agg(performances=('key','nunique'),works=('work','nunique')).reset_index())}

{table(comparison)}

`beat_within_quarter`: **모든 유효 annotated beat 중 오차가 그 지점의 실제 다음 박 간격의 0.25배 이하인 비율**. 추정 범위 밖의 beat는 오답, 실행 실패한 연주는 0점으로 계산한다. 연주별 비율을 작품 안에서 평균한 뒤 작품별 동일 가중치로 평균한다.
`beat_within_100ms`도 같은 방식이다. MAE/p90은 추정 범위 안의 성공 사례에만 계산하므로 coverage·실패율과 함께 읽어야 한다. `beat_catastrophic_fraction`은 범위 안에서 1박 초과 오차의 비율이며 범위 밖은 coverage에 나타난다.
`note_f1`은 아래 조건을 만족한 note reference 부분집합의 작품별 평균이다. 초 단위 숫자는 seconds, 나머지 비율은 0~1이다.
Note onset MAE는 동일 reference score note에 모델이 연결한 performance note의 시작 시간과 reference performance note의 시작 시간 차이다. 미대응 음은 이 MAE에서 제외되므로 recall/F1과 함께 읽는다.

![정렬 오차 비교](alignment_comparison.png)

## 같은 작품에서의 차이와 불확실성

TheGlueNote − DualDTW, 0.25박 이내 beat 정확도: **{interval(selected['beat_difference_gluenote_minus_dual'])}**.

TheGlueNote − DualDTW, 양쪽 모두 note 평가 가능한 동일 연주의 F1: **{interval(selected['note_difference_gluenote_minus_dual'])}**.

작품 단위 paired bootstrap 2,000회를 사용한다. beat 정확도를 사전 고정한 1차 기준으로 사용한다. 그 CI가 0을 포함하면 note F1의 paired CI를 2차로 사용한다. 둘 다 판별되지 않으면 runtime으로 운영상 선택하며 정확도 우월성을 주장하지 않는다. 작품별 runtime/실패/작곡가 차이는 CSV로 공개한다.

## 공정한 입력과 정렬 실행

- 두 방법에 **동일한 원본 score MIDI와 performance MIDI**에서 읽은 음높이·시작·길이를 제공했다.
- score 위치는 MIDI tick을 4분음표 단위로 바꿨다. score MIDI의 임의 재생 tempo를 표현 tempo로 사용하지 않았다. performance 시간은 초 단위다.
- 정답 beat, 정답 note pair, 정답 기반 anchor를 정렬기에 제공하지 않았다. 예측을 저장한 뒤 reference match 파일을 읽었다.
- 동일 score/performance note에 서로 다른 대응을 출력한 경우, 충돌한 edge를 **모두 제외**했다. 어떤 대응이 맞는지 임의로 고르거나 새로운 대응을 추가하지 않았다. 동일한 pair의 단순 중복은 한 번만 센다. 양쪽 정렬기에 동일하게 적용하며 제거 수는 `conflicting_prediction_pairs_removed`에 기록한다.
- MIDI에는 장식음/성부의 충분한 악보 주석이 없으므로 DualDTW의 `process_ornaments=False`를 사용했다. 이는 **ATEPP에서 사용 가능한 MIDI 입력 조건의 비교**이며, MusicXML 장식음 정보를 추가한 DualDTW 최적 설정의 비교가 아니다.
- 정렬기 출력 note pair로 동일한 tempo map을 구성했다: 같은 악보 onset의 연주 onset 중앙값 → 순증가하는 최장 부분열 → 선형 보간. 정렬기 차이와 후처리 차이가 섞이지 않도록 했다.
- beat 오차는 이 map을 원본 ASAP score beat 위치에서 평가하고 원본 performance beat 시간과 비교했다. `bR` 및 다음 박 간격이 양수가 아닌 위치는 제외했다. 범위 밖 외삽은 하지 않았다.
- 기존 ATEPP 품질 gate는 별도로 기록했다. **benchmark 점수는 gate 통과 여부로 필터링하지 않았다.** gate는 정렬 정확도의 정답 검증이 아니다.
- TheGlueNote는 Parangonar에 포함된 **small 공개 checkpoint**를 사용했다. medium/large checkpoint와 별도 재학습 모델은 이번 비교 대상이 아니다.
- TheGlueNote는 처음 PyTorch CPU로 평가하다가 Intel Iris Xe에서 **OpenVINO FP32**로 동일 network graph의 inference를 가속했다. 가중치/구조/DTW 후처리는 유지했고 FP16 압축·quantization을 하지 않았다. 변환 forward의 최대 절대 차이는 약 1e-5 이하, 평균 약 2e-9였으며 실제 짧은 4개 연주와 8천 음표 규모 1개 연주에서 native PyTorch와 **최종 note pair가 완전히 일치**했다. GPU와 CPU의 부동소수점 계산은 bit-exact 같다는 뜻은 아니다. backend별 기록은 `inference_backend`와 runtime_conditions.csv에 있다. native와 변환 경로를 모두 지원한다.
- 공개 구현의 0/1 pitch-membership 누적 비용 및 pitch별 scalar onset DTW에는 Python 이중 반복문이 있다. 같은 float64 거리식·전체 matrix·누적 재귀식을 Numba로 컴파일하고 원본 traceback을 유지해 가속했다. 1차원 Euclidean 거리는 같은 abs(x-y)로 계산했다. band/pruning/근사 정렬을 추가하지 않았다. 원본 matrix와의 완전 일치 테스트 및 기존 실제 MIDI 예측 pair와의 완전 일치 여부를 검증했다. 이전 예측이 있는 사례의 일치 결과는 `exact_kernel_previous_pairs_equal`이다. **양쪽 runtime은 이 가속 구현 기준**이며 원본 Python loop 속도로 일반화하지 않는다.
- per-file runtime에서 checkpoint 초기 로딩을 제외했다. 첫 호출의 JIT 비용이 남을 수 있으므로 중앙값을 보고했다. 메모리 경합 때문에 CPU worker 4/torch thread 2에서 worker 2/thread 4로 조정해 이어서 실행했다. resource 조건별 표는 [runtime_conditions.csv](runtime_conditions.csv)다. 전체 중앙값은 여러 자원 조건의 기술적 요약이며 통제된 hardware speed benchmark가 아니다. 정확도 선택을 우선하며 속도는 운영 참고치로 해석한다.

## Note reference와 ID 대응

원본 ASAP beat annotation을 독립적인 시간 reference로, 로컬 (n)ASAP `.match`의 note 대응을 note reference로 사용했다. (n)ASAP note reference는 자동 생성·검수 조건을 가진 데이터로 **모든 음표가 독립적으로 수작업 검증된 정답이라는 뜻은 아니다**.

서로 다른 파일의 note ID를 그대로 비교하지 않았다. score 쪽은 pitch·score onset으로, performance 쪽은 pitch·실제 onset으로 파일 내부 identity를 대응시켰다. score-only 좌표 변환(scale/offset)은 score 음표만 이용해 찾았으며 performance 정답 대응을 사용하지 않았다. tolerance는 score 0.002 quarter, performance 0.005초다. 같은 pitch/onset이 중복되어 identity가 모호하면 해당 음표를 제외했다.

`robust_note_alignment=true`, score identity 대응률 ≥95%, performance identity 대응률 ≥98%인 경우만 주 note 평가에 사용했다. 대응된 note domain 안에서 예측 pair와 reference pair의 집합 교집합으로 precision/recall/F1을 계산한다. 따라서 F1은 **평가 가능한 부분집합의 reference 일치도**이며 MIDI에 표현되지 않은 장식음 등을 포함한 전체 악보 정확도가 아니다. 대응률·좌표 변환과 평가 domain pair 수는 per_performance.csv에 남겼다.

{table(reference.reset_index())}

## 해석의 한계

1. **TheGlueNote의 공개 학습 데이터에 (n)ASAP이 포함되어 있다.** 이번 결과는 완전히 독립적인 미학습 작품의 일반화 성능 평가가 아니다. ASAP를 학습 데이터에서 제거한 재학습을 수행하지 않았다.
2. note reference 역시 기존 알고리즘의 영향을 받는다. 동일 reference에 대한 F1이 실제 인간 판단의 우월성과 같다고 보지 않는다. 독립 beat 평가를 우선한 이유다.
3. ATEPP는 자동 전사 MIDI이므로 실제 건반 MIDI인 ASAP와 오류 분포가 다르다. ASAP에서 선택된 방법을 적용하되 ATEPP quality gate와 극단값 검증을 다시 수행해야 한다.
4. 정렬기가 누락음·잘못된 음의 대응을 추정할 수 있어도, 전사에서 잃은 velocity/note-off/pedal의 실제 연주 정보를 복원하는 것은 아니다.

## 큰 오차 사례

{table(difficult[['key','model','beat_p90_seconds','beat_p90_relative','beat_coverage','accepted']].head(12))}

전체 결과: [per_performance.csv](per_performance.csv), [작곡가별](by_composer.csv), [제외 목록](excluded_inputs.csv), [선택 결과](selection.json). 개별 예측과 beat 오차 배열은 predictions 폴더에 저장한다.

## 연구·공식 구현 출처

- [Parangonar 공식 구현](https://github.com/sildater/parangonar): DualDTWNoteMatcher / TheGlueNoteMatcher.
- [TheGlueNote 논문, Peter & Widmer, ISMIR 2024](https://arxiv.org/abs/2408.04309).
- [TheGlueNote 코드·공개 가중치·(n)ASAP 학습 안내](https://github.com/sildater/thegluenote).
- [Automatic Note-Level Score-to-Performance Alignments in the ASAP Dataset, 2023](https://carloscancinochacon.com/documents/peer_reviewed/PeterEtAl-TISMIR-2023.pdf).

## 재현

저장소 루트에서 실행:

```powershell
.venv/Scripts/python.exe classicfy-ai/scripts/benchmark_research_alignment.py {benchmark_options}
.venv/Scripts/python.exe classicfy-ai/scripts/write_research_alignment_reports.py --benchmark-only
```

results.jsonl에 완료한 작업을 추가 저장하므로 중단 후 같은 명령으로 이어서 실행할 수 있다. 이전 실패도 보존하며, 수정한 설정을 비교하려면 별도의 output 경로를 사용한다.

이번처럼 저장된 paired subset만 비교하려면 benchmark에 `--summarize-only --paired-only`를 사용한다. reference를 다시 읽어 동일 평가 로직으로 갱신하려면 `--refresh-reference --paired-only`를 사용한다. 남은 ASAP inference를 실행하지 않고 ATEPP 단계로 넘어가는 runner 옵션은 `--skip-benchmark --paired-only`다.

이미 reference 오차가 저장된 이번 subset을 그대로 사용해 바로 다음 단계로 진행한 옵션은 `--skip-benchmark --skip-reference-refresh --paired-only`다. 정렬과 정답 오차를 다시 계산하지 않고 동일 paired subset을 집계한다.

환경: `{json.dumps(env['versions'],ensure_ascii=False)}`. NumPy에서 제거된 `row_stack` 이름을 동일 기능인 `vstack`에 연결한 호환 shim과 torch inference_mode를 사용했다. 알고리즘·가중치는 변경하지 않았다.

Intel 가속 재현 시 requirements-alignment-openvino.txt를 추가 설치하고 `probe_gluenote_acceleration.py`로 FP32 IR을 생성/검증한다. Intel GPU가 없는 환경은 `--glue-backend torch`를 사용한다. [OpenVINO 공식 precision 제어](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html), [PyTorch graph 변환](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-to-ir.html).
'''
    (out/'README.md').write_text(md,encoding='utf-8')


def atepp_report(out,benchmark,old):
    selection=json.loads((benchmark/'selection.json').read_text(encoding='utf-8'))
    stats=json.loads((out/'validation_stats.json').read_text(encoding='utf-8'))
    oldstats=json.loads((old/'validation_stats.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(out/'performance_summary.csv',dtype={'perf_id':str})
    audit=pd.read_csv(out/'midi_audit.csv',dtype={'perf_id':str})
    quality=pd.read_csv(out/'feature_quality.csv')
    qt=quality.groupby('channel').agg(valid_cells=('valid','sum'),missing_cells=('missing','sum'),
        over_6=('over_6','sum'),over_10=('over_10','sum'),max_abs=('maximum_abs','max')).reset_index()
    retrieval=pd.DataFrame([dict(config=c,Top1=m['hit1']['mean'],low=m['hit1']['low'],high=m['hit1']['high'],
                                  chance=m['chance_hit1']['mean'],MRR=m['mrr']['mean'],works=m['hit1']['works'])
                            for c,m in stats['retrieval'].items()])
    proxy=pd.DataFrame([dict(model=c,Top1=m['hit1']['mean'],low=m['hit1']['low'],high=m['hit1']['high'],
                              MRR=m['mrr']['mean'],works=m['hit1']['works'])
                        for c,m in stats['performer_proxy'].items()])
    comparison=pd.DataFrame([dict(alignment=s.get('alignment_method','heuristic'),
        normalized=s['quality']['normalized_performances'],works=s['works'],
        invalid_issues=s['quality']['invalid_feature_issues'],
        identity_Top1=s['retrieval']['all']['hit1']['mean'],
        preference_proxy_Top1=s['performer_proxy']['feature']['hit1']['mean'],
        outlier_top1_unchanged=s['outlier_sensitivity']['top1_unchanged']['mean']) for s in [oldstats,stats]])
    oldcatalog=pd.read_csv(old/'performance_summary.csv',dtype={'perf_id':str})
    common=summary.merge(oldcatalog,on='perf_id',suffixes=('_new','_old'))
    deltas=[]
    for channel in ('tempo','rubato','dynamics','articulation','pedal_depth','pedal_down_ratio','pedal_changes'):
        for layer in ('raw','relative','norm'):
            column=f'{layer}_{channel}_mean'
            a=common[column+'_new'].to_numpy(); b=common[column+'_old'].to_numpy(); valid=np.isfinite(a)&np.isfinite(b)
            if valid.any(): deltas.append(dict(layer=layer,channel=channel,common_performances=int(valid.sum()),
                median_absolute_summary_difference=float(np.median(np.abs(a[valid]-b[valid])))))
    pd.DataFrame(deltas).to_csv(out/'previous_alignment_feature_differences.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(perf_id=r.perf_id,sha256=__import__('hashlib').sha256(Path(r.normalized_cache).read_bytes()).hexdigest(),
                       path=r.normalized_cache) for r in summary.itertuples()]).to_csv(
        out/'normalized_file_manifest.csv',index=False,encoding='utf-8-sig')
    stats_table=pd.DataFrame([dict(test=k,mean=v['mean'],low=v['low'],high=v['high'],works=v['works'])
        for k,v in stats['outlier_sensitivity'].items()])
    def ci_table(values):
        return pd.DataFrame([dict(comparison=k,mean=v['mean'],low=v['low'],high=v['high'],works=v['works'])
                             for k,v in values.items()])
    paired=ci_table(stats['paired_proxy_comparisons'])
    ablation=ci_table(stats['paired_ablation_comparisons'])
    variance=pd.read_csv(out/'variance_decomposition.csv')
    preservation=pd.read_csv(out/'pair_difference_preservation.csv')
    preservation_summary=pd.DataFrame([dict(checked_performance_pairs=len(preservation),
        works=preservation.composition_id.nunique(),
        maximum_pair_residual_error=preservation.max_pair_residual_error.max(),
        pair_errors_above_1e_9=int((preservation.max_pair_residual_error>1e-9).sum()))])
    feature_ci=stats['paired_proxy_comparisons']['feature_minus_composer_era']
    if feature_ci['low']>0:
        proxy_judgment='동일 연주자 선호라는 대리 과제에서는 작곡가·시대 metadata보다 추가 구별력이 관측됐다.'
    elif feature_ci['high']<0:
        proxy_judgment='동일 연주자 선호라는 대리 과제에서는 작곡가·시대 metadata보다 성능이 낮았다.'
    else:
        proxy_judgment='동일 연주자 선호라는 대리 과제에서 작곡가·시대 metadata보다 낫다는 근거는 불확실하다.'
    metadata_neighbors=pd.read_csv(out/'cross_work_metadata_neighborhoods.csv')
    neighborhood_summary=metadata_neighbors.groupby('metadata').agg(
        queries=('perf_id','count'),
        mean_top10_same_fraction=('top10_same_fraction','mean'),
        mean_pool_same_fraction=('pool_same_fraction','mean'),
        median_score_metadata_spearman=('score_metadata_spearman','median')).reset_index()
    md=f'''# ATEPP: 研究 기반 정렬 후 피처 재추출·검증

## 결과

ASAP 비교에서 선택한 **{selection['selected_model']}**을 ATEPP 악보–연주 MIDI에 적용했다. [선택 근거와 독립 beat 검증](../alignment_comparison/README.md).

원본 {len(audit)}개 연주를 감사했다. 정렬 완료(quality gate 통과) **{int((audit.status=='aligned').sum())}개**, 동일 작품 공통 패턴 제거·정규화 **{len(summary)}개 / {summary.composition_id.nunique()}개 작품**이다. exact duplicate MIDI를 제외한 분석 데이터는 {stats['unique_analyzed_performances']}개다. **수치·shape·mask·domain 오류 {stats['quality']['invalid_feature_issues']}건**이다. 이것은 정렬/전사 오차가 없다는 뜻은 아니다.

{table(audit.status.value_counts().rename_axis('상태').reset_index(name='연주수'))}

정렬기는 공식 구현을 사용했으며 임의 pitch-class 휴리스틱을 fallback으로 섞지 않았다. DualDTW는 학습 가중치를 사용하지 않는 알고리즘이다. 비교 대상 TheGlueNote에는 공개 small checkpoint를 사용했다. 실패·품질 기준 미달 사례는 제외 기록을 남겼다. 악보가 없는 연주는 악보 의존 피처를 생성하지 않는다.

## 파이프라인과 저장 파일

`score MIDI + performance MIDI → 선택 정렬기 → note pair → 단조 tempo map → quarter grid → 기존 5개 피처군 → 작품의 공통 패턴 제거 → 정규화 → 동일 검증`

- **5개 피처군**: Tempo, Rubato, Dynamics, Articulation, Pedaling. Pedaling은 depth/down-ratio/changes로 나뉘므로 저장 배열은 7채널이다.
- note 대응에서 같은 악보 onset의 연주 onset 중앙값을 취하고 순증가 부분열·선형 보간으로 시간 지도를 만든다. 입력 MIDI에는 XML의 장식음 정보가 충분하지 않으므로 Articulation의 장식음 제외 한계는 남아 있다.
- 이전과 동일한 quality gate를 사용했다: score 대응률 ≥0.70, performance 대응률 ≥0.60, 범위 coverage ≥0.90, 단조 anchor 유지율 ≥0.90, note 수 비율 0.65~1.6, 자체 map timing p90 ≤0.5박, 16개 이상 interval, 극단 박 길이 비율 ≤2%. 대응률은 **reference precision/recall이 아니다**. map으로 자신을 평가하는 잔차 역시 독립 정확도가 아니다.
- 정렬 전체는 `ATEPP_dataset/alignments/{selection['selected_model']}/<perf_id>.npz`에 저장하며 gate 미달이더라도 정렬기가 출력한 대응은 남긴다.
- raw: `ATEPP_dataset/features/{selection['selected_model']}/raw/<perf_id>.npz`.
- normalized: `ATEPP_dataset/features/{selection['selected_model']}/normalized/<perf_id>.npz`.
- normalized 배열: `raw`, `relative`, `normalized`, `mask`: (T,7), `score_beats`: (T+1). 결측은 NaN+mask로 표현한다. CC64 미제공을 무페달로 해석하지 않는다.
- 기존 휴리스틱 결과와 새 결과를 별도 경로에 보존했다. manifest에는 새 normalized 파일별 SHA256을 기록했다.

## 공통 요소 제거·정규화

같은 composition_id 및 같은 악보 MIDI의 연주를 묶고, 모두 관측되는 공통 악보 범위만 사용한다. 단일 연주 작품은 공통 패턴 추정에서 제외한다. Tempo의 score reference는 quarter당 0.5초이며 위치별 공통 패턴을 뺄 때 상수 reference가 상쇄된다. Rubato는 기존 median 기준 로직을 유지한다. D/A와 T/R은 기존 robust scale, pedal 각 채널은 SD 기준을 유지하며 clipping을 하지 않는다.

후보 pool 전체에서 위치별 중앙값과 작품별 scale을 추정한다. 따라서 **이 검증은 후보 집합에 대한 기술적/transductive 분석**이다. 새로운 작품의 완전히 독립적인 train/test 일반화 성능은 별도 실험이 필요하다.

공통/해석 분산과 차이 보존:

{table(variance)}

{table(preservation_summary)}

공통 패턴을 뺀 뒤에도 연주 간 원래 차이가 유지되는지 검사했다. 전체 pair 검사는 [pair_difference_preservation.csv](pair_difference_preservation.csv)에 있다.

## 수치 무결성·극단값

유효 feature cell {stats['quality']['valid_cells']:,}개를 검사했다. 극단값은 삭제하지 않고 검수 queue와 별도 순위 민감도 분석에 사용했다.

{table(qt)}

검수 queue: **{stats['review_queue']['flagged_performances']}개 연주 / {stats['review_queue']['flagged_channel_rows']}개 채널 행**. max |x|>20 또는 |x|>6의 비율>2%인 채널을 표시했으며 오류 확정 label은 아니다. 작은 작품별 scale, 실제 긴 pause, 전사/정렬 오류가 극단값의 원인일 수 있다. 자동 정렬기 교체만으로 모든 극단값이 정상화된다고 가정하지 않는다.

`|normalized|>6` cell 제외 시 추천 순위 변화:

{table(stats_table)}

수치 예외: [invalid_feature_values.csv](invalid_feature_values.csv). 극단값 사례: [extreme_beat_examples.csv](extreme_beat_examples.csv), [feature_review_queue.csv](feature_review_queue.csv).

## 시대·작곡가 효과와 metadata와의 차이

작품 단위로 분리한 5-fold composer/era probe 결과:

{table(pd.read_csv(out/'metadata_leakage_probes.csv'))}

raw → 공통 패턴 제거 → 정규화 각 단계의 metadata 예측성을 비교한다. 작품별 permutation effect size/BH 보정은 [metadata_effect_sizes.csv](metadata_effect_sizes.csv), 시대/작곡가별 평균·범위는 [era_composer_feature_comparison.csv](era_composer_feature_comparison.csv)다. era는 작곡가 기반의 대략적인 label이며 작품별 작곡 연대 정답이 아니다.

다른 작품의 feature 최근접 이웃이 metadata 그룹을 얼마나 유지하는지, metadata 점수와 feature 순위의 Spearman은 [cross_work_metadata_neighborhoods.csv](cross_work_metadata_neighborhoods.csv)에 기록한다. 동일 작품 안에서는 작곡가/시대가 같기 때문에 이 metadata만으로 연주를 구별할 수 없다.

{table(neighborhood_summary)}

위 요약은 query 수에 가중된 기술적 통계다. 같은 label의 후보가 원래 얼마나 많은지(pool 비율)와 함께 읽으며, 순위가 metadata와 다르다는 사실만으로 사용자 취향에 더 적합하다고 결론짓지 않는다.

## 동일 작품의 연주 차이와 피처 ablation

같은 연주의 겹치지 않는 beat 절반 두 집합으로 self-retrieval을 수행한다. 후보가 공통으로 관측한 beat를 사용하며 작품 단위 CI를 구한다. **연주 identity 구별력이며 취향 정확도가 아니다.**

{table(retrieval)}

피처 제거 전후의 동일 작품 paired 차이:

{table(ablation)}

시간상 전반부→후반부 retrieval: `{json.dumps(stats['section_retrieval'],ensure_ascii=False)}`.

## 사용자 취향 추천을 위한 대리 검증

대상 작품을 포함하지 않는 3~5개 다른 작품의 동일 연주자 연주로 profile을 구성하고 대상 작품에서 해당 연주자의 연주를 찾는다. feature와 composer/era metadata, artist metadata를 비교한다. **동일 artist 선호를 가정한 proxy이며 실제 사용자 좋아요 정확도나 미관측 연주자 일반화가 아니다.**

{table(proxy)}

동일 query·동일 작품에서 metadata 기준과의 paired 비교:

{table(paired)}

profile beat dropout에 대한 순위 안정성: `{json.dumps(stats['ranking_stability'],ensure_ascii=False)}`.

페달의 세 하위 채널은 하나의 피처군으로 동일 가중치를 갖는다. 후보 결측이 거리를 유리하게 만들지 않도록 query와 후보의 공통 완전한 피처군 조건을 유지한다.

## 이전 정렬과의 비교

{table(comparison)}

정렬 gate 통과 작품/연주가 달라지므로 두 전체 cohort의 추천 지표 차이를 정렬기의 인과적 개선으로 해석하지 않는다. 공통 {len(common)}개 연주의 피처 요약 차이는 [previous_alignment_feature_differences.csv](previous_alignment_feature_differences.csv)에 있다. 이 차이에는 각 정렬의 beat 구간 및 normalization pool 변화도 포함된다.

## 활용 판단

동일 작품에서 연주 identity Top1은 **{stats['retrieval']['all']['hit1']['mean']:.3%}**, 후보 수에 따른 무작위 기준은 **{stats['retrieval']['all']['chance_hit1']['mean']:.3%}**다. 이는 해석 차이에 대한 구별력의 기술적 지표다.

{proxy_judgment} Feature − composer/era Top1 차이는 **{interval(feature_ci)}**다. 사용자 취향을 동일 연주자 선호로 대체한 결과이므로 실제 사용자 좋아요의 정확도로 해석하지 않는다. artist metadata와의 별도 비교도 위 표에 제공한다.

극단값 cell을 제외했을 때 Top1이 바뀐 비율은 **{1-stats['outlier_sensitivity']['top1_unchanged']['mean']:.3%}**다. 따라서 수치 무결성 통과와 추천 순위의 신뢰성을 구분해야 한다. 검수 queue를 확인하고, 검수 전후 순위 변화와 피처 ablation 결과를 바탕으로 사용할 피처를 정해야 한다.

현재 결과는 실제 사용자 평가에 투입할 후보 피처와 검수 대상을 결정하는 근거다. 자동 전사 정보 손실·자동 정렬·극단값 민감도·실제 사용자 label 부재 때문에 최종 취향 추천 정확도를 확정하지 않는다. 다음 검증은 사용자별 선호 연주로 profile을 만든 뒤, profile에 넣지 않은 동일 작품의 연주에 대한 쌍별 선호/순위를 예측하고 metadata 기준과 비교하는 것이다. 평가 대상 사용자·작품을 분리하고 학습/평가 경계를 넘는 공통 패턴 추정을 막아야 한다.

## 재현

저장소 루트에서:

```powershell
.venv/Scripts/python.exe classicfy-ai/scripts/extend_atepp.py --aligner selected --output classicfy-ai/analysis/atepp_research --workers 2 --resume
.venv/Scripts/python.exe classicfy-ai/scripts/validate_atepp_recommendation.py --output classicfy-ai/analysis/atepp_research
.venv/Scripts/python.exe classicfy-ai/scripts/write_research_alignment_reports.py
```

전체 순차 실행은 `run_research_alignment_pipeline.py`로 제공한다. `selection.json`에서 정렬기를 읽으므로 선택 결과가 바뀌면 후속 단계에도 반영된다. 기존 피처 수식과 기존 검증 함수를 재사용하며 동일한 검증 조건을 유지한다.
'''
    (out/'README.md').write_text(md.replace('研究','연구'),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--benchmark',type=Path,default=Path('classicfy-ai/analysis/alignment_comparison'))
    p.add_argument('--atepp',type=Path,default=Path('classicfy-ai/analysis/atepp_research'))
    p.add_argument('--old',type=Path,default=Path('classicfy-ai/analysis/atepp'))
    p.add_argument('--benchmark-only',action='store_true'); args=p.parse_args()
    benchmark_report(args.benchmark)
    if not args.benchmark_only: atepp_report(args.atepp,args.benchmark,args.old)


if __name__=='__main__': main()
