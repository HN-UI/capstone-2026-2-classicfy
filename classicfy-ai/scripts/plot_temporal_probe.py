"""Readable figures for frozen-embedding four-quarter trajectory probes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

from validate_tempo import setup_style, plt
from analyze_temporal_failures import sha


ARMS=("cnn64","bilstm64","transformer64")
NAMES={"flat":"평평한 예측", "train_mean":"학습 작품의 평균 흐름", "summary14":"기존 14D 요약 + 선형 예측",
       "pca64":"단순 PCA 64D + 선형 예측", "raw_reference":"원본에서 직접 계산 (참조)",
       "cnn64":"1D CNN", "bilstm64":"BiLSTM", "transformer64":"Transformer"}
STAGE_NAMES={"initial":"학습 전", "best_extended":"학습 후"}
FEATURES=("Tempo","Rubato","Dynamics","Articulation","Pedaling")
COLORS={"cnn64":"#327BA5","bilstm64":"#27856A","transformer64":"#D07539", "summary14":"#8A719E", "pca64":"#677B8D"}
QUARTERS=("1–16 beat","17–32 beat","33–48 beat","49–64 beat")


def method_order():
    return [(a,"baseline") for a in ("flat","train_mean","summary14","pca64")]+[(a,s) for a in ARMS for s in ("initial","best_extended")]+[("raw_reference","baseline")]


def label(arm,stage):
    return NAMES[arm]+(" · "+STAGE_NAMES[stage] if stage in STAGE_NAMES else "")


def render(out):
    out=Path(out)
    setup_style()
    summary=pd.read_csv(out/"summary.csv")
    channels=pd.read_csv(out/"channel_metrics.csv")
    examples=pd.read_csv(out/"example_quarters.csv")
    protocol=json.loads((out/"protocol.json").read_text())
    manifest,violations=[],[]

    def save(fig,name):
        canvas=FigureCanvasAgg(fig);canvas.draw();renderer=canvas.get_renderer()
        for text in fig.findobj(match=lambda item:isinstance(item,Text)):
            if not text.get_visible() or not text.get_text().strip():continue
            box=text.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
                violations.append(dict(figure=name,text=text.get_text(),bounds=list(box.extents)))
        for extension in ('png','svg'):
            path=out/f"{name}.{extension}";fig.savefig(path,dpi=150)
            manifest.append(dict(file=path.name,sha256=sha(path)))
        plt.close(fig)

    def header(fig,title,subtitle,note):
        fig.suptitle(title,x=.06,y=.975,ha='left',fontsize=23,fontweight='bold')
        fig.text(.06,.925,subtitle,fontsize=12,va='top',color='#56616A')
        fig.text(.06,.025,note,fontsize=10.5,va='bottom',color='#56616A',linespacing=1.5)

    current=examples[(examples.split=='validation')&(examples.example==2)&(examples.channel==2)&(examples.arm=='flat')].sort_values('quarter')
    fig,ax=plt.subplots(figsize=(13,7))
    ax.plot(range(4),current.target,color='#303B42',marker='o',lw=3,label='정답: 실제 강약의 네 구간 평균 − 전체 평균')
    ax.axhline(0,color='#929AA2',ls='--',lw=2,label='기준: 변화 없이 평평하게 예측')
    ax.set_xticks(range(4),QUARTERS);ax.set_ylabel('구간 평균 강약 − 64-beat 전체 평균');ax.grid(alpha=.25);ax.legend(fontsize=10,loc='best')
    first=current.iloc[0]
    header(fig,'임베딩만 보고 실제 연주의 앞·중간·뒤 변화를 읽을 수 있나?',
           '64 beat를 16 beat씩 네 구간으로 나눔 → 각 평균에서 전체 평균을 뺌 → 고정 임베딩에서 작은 선형 예측기로 읽기',
           f"고정 검증 사례: {first['piece']} / {Path(first['key']).stem} / 시작 beat {first['start']}\n평균 수준만 잘 맞히는 효과를 제거합니다. 가려진 값·미래 값 예측이 아니며, 16 beat 내부의 세밀한 움직임까지 평가하지 않습니다.")
    fig.subplots_adjust(left=.12,right=.96,top=.82,bottom=.18)
    save(fig,'00_probe_task')

    fig,axes=plt.subplots(1,2,figsize=(19,10))
    for ax,split in zip(axes,('validation','test')):
        values_for_limit=[]
        for i,(arm,stage) in enumerate(method_order()):
            values=summary[(summary.split==split)&(summary.arm==arm)&(summary.stage==stage)].dynamics_score_percent.to_numpy()
            mean=values.mean();values_for_limit.extend(values)
            color='#ABB1B8' if stage=='initial' else COLORS.get(arm,'#78818A')
            ax.barh(i,mean,height=.7,color=color,alpha=.85)
            ax.errorbar(mean,i,xerr=[[mean-values.min()],[values.max()-mean]],fmt='none',color='#263641',capsize=3)
            text_x=values.max()+1.3 if mean>=0 else values.min()-1.3
            ax.text(text_x,i,f'{mean:.1f}%',va='center',ha='left' if mean>=0 else 'right',fontsize=10)
        ax.set_yticks(range(len(method_order())),[label(*m) for m in method_order()],fontsize=9.5)
        ax.invert_yaxis();ax.axvline(0,color='#515B63',ls='--');ax.grid(axis='x',alpha=.25)
        ax.set_xlim(min(-12,min(values_for_limit)-16),113)
        support=next(s for s in protocol['support'] if s['split']==split)
        ax.set_title(f"{'검증' if split=='validation' else '기존 test'} · {support['works']}작품 / {support['windows']}구간",fontsize=16,pad=15)
        ax.set_xlabel('평평한 예측 대비 강약 변화 오차 감소율 (%) ↑')
    header(fig,'강약이 언제 커지고 작아지는지 임베딩에서 읽히는가?',
           '주평가: Dynamics의 네 구간 변화 | 인코더 고정 | 예측기·PCA·규제 강도는 학습 작품에서만 학습·선택',
           '0%=평평한 예측과 같은 오차, 음수=더 나쁨, 100%=네 구간 변화가 정확히 일치. 검색 정확도나 사람 평가 비율이 아닙니다.\n신경망 막대=3 model seed 평균, 선=최소~최대(신뢰구간 아님). 원본 직접 계산은 정답 생성의 참조이며 압축 표현의 성능이 아닙니다.')
    fig.subplots_adjust(left=.22,right=.95,top=.84,bottom=.14,wspace=.75)
    save(fig,'01_dynamics_comparison')

    heat_methods=[('summary14','baseline'),('pca64','baseline')]+[(a,s) for a in ARMS for s in ('initial','best_extended')]
    matrices=[]
    for split in ('validation','test'):
        matrix=[]
        for arm,stage in heat_methods:
            data=channels[(channels.split==split)&(channels.arm==arm)&(channels.stage==stage)]
            groups=((0,),(1,),(2,),(3,),(4,5,6))
            matrix.append([data[data.channel.isin(group)].score_percent.mean() for group in groups])
        matrices.append(np.array(matrix))
    lower=min(-10,float(min(m.min() for m in matrices)))
    fig,axes=plt.subplots(1,2,figsize=(19,9))
    for ax,split,matrix in zip(axes,('validation','test'),matrices):
        im=ax.imshow(matrix,vmin=lower,vmax=100,cmap='RdYlGn',aspect='auto')
        ax.set_yticks(range(len(heat_methods)),[label(*m) for m in heat_methods],fontsize=10)
        ax.set_xticks(range(5),FEATURES,rotation=15,fontsize=11)
        for i in range(len(heat_methods)):
            for j in range(5):
                ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=11,color='white' if matrix[i,j]>75 or matrix[i,j]<lower+10 else '#26343D')
        ax.set_title('검증' if split=='validation' else '기존 test',fontsize=16,pad=15)
    cbar=fig.colorbar(im,ax=axes,shrink=.7,pad=.02,label='평평한 예측 대비 변화 오차 감소율 (%)')
    header(fig,'강약 이외의 연주 특징에서도 시간 변화가 읽히는가?',
           '동일한 네 구간 변화 과제 | 각 칸=작품 동일 비중으로 계산한 채널 점수의 3 seed 평균',
           'Pedaling은 깊이·사용 비율·전환 3채널 점수의 평균입니다. 각 채널은 해당 평가 split의 평평한 예측 오차를 기준으로 합니다.\nPCA64는 학습 작품의 분산을 보존하는 단순 압축입니다. 예측기는 모든 표현에 동일한 선형 ridge 절차를 적용했습니다.')
    fig.subplots_adjust(left=.21,right=.86,top=.84,bottom=.17,wspace=.75)
    cbar.ax.set_position([.91,.24,.018,.52])
    save(fig,'02_feature_comparison')

    effects=[]
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    for row,split in enumerate(('validation','test')):
        for col,(metric,title) in enumerate((('dynamics_score_percent','강약'),('macro_score_percent','5 feature 평균'))):
            ax=axes[row,col]
            for i,arm in enumerate(ARMS):
                data=summary[(summary.split==split)&(summary.arm==arm)]
                wide=data.pivot(index='seed',columns='stage',values=metric)
                difference=wide.best_extended-wide.initial
                mean=difference.mean()
                ax.errorbar(i,mean,yerr=[[mean-difference.min()],[difference.max()-mean]],fmt='o',color=COLORS[arm],capsize=5,ms=8)
                ax.annotate(f'{mean:+.1f} pp',(i,difference.max()),xytext=(0,10),textcoords='offset points',ha='center',fontsize=11)
                effects.append(dict(split=split,arm=arm,metric=metric,mean_pp=mean,min_pp=difference.min(),max_pp=difference.max(),positive_seeds=int((difference>0).sum())))
            ax.axhline(0,color='#515B63',ls='--');ax.set_xticks(range(3),[NAMES[a] for a in ARMS]);ax.grid(axis='y',alpha=.25)
            ax.set_title(('검증' if split=='validation' else '기존 test')+' · '+title,fontsize=15)
            ax.set_ylabel('학습 후 − 학습 전 (퍼센트포인트)');ax.margins(x=.2,y=.3)
    header(fig,'복원 학습이 시간 정보를 더 잘 읽을 수 있게 만들었는가?',
           '같은 architecture·같은 model seed의 학습 후 점수 − 학습 전 점수 | 각 표현의 예측기 규제 강도는 train-only CV로 별도 선택',
           '점=3 seed의 짝지은 차이 평균, 선=최소~최대이며 신뢰구간이 아닙니다. 양수는 이 선형 읽기 과제의 개선입니다.\n음수는 이 시간 범위의 정보가 선형 예측기로 덜 읽힌다는 뜻이며, 모든 정보가 사라졌거나 추천이 나빠졌다는 결론은 아닙니다.')
    fig.subplots_adjust(left=.10,right=.96,top=.84,bottom=.14,hspace=.55,wspace=.32)
    save(fig,'03_training_effect')
    pd.DataFrame(effects).to_csv(out/'training_effects.csv',index=False)

    fig,axes=plt.subplots(3,2,figsize=(18,14))
    for col,split in enumerate(('validation','test')):
        for row,number in enumerate((1,2,3)):
            ax=axes[row,col]
            data=examples[(examples.split==split)&(examples.example==number)&(examples.channel==2)]
            truth=data[data.arm=='flat'].sort_values('quarter')
            ax.plot(range(4),truth.target,color='#27343C',lw=3,marker='o',label='실제 네 구간 변화')
            ax.axhline(0,color='#A8AFB5',ls=':',lw=1)
            for arm in ('summary14','pca64',*ARMS):
                example_seed=-1 if arm in ('summary14','pca64') else 20261006
                prediction=data[(data.arm==arm)&(data.seed==example_seed)]
                if arm in ARMS:prediction=prediction[prediction.stage=='best_extended']
                prediction=prediction.sort_values('quarter')
                ax.plot(range(4),prediction.prediction,marker='o',lw=1.5,alpha=.85,color=COLORS[arm],label=NAMES[arm])
            first=truth.iloc[0]
            work=first['piece'].replace('/midi_score.mid','').replace('Piano_Sonatas/','').replace('Transcendental_Etudes/','')
            ax.set_title(f"{'검증' if split=='validation' else '기존 test'} 사례 {number} · {work}\n{Path(first['key']).stem} / 시작 beat {first['start']}",fontsize=12,pad=10)
            ax.set_xticks(range(4),['앞 1/4','2/4','3/4','뒤 1/4'],fontsize=10);ax.grid(alpha=.25)
            ax.set_ylabel('강약 구간 평균 − 전체 평균',fontsize=11)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.07),ncol=3,fontsize=10)
    header(fig,'실제 구간의 강약 전개를 얼마나 되살려 읽었는가?',
           '작품 이름순 첫·중간·마지막 작품 → 중앙 연주 → 중앙 선택 구간 | 결과를 보고 좋은 사례를 고르지 않음',
           '신경망 예시는 model seed 20261006의 학습 후 표현과 해당 train-only 예측기입니다. 전체 점수는 모든 구간·세 seed로 계산합니다.\n검정=실제 네 구간 변화. 각 선은 고정 임베딩에 붙인 선형 예측기의 출력이며, 기존 오토인코더 복원기의 출력이 아닙니다.')
    fig.subplots_adjust(left=.09,right=.96,top=.86,bottom=.16,hspace=.55,wspace=.26)
    save(fig,'04_fixed_examples')

    (out/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'layout_audit.json').write_text(json.dumps(dict(status='passed' if not violations else 'failed',figures=5,violations=violations,
        limitation='Canvas-bound checks complement visual review; not all text overlaps are detected.'),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'report_generation.json').write_text(json.dumps(dict(plot_source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(out/'protocol.json'),summary_sha256=sha(out/'summary.csv')),ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(out,summary,protocol,effects)
    if violations:raise ValueError('Figure text is clipped: see layout_audit.json')


def write_report(out,summary,protocol,effects):
    lines=['# 2단계: 고정 임베딩에서 실제 연주의 시간 변화 읽기','',
        '**질문: 복원 학습 후 표현은 기존 요약·학습 전 표현보다 실제 구간의 앞·중간·뒤 변화를 더 잘 읽게 하는가?**','',
        '저장된 1D CNN·BiLSTM·Transformer 인코더를 고정하고, 각 표현에 작은 선형 ridge 예측기만 학습했다. 인코더를 재학습하거나 checkpoint를 다시 선택하지 않았다.','',
        '## 평가 문제','',
        '64 beat를 16 beat씩 네 구간으로 나눈다. 채널별 네 평균에서 해당 64-beat 전체 평균을 빼고, 이 네 변화값을 임베딩만으로 예측한다. 전체 수준을 맞히는 효과를 제거한다.',
        '주평가는 **Dynamics**이고, 같은 절차로 7채널 전체와 5 feature 동일 비중 평균도 확인한다. 실제 관측 feature를 사용하며 인위적인 순서 변경이나 가림은 없다.','',
        '![평가 문제](00_probe_task.png)','',
        '## 결과','',
        '**점수 = 100 × (1 − 예측 MSE / 평평한 0 예측 MSE).** 0%는 평평하게 예측한 것과 같은 오차, 음수는 더 나쁨, 100%는 네 구간 변화가 정확히 일치함을 뜻한다. 사람 평가 비율이나 검색 정확도가 아니다.',
        '각 MSE는 구간→연주→작품 순으로 평균해 작품을 동일 비중으로 둔다. 채널별 점수를 계산한 뒤 Pedaling 세 채널을 한 그룹으로 묶어 5 feature를 동일 비중으로 평균한다.','',
        '| 표현 | 검증 강약 | 기존 test 강약 | 검증 5 feature | 기존 test 5 feature |','|---|---:|---:|---:|---:|']
    for arm,stage in method_order():
        scores=[]
        for metric in ('dynamics_score_percent','macro_score_percent'):
            for split in ('validation','test'):
                data=summary[(summary.arm==arm)&(summary.stage==stage)&(summary.split==split)][metric]
                scores.append(f'{data.mean():.2f}%'+(f' ({data.min():.2f}~{data.max():.2f})' if len(data)>1 else ''))
        lines.append(f"| {label(arm,stage)} | {' | '.join(scores)} |")
    lines += ['', '괄호는 세 model seed의 최소~최대이며 신뢰구간이 아니다. 원본 직접 계산은 정답 생성의 참조값으로, 학습되거나 압축된 표현이 아니다.','',
        '![강약 주평가](01_dynamics_comparison.png)','', '## 학습 전후의 차이','',
        '| 모델 | 검증 강약 변화 | 기존 test 강약 변화 | 기존 test 5 feature 변화 |','|---|---:|---:|---:|']
    for arm in ARMS:
        vals=[]
        for split,metric in (('validation','dynamics_score_percent'),('test','dynamics_score_percent'),('test','macro_score_percent')):
            r=next(e for e in effects if e['arm']==arm and e['split']==split and e['metric']==metric)
            vals.append(f"{r['mean_pp']:+.2f} pp (상승 {r['positive_seeds']}/3 seed)")
        lines.append(f"| {NAMES[arm]} | {' | '.join(vals)} |")
    lines += ['', '양수는 해당 시간 특징이 선형 예측기로 더 잘 읽힌다는 근거다. PCA64와도 비교해 단순한 분산 보존 압축 대비 추가 가치를 확인한다. 음수는 이 과제에서의 악화이며 모든 시간 정보가 사라졌다는 뜻은 아니다.','',
        '### 이번 결과의 판단','',
        '1. **기존 요약과는 차이가 있다.** 전체 수준을 제거한 네 구간 변화는 14D 요약에서 거의 읽히지 않지만, 학습된 세 시계열 표현에서는 읽힌다.',
        '2. **복원 학습의 효과도 관찰된다.** 세 모델 모두 validation/test의 강약·5 feature 평균에서 같은 seed의 학습 전보다 개선됐다. 다만 학습 전 인코더에도 일부 시간 정보가 있었으므로 초기 표현을 무정보라고 볼 수 없다.',
        '3. **복잡한 모델의 우위는 특징에 따라 다르다.** 강약은 학습된 BiLSTM·Transformer가 PCA64를 앞선다. 5 feature 평균에서는 PCA64와 비슷한 수준이며, Tempo·Rubato·Pedaling은 PCA64가 더 높다. 모든 특징에서 신경망이 가장 좋다는 결과는 아니다.',
        '4. 따라서 **“이 모델들은 시간 변화를 전혀 담지 못한다”는 판단은 수정해야 한다.** 세밀한 가림 복원이 약했던 결과와, 관측된 16-beat 평균 변화가 표현에서 읽힌다는 결과는 함께 성립한다. 다음 확인 대상은 이 정보가 실제 임베딩 거리와 검색에서도 작동하는지다.','',
        '![feature별 결과](02_feature_comparison.png)','', '![학습 효과](03_training_effect.png)','',
        '## 실제 고정 사례','', '![고정된 여섯 사례](04_fixed_examples.png)','',
        '사례는 각 split에서 작품 이름순 첫·중간·마지막 작품, 중앙 연주, 중앙 선택 구간으로 고정했다. 좋은 사례를 결과에서 탐색하지 않았다.','',
        '## 데이터와 누출 방지','', '| split | 원래 작품/연주/구간 | 선택 작품/연주/구간 |','|---|---:|---:|']
    for s in protocol['support']:
        lines.append(f"| {s['split']} | {s['original_works']}/{s['original_performances']}/{s['original_windows']} | {s['works']}/{s['performances']}/{s['windows']} |")
    lines += ['',
        '- 완전 유효한 64-beat 구간만 선택하고 연주 내 구간 겹침을 제외했다. validation/test 입력은 앞선 순서 평가 입력과 byte 단위로 동일하다.',
        '- 예측기 학습과 PCA는 train 작품만 사용한다. 입력 평균·표준편차도 train에서만 계산한다.',
        '- 규제 강도 후보는 `0.0001, 0.001, 0.01, 0.1, 1, 10`. train 작품을 분리한 고정 5-fold CV로 선택한다. fold마다 정규화·PCA를 다시 train fold에서만 계산한다.',
        '- CV 선택 지표는 채널 오차를 fold-training 정답 에너지로 나눈 뒤 5 feature 동일 비중으로 평균한다. validation/test로 규제 강도나 예측기를 선택하지 않는다.',
        '- 모든 표현에 동일한 28출력 선형 ridge 절차를 적용한다. 기존 요약은 14D, 신경망과 PCA는 64D다. 인코더 출력은 L2 정규화하지 않으며 거리 검색 평가는 별도다.',
        '- 원본 448D를 train-only 표준화한 뒤 weighted PCA로 64D 압축한다. PCA도 각 train CV fold에서 별도로 학습한다.',
        '- 학습 전·후 표현별로 예측기를 별도로 학습한다. 신경망 seed는 20261006/07/08이며, 학습 후는 기존 확대 예산의 복원 validation 최선 checkpoint다.',
        '- 기존 validation/test는 이미 관찰한 작품이므로 새로운 독립 holdout으로 주장하지 않는다. 작품별 reference cohort 정규화 정책도 유지한다.','',
        '## 해석 범위와 종료 조건','',
        '이 단계는 **관측된 16-beat 평균 변화가 고정 표현에서 선형적으로 읽히는가**를 평가한다. 가려진 beat의 예측 가능성, 16 beat 내부의 세밀한 변화, 음악적 유사성, 사용자 선호를 검증하지 않는다.',
        '정답은 추출 feature에서 계산하므로 입력 정보 보존을 검증하며, feature 추출 규칙 자체의 타당성을 독립 검증하지 않는다.',
        '학습 후가 요약·초기 표현을 앞서면 이 시간 범위에서의 표현 학습 효과를 지지한다. 차이가 작거나 악화돼도 이 단계의 완결된 결과로 기록한다. 결과에 맞춰 시간 범위·모델·정답을 추가 탐색하지 않는다.',
        '3단계인 실제 임베딩 거리·검색 평가는 아직 수행하지 않았다. 선형 예측기로 정보가 읽히는 것과 cosine 거리 또는 전곡 평균/표준편차 표현이 이를 활용하는 것은 별개다.','',
        '## 산출물과 재현','',
        '- [summary.csv](summary.csv), [channel_metrics.csv](channel_metrics.csv), [work_metrics.csv](work_metrics.csv): 전체·채널·작품별 결과',
        '- [training_effects.csv](training_effects.csv): 학습 전후 짝지은 점수 차이',
        '- [cv.csv](cv.csv), [probe_parameters.csv](probe_parameters.csv): train-only 조정과 선택 규제 강도',
        '- [selected_windows.csv](selected_windows.csv), [example_quarters.csv](example_quarters.csv): 선택과 고정 사례',
        '- [protocol.json](protocol.json), [evaluation_audit.json](evaluation_audit.json), [independent_audit.json](independent_audit.json), [layout_audit.json](layout_audit.json), [figure_manifest.json](figure_manifest.json), [report_generation.json](report_generation.json): 설정·검증·최종 보고서 생성 기록',
        '- 대용량 원본 입력·목표·임베딩·예측기·예측: 저장소 밖 `Classicfy/datasets/temporal_probe_run01`','',
        '저장소 루트에서 실행한다. 기존 결과는 덮어쓰지 않는다.','', '```bash',
        'XDG_CACHE_HOME=/tmp/classicfy-cache MPLCONFIGDIR=/tmp/classicfy-mpl VECLIB_MAXIMUM_THREADS=2 \\',
        'classicfy-ai/.venv/bin/python classicfy-ai/scripts/evaluate_temporal_probe.py','',
        'classicfy-ai/.venv/bin/python classicfy-ai/scripts/evaluate_temporal_probe.py --report-only','',
        'classicfy-ai/.venv/bin/python classicfy-ai/scripts/audit_temporal_probe.py','```','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    render(parser.parse_args().out)
