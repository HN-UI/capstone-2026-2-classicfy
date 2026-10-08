"""Readable validation-only figures for the approved budget/bottleneck experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from validate_tempo import setup_style, plt
from analyze_temporal_failures import canvas, finish, NAMES, sha
from plot_sequence_autoencoders import piece_label
from embedding.data import CHANNELS

LABELS={"cnn64":"1D CNN 64D","bilstm64":"BiLSTM 64D","transformer64":"Transformer 64D",
        "bilstm128":"BiLSTM 128D","bilstm256":"BiLSTM 256D",
        "visible_mean":"보이는 값의 평균","linear_interpolation":"직선 보간"}
COLORS={"cnn64":"#647888","bilstm64":"#257AB8","transformer64":"#D17535",
        "bilstm128":"#8D68AD","bilstm256":"#D17535","visible_mean":"#A29AA9",
        "linear_interpolation":"#298E70"}
BUDGET=("cnn64","bilstm64","transformer64")
BOTTLENECK=("bilstm64","bilstm128","bilstm256")
METRICS=(("mse","숨긴 값의 오차","값 MSE ↓",None),
         ("shape_correlation","구간 평균을 빼도 모양이 비슷한가?","구간 중심 제거 상관 ↑",0),
         ("shape_amplitude_ratio","실제 변화 폭의 몇 %를 복원했나?","예측/실제 표준편차 (%) · 100이 일치",100),
         ("slope_mse","인접 beat 변화량의 오차","변화량 MSE ↓",None),
         ("direction_percent","상승·하강 방향을 맞혔나?","방향 일치 (%) ↑",50))


def points(ax,table,arms,metric,offset=0,color=None,label=None):
    scale=100 if metric=="shape_amplitude_ratio" else 1
    for i,arm in enumerate(arms):
        y=table[table.model==arm][metric].dropna().to_numpy()*scale
        if not len(y):continue
        avg=y.mean()
        ax.errorbar(i+offset,avg,yerr=[[avg-y.min()],[y.max()-avg]],fmt="o",ms=8,lw=2,capsize=4,
                    color=color or COLORS[arm],label=label if i==0 else None)
        ax.annotate(f"{avg:.3f}" if metric in ("mse","slope_mse","shape_correlation") else f"{avg:.1f}%",
                    (i+offset,y.max()),xytext=(0,9),textcoords="offset points",ha="center",fontsize=10)
    ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms],fontsize=10,rotation=12)
    ax.margins(x=.15,y=.25)


def learning(directory,history,arms,title):
    fig,axes=canvas(title,"같은 입력·학습 순서·가림 · 실선=검증 · 점선=학습 batch 평균 · 각 색=하나의 seed",1,3,(20,8))
    for ax,arm in zip(axes.flat,arms):
        for seed,g in history[history.arm==arm].groupby("seed"):
            line=ax.plot(g.epoch,g.validation_value_mse,lw=2,label=f"seed {seed}")[0]
            ax.plot(g.epoch,g.train_objective,ls="--",color=line.get_color(),alpha=.45)
            row=g.loc[g.validation_value_mse.idxmin()]
            ax.scatter(row.epoch,row.validation_value_mse,marker="*",s=130,color=line.get_color(),zorder=5)
        ax.axvline(20,color="#888",ls=":",label="이전 예산 20 epoch")
        ax.set(title=LABELS[arm],xlabel="학습 epoch",ylabel="5 feature 동일 비중 MSE ↓",xlim=(0,80));ax.legend(fontsize=9)
    finish(fig,directory/"01_learning",
        "★=검증 값 MSE가 가장 작은 epoch. 최소 20 epoch 이후 10회 연속 개선이 없으면 종료하며 최대 80 epoch입니다.\n"
        "학습 손실은 매 batch 평균이며 가림이 바뀝니다. 검증은 고정 mask의 전체 합계입니다. 곡선 높이를 그대로 일반화 차이로 해석하지 않습니다.",top=.80,bottom=.20,wspace=.32)


def budget_summary(directory,summary):
    fig,axes=canvas("1번 · 더 오래 학습하면 값과 시간 변화가 함께 좋아지는가?",
        "검증 8작품·93연주·935구간 | 같은 학습 경로의 첫 20 epoch 최선 vs 최대 80 epoch 내 최선",2,3,(20,11))
    for ax,(metric,title,ylabel,reference) in zip(axes.flat,METRICS):
        for cond,offset,color,label in (("best20",-.12,"#7D8A95","20 epoch 내 최선"),("best_extended",.12,"#267AB8","확대 예산 내 최선")):
            points(ax,summary[summary.condition==cond],BUDGET,metric,offset,color,label)
        ax.set_title(title,loc="left",fontsize=14,pad=14);ax.set_ylabel(ylabel);ax.legend(fontsize=9)
        if reference is not None:ax.axhline(reference,color="#666",ls="--",lw=1)
        if metric=="direction_percent":ax.set_ylim(0,100)
        if metric=="shape_amplitude_ratio":ax.set_ylim(bottom=0)
    axes.flat[-1].set_axis_off()
    lines=[]
    for arm in BUDGET:
        sub=summary[summary.model==arm].pivot(index="seed",columns="condition",values="mse")
        gain=(100*(1-sub.best_extended/sub.best20)).mean()
        lines.append(f"{LABELS[arm]} 값 오차 {gain:.2f}% 감소")
    axes.flat[-1].text(.02,.94,"이번 예산 확대 결과\n\n"+"\n".join(lines)+
        "\n\n추가 epoch 안에 이전 checkpoint도 포함됩니다.\n선택 기준인 값 MSE는 같거나 낮아질 수밖에\n있으므로, 다른 지표와 작품별 결과도 봅니다.",va="top",fontsize=12,linespacing=1.6)
    mean_baseline=summary[(summary.model=="visible_mean")&(summary.condition=="baseline")].iloc[0]
    axes.flat[3].axhline(mean_baseline.slope_mse,color="#8D68AD",ls=":",label="보이는 값 평균 예측")
    axes.flat[3].legend(fontsize=8.5,loc="lower left")
    axes.flat[2].legend(fontsize=9,loc="upper right")
    finish(fig,directory/"02_budget_quality",
        "점=3 seed 평균, 선=최소~최대이며 신뢰구간이 아닙니다. 페달 세 채널을 한 그룹으로 묶어 5feature를 같은 비중으로 평균합니다.\n"
        "모양과 변화 폭은 각 64-beat 구간의 hidden 평균 제거 후 계산합니다. 변화량은 두 이웃 beat가 모두 hidden일 때, 방향은 실제 변화가 0이 아닐 때 평가합니다.\n"
        "변화 폭은 100%가 크기의 일치이며 높을수록 무조건 좋은 지표가 아닙니다. 50% 선은 균등 무작위 상승·하강 부호의 기대값입니다.",top=.82,bottom=.20,hspace=.65)


def stopped_and_cost(directory,runs,history,arms):
    rows=pd.DataFrame(runs).set_index(["arm","seed"])
    fig,axes=canvas("얼마나 더 학습했고, 언제의 모델을 선택했는가?",
        "점=seed별 실제 값 | 80에 도달했다는 사실은 수렴의 증명이 아닙니다",1,2,(17,8))
    for ax,key,title in zip(axes.flat,("best_epoch","completed_epochs"),("검증 최선 epoch","실제 학습 종료 epoch")):
        for i,arm in enumerate(arms):
            g=rows.loc[arm]
            ax.scatter(i+np.linspace(-.10,.10,len(g)),g[key],color=COLORS[arm],s=65)
        ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms],rotation=12)
        ax.set(title=title,ylabel="Epoch",ylim=(0,85));ax.axhline(20,color="#888",ls=":")
    finish(fig,directory/"03_epoch_selection",
        "최선은 검증 값 MSE로만 선택합니다. 종료는 최소 20 epoch 이후 10회 연속 무개선 또는 최대 80 epoch입니다.\n"
        "종료 epoch와 최선 epoch는 다릅니다. 예산 끝까지 오차가 개선되는 모델은 아직 충분히 수렴했는지 알 수 없습니다.",top=.80,bottom=.20)
    fig,axes=canvas("학습 예산 확대에 실제로 얼마의 시간이 들었는가?",
        "같은 실행의 epoch별 누적 시간 | CPU 2 threads | 막대=3 seed 평균 · 선=최소~최대",size=(17,8))
    ax=axes[0,0]
    for offset,kind,color,label in ((-.15,"first20","#83909A","첫 20 epoch"),(.15,"complete","#267AB8","실제 전체 학습")):
        for i,arm in enumerate(arms):
            vals=[]
            for seed,g in history[history.arm==arm].groupby("seed"):
                vals.append(g[g.epoch==20].cumulative_seconds.iloc[0] if kind=="first20" else g.cumulative_seconds.iloc[-1])
            vals=np.asarray(vals);avg=vals.mean()
            ax.bar(i+offset,avg/60,.28,color=color,label=label if i==0 else None,
                   yerr=[[(avg-vals.min())/60],[(vals.max()-avg)/60]],capsize=4)
    ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms]);ax.set_ylabel("학습·검증 소요 시간 (분)");ax.legend()
    finish(fig,directory/"04_compute_cost",
        "epoch 시간은 학습·검증·개선 checkpoint 저장을 포함하고, 공통 mask schedule 생성과 epoch 종료 resume 저장은 제외합니다.\n"
        "같은 장비라도 시스템 부하가 달라질 수 있어 엄밀한 속도 벤치마크가 아닙니다. 모델별 실제 epoch 수가 다릅니다.",top=.80,bottom=.18)


def work_heatmap(directory,works,arms,condition,reference_arm=None):
    fig,axes=canvas("개선이 여러 작품에서 반복되는가?",
        "같은 seed·같은 작품에서 비교한 MSE 감소율의 3 seed 평균 | +파랑=개선 · −주황=악화",1,2,(18,9))
    for ax,metric,title in zip(axes.flat,("mse","slope_mse"),("값 복원","인접 변화량 복원")):
        records=[]
        for arm in arms:
            new=works[(works.model==arm)&(works.condition==condition)].set_index(["seed","piece"])[metric]
            if reference_arm:
                base=works[(works.model==reference_arm)&(works.condition==condition)].set_index(["seed","piece"])[metric]
            else:
                base=works[(works.model==arm)&(works.condition=="best20")].set_index(["seed","piece"])[metric]
            records.append((100*(1-new/base)).groupby("piece").mean().rename(arm))
        table=pd.concat(records,axis=1)
        limit=max(1.,float(np.nanmax(np.abs(table.to_numpy()))))
        im=ax.imshow(table.to_numpy(),aspect="auto",cmap="RdBu",vmin=-limit,vmax=limit);ax.grid(False)
        ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms],rotation=12,fontsize=10)
        ax.set_yticks(range(len(table)),[piece_label(p) for p in table.index],fontsize=9);ax.set_title(title,pad=12)
        for y in range(len(table)):
            for x in range(len(arms)):
                value=table.iloc[y,x]
                ax.text(x,y,f"{value:+.1f}%",ha="center",va="center",fontsize=11,color="white" if abs(value)>.65*limit else "#23343F")
        fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label="MSE 감소율 (%)")
    reference="같은 seed의 BiLSTM 64D 확대 예산 최선" if reference_arm else "같은 seed·같은 모델의 첫 20 epoch 내 최선"
    finish(fig,directory/"05_work_effects",f"비교 기준: {reference}. 감소율=100×(1−새 MSE/기준 MSE).\n"
        "각 작품에서 5feature를 같은 비중으로 요약한 뒤 seed별 감소율을 평균합니다. 작품별 구간 수가 달라 전체 pooled 결과와는 다를 수 있습니다.",top=.81,bottom=.18,wspace=.75,right=.92)


def bottleneck_structure(directory,runs):
    fig,axes=canvas("3번 · 임베딩 크기를 늘릴 때 무엇이 달라지는가?",
        "BiLSTM 인코더와 복원기 중간 폭은 고정 | 64D 기준 실행은 1번 실험에서 재사용",size=(17,9))
    ax=axes[0,0];ax.set_axis_off()
    texts=[("동일 입력", "7 feature × 64 beat + 유효 여부·가림 mask"),
           ("동일 BiLSTM 인코더", "입력 projection 64 → 양방향 LSTM 2층, 방향당 상태 32 → beat별 64차원"),
           ("이번에 바꾸는 압축 크기", "64 beat × beat별 64차원 = 4096 → 압축 64 / 128 / 256차원"),
           ("복원기", "64·128·256 → 중간 128차원 → 448개 값 (7 × 64)")]
    for i,(title,body) in enumerate(texts):
        y=.93-i*.23
        ax.add_patch(FancyBboxPatch((.08,y-.17),.84,.17,boxstyle="round,pad=.015",transform=ax.transAxes,
                    fc="#F7EBDD" if i==2 else "#E7F0F6",ec="#C5D2DB"))
        ax.text(.5,y-.025,title,ha="center",va="top",fontsize=15,fontweight="bold",transform=ax.transAxes)
        ax.text(.5,y-.085,body,ha="center",va="top",fontsize=12,transform=ax.transAxes)
        if i<3:ax.annotate("",(.5,y-.22),(.5,y-.18),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",lw=2,color="#74848F"))
    finish(fig,directory/"00_structure",
        "압축 벡터를 우회하는 연결은 없습니다. 인코더와 복원기 출력층은 같은 seed에서 동일한 초기 가중치를 사용합니다.\n"
        "차원이 커지면 압축층과 복원기 입력층의 파라미터 수도 늘고 초기화가 달라집니다. 순수한 정보 용량만의 효과를 증명하는 실험은 아닙니다.",top=.85,bottom=.18)


def feature_effects(directory,channels,arms,reference_arm=None):
    fig,axes=canvas("어떤 feature에서 개선 또는 악화가 생겼는가?",
        "행=개별 채널 | 같은 seed끼리 계산한 MSE 감소율의 평균 | +파랑=개선 · −주황=악화",1,2,(18,9))
    for ax,metric,title in zip(axes.flat,("mse","slope_mse"),("값 복원","인접 변화량 복원")):
        records=[]
        for arm in arms:
            new=channels[(channels.model==arm)&(channels.condition=="best_extended")].set_index(["seed","channel"])[metric]
            base=channels[(channels.model==(reference_arm or arm))&
                          (channels.condition==("best_extended" if reference_arm else "best20"))].set_index(["seed","channel"])[metric]
            records.append((100*(1-new/base)).groupby("channel").mean().rename(arm))
        table=pd.concat(records,axis=1).reindex(CHANNELS)
        limit=max(1.,float(np.nanmax(np.abs(table.to_numpy()))))
        im=ax.imshow(table,aspect="auto",cmap="RdBu",vmin=-limit,vmax=limit);ax.grid(False)
        ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms],rotation=12,fontsize=10)
        ax.set_yticks(range(7),NAMES,fontsize=10)
        ax.set_title(title,pad=12)
        for y in range(7):
            for x in range(len(arms)):
                value=table.iloc[y,x]
                ax.text(x,y,f"{value:+.1f}%",ha="center",va="center",fontsize=11,
                        color="white" if abs(value)>.65*limit else "#23343F")
        fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label="MSE 감소율 (%)")
    reference="같은 seed의 BiLSTM 64D 확대 예산 최선" if reference_arm else "같은 모델·seed의 첫 20 epoch 내 최선"
    finish(fig,directory/"06_feature_effects",f"비교 기준: {reference}. 감소율=100×(1−새 MSE/기준 MSE).\n"
        "이 그림은 페달 3개를 각각 표시합니다. 종합 지표에서는 페달 3채널을 한 그룹으로 묶은 5feature 동일 비중을 사용합니다.",top=.81,bottom=.18,wspace=.65,right=.92)


def bottleneck_quality(directory,summary):
    fig,axes=canvas("3번 · 압축 차원을 늘리면 시간 변화가 더 잘 살아나는가?",
        "검증 전체 구간 · 최대 80 epoch 내 값 MSE 최선 checkpoint · 점=3 seed 평균 · 선=최소~최대",2,3,(20,11))
    selected=summary[summary.condition.isin(("best_extended","baseline"))]
    arms=(*BOTTLENECK,"visible_mean","linear_interpolation")
    for ax,(metric,title,ylabel,reference) in zip(axes.flat,METRICS):
        points(ax,selected,arms,metric);ax.set_title(title,loc="left",fontsize=14,pad=14);ax.set_ylabel(ylabel)
        if reference is not None:ax.axhline(reference,color="#666",ls="--",lw=1)
        if metric=="direction_percent":ax.set_ylim(0,100)
        if metric=="shape_amplitude_ratio":ax.set_ylim(bottom=0)
    axes.flat[-1].set_axis_off()
    axes.flat[-1].text(.02,.94,"판단할 내용\n\n① 값 오차가 줄었는가?\n② 모양 상관과 변화량 오차도 개선됐는가?\n③ 변화 폭 증가가 정확한 움직임인가?\n\n평균 예측의 모양 상관은 정의되지 않아\n점이 없습니다.\n변화 폭만 커지는 것은 성공이 아닙니다.",va="top",fontsize=12,linespacing=1.6)
    finish(fig,directory/"02_bottleneck_quality",
        "값·모양·변화 폭·인접 변화량·방향은 서로 다른 지표입니다. 변화 폭 100%만으로 실제 곡선의 정확한 복원을 뜻하지 않습니다.\n"
        "모든 차원이 같은 최대 예산과 종료 규칙을 사용하지만 실제 종료·선택 epoch는 다릅니다. 범위는 seed 최소~최대이며 신뢰구간이 아닙니다.\n"
        "5feature 동일 비중. 변화량은 두 이웃 위치 모두 hidden인 경우, 방향은 실제 변화 절댓값>10⁻⁶인 쌍만 평가합니다.",top=.82,bottom=.20,hspace=.65)


def parameter_counts(directory,runs):
    table=pd.DataFrame(runs).groupby("arm").mean(numeric_only=True).reindex(BOTTLENECK)
    fig,axes=canvas("임베딩 확대는 파라미터를 얼마나 늘리는가?",
        "인코더는 고정 · 압축층과 복원기 입력층은 차원에 따라 증가",size=(16,8))
    ax=axes[0,0];bottom=np.zeros(3)
    for key,label,color in (("encoder_parameters","BiLSTM 인코더", "#748796"),
                           ("bottleneck_parameters","압축층", "#277EBC"),
                           ("decoder_parameters","복원기", "#D08345")):
        vals=table[key].to_numpy()/10000;ax.bar(range(3),vals,bottom=bottom,color=color,label=label);bottom+=vals
    for i,y in enumerate(bottom):ax.text(i,y*1.015,f"{y*10000:,.0f}개",ha="center",fontsize=12)
    ax.set_xticks(range(3),[LABELS[a] for a in BOTTLENECK]);ax.set_ylabel("학습 가능한 파라미터 수 (만 개)");ax.legend();ax.margins(y=.2)
    finish(fig,directory/"04_parameters",
        "복원기의 중간 폭은 128로 동일하지만 첫 Linear의 입력 크기가 달라집니다. 인코더 파라미터 수는 동일합니다.\n"
        "차원 확대에 따른 표현 용량·파라미터·head 초기화 변화가 함께 있으므로, 차원 자체만의 인과 효과로 단정하지 않습니다.",top=.80,bottom=.18)


def paired_bottleneck_effects(directory,summary):
    selected=summary[summary.condition=="best_extended"]
    reference=selected[selected.model=="bilstm64"].set_index("seed")
    specs=(("mse","값 MSE 감소율","% · 양수=개선",True),
           ("shape_correlation","모양 상관 증가","상관 차이 · 양수=개선",False),
           ("shape_amplitude_ratio","변화 폭 비율 증가","퍼센트포인트 · 증가가 곧 개선은 아님",False),
           ("slope_mse","변화량 MSE 감소율","% · 양수=개선",True),
           ("direction_percent","방향 일치 증가","퍼센트포인트 · 양수=개선",False))
    fig,axes=canvas("64차원 대비 개선이 seed마다 반복되는가?",
        "같은 seed의 64D 모델과 비교 | 작은 점=각 seed · ◆=3 seed 평균",2,3,(20,11))
    for ax,(metric,title,ylabel,relative) in zip(axes.flat,specs):
        for i,arm in enumerate(BOTTLENECK[1:]):
            larger=selected[selected.model==arm].set_index("seed")[metric]
            base=reference[metric]
            values=(100*(1-larger/base) if relative else larger-base).sort_index()
            if metric=="shape_amplitude_ratio":values*=100
            ax.scatter(i+np.linspace(-.09,.09,len(values)),values,color=COLORS[arm],s=55)
            ax.scatter(i,values.mean(),color=COLORS[arm],marker="D",s=100,edgecolor="white",zorder=5)
            ax.annotate(f"{values.mean():+.4f}" if metric=="shape_correlation" else f"{values.mean():+.2f}",
                        (i,values.max()),xytext=(0,10),textcoords="offset points",ha="center",fontsize=11)
        ax.axhline(0,color="#68747E",ls="--",lw=1)
        ax.set_xticks((0,1),["128D − 64D","256D − 64D"])
        ax.set(title=title,ylabel=ylabel);ax.margins(x=.35,y=.3)
    axes.flat[-1].set_axis_off()
    axes.flat[-1].text(.02,.94,"어떻게 읽나요?\n\n0선 위: 값·변화량 오차 감소,\n또는 상관·방향 일치 증가\n\n변화 폭 패널은 크기의 증가만 뜻합니다.\n100%에 더 가까워졌는지와\n모양·방향 지표를 함께 확인합니다.\n\n3 seed는 통계적 유의성의 증명이 아닙니다.",va="top",fontsize=12,linespacing=1.6)
    finish(fig,directory/"03_paired_effects",
        "각 크기의 확대 예산 내 검증 값 MSE 최선 checkpoint를 비교합니다. 값·변화량 감소율=100×(1−큰 차원 MSE/64D MSE).\n"
        "상관·방향·변화 폭은 큰 차원 값에서 64D 값을 뺍니다. 퍼센트포인트는 두 백분율의 차이입니다. 신뢰구간이나 검정 결과가 아닙니다.",top=.82,bottom=.20,hspace=.65)


def example_figures(directory,examples,arms):
    target_dir=directory/"examples";target_dir.mkdir(exist_ok=True)
    for number,group in examples.groupby("example"):
        for ci,channel in enumerate(CHANNELS):
            g=group[group.channel==channel].sort_values("beat")
            fig,axes=canvas(f"고정 사례 {number} · {NAMES[ci].replace(chr(10),' ')} 복원",
                f"검증 · seed 20261006 · {g.key.iloc[0]}",size=(17,8))
            ax=axes[0,0]
            ax.plot(g.beat,np.where(g.valid,g.target,np.nan),color="#242C32",lw=1.8,label="실제 값 (숨긴 정답 포함)")
            for arm in arms:
                ax.plot(g.beat,np.where(g.hidden,g[f"{arm}_best_extended"],np.nan),color=COLORS[arm],marker="o",ms=3,lw=1.8,label=LABELS[arm])
            ax.plot(g.beat,np.where(g.hidden,g.linear_interpolation,np.nan),color=COLORS["linear_interpolation"],marker="o",ms=3,lw=1.2,label="직선 보간")
            ax.scatter(g.beat[g.hidden],g.target[g.hidden],marker="x",s=40,color="#242C32",label="모델에 가린 정답",zorder=8)
            ax.set(xlabel="원래 score beat 구간 번호",ylabel="작품 공통 패턴 제거 후 표준화된 feature");ax.legend(ncol=3,fontsize=10)
            finish(fig,target_dir/f"{int(number):02d}_{channel}",
                "색 선은 확대 예산에서 선택한 모델의 hidden 위치 예측만 표시합니다. 서로 떨어진 가림 구간을 연결하지 않습니다.\n"
                "검증 작품 이름순 첫·중간·마지막 작품의 중앙 연주·중앙 구간으로 고정했습니다. 성능을 보고 선택한 사례가 아닙니다.",top=.80,bottom=.20)


def report(out,summary,runs,available):
    test_record=("자동 테스트 기록은 [unit_tests.txt](unit_tests.txt)에 있습니다." if (out/"unit_tests.txt").exists()
                 else "자동 테스트는 classicfy-ai에서 `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/unit/embedding -v`로 실행합니다.")
    def table(arms,condition):
        data=summary[summary.condition==condition].groupby("model").mean(numeric_only=True)
        rows=[]
        for arm in arms:
            row=data.loc[arm]
            correlation=f"{row.shape_correlation:.3f}" if pd.notna(row.shape_correlation) else "정의 안 됨"
            rows.append(f"| {LABELS[arm]} | {row.mse:.4f} | {correlation} | {100*row.shape_amplitude_ratio:.1f}% | {row.slope_mse:.4f} | {row.direction_percent:.1f}% |")
        return "\n".join(rows)
    header="| 모델 | 값 MSE ↓ | 모양 상관 ↑ | 변화 폭 비율 | 변화량 MSE ↓ | 방향 일치 ↑ |\n|---|---:|---:|---:|---:|---:|\n"
    selected=summary[summary.condition=="best_extended"].groupby("model").mean(numeric_only=True)
    mean_baseline=summary[(summary.model=="visible_mean")&(summary.condition=="baseline")].iloc[0]
    best_budget=min(BUDGET,key=lambda a:selected.loc[a,"mse"])
    budget_slopes_above=sum(selected.loc[a,"slope_mse"]>mean_baseline.slope_mse for a in BUDGET)
    budget_findings=[]
    cnn_comparisons=[]
    extended_by_seed=summary[summary.condition=="best_extended"].pivot(index="seed",columns="model",values="mse")
    for arm in BUDGET[1:]:
        gain=100*(1-extended_by_seed[arm]/extended_by_seed.cnn64)
        cnn_comparisons.append(f"- 확대 예산의 **{LABELS[arm]} vs CNN**: 값 MSE 감소율 평균 {gain.mean():+.2f}% "
            f"(seed 범위 {gain.min():+.2f}~{gain.max():+.2f}%, 개선 {int((gain>0).sum())}/3).")
    budget_rows=[]
    for arm in BUDGET:
        paired=summary[summary.model==arm].pivot(index="seed",columns="condition",values="mse")
        gain=100*(1-paired.best_extended/paired.best20)
        budget_rows.append(f"| {LABELS[arm]} | {paired.best20.mean():.5f} | {paired.best_extended.mean():.5f} | {gain.mean():.2f}% | {gain.min():.2f}~{gain.max():.2f}% |")
        before=summary[(summary.model==arm)&(summary.condition=="best20")].mean(numeric_only=True)
        after=selected.loc[arm]
        budget_findings.append(f"- **{LABELS[arm]}**: 값 오차가 감소한 seed {int((gain>1e-9).sum())}/3. "
            f"모양 상관 {before.shape_correlation:.3f}→{after.shape_correlation:.3f}, "
            f"변화량 MSE {before.slope_mse:.4f}→{after.slope_mse:.4f}, "
            f"방향 일치 {before.direction_percent:.1f}%→{after.direction_percent:.1f}%.")
    run_table="\n".join(f"| {LABELS[r['arm']]} | {r['seed']} | {r['best_epoch']} | {r['completed_epochs']} | {'10회 무개선' if r['stop_reason']=='patience' else '80 epoch 한도'} |" for r in runs)
    budget_text=f"""# 1번: 학습 예산 확대 결과

첫 20 epoch와 최대 80 epoch를 **같은 학습 경로 안에서** 비교했습니다.
입력·작품 분할·optimizer·학습률·가림·seed는 유지하고 최대 학습 예산만 늘렸습니다.

| 모델 | 첫 20 epoch 내 값 MSE | 확대 예산 내 값 MSE | 감소율 평균 | seed별 감소율 범위 |
|---|---:|---:|---:|---:|
{chr(10).join(budget_rows)}

MSE는 3 seed 평균이며, 감소율은 각 seed 안에서 먼저 계산한 후 평균합니다.

{chr(10).join(budget_findings)}

확대 예산에서도 seed 평균 값 MSE가 가장 낮은 모델은 **{LABELS[best_budget]}**입니다.

{chr(10).join(cnn_comparisons)}

인접 변화량 MSE가 보이는 값 평균 기준선보다 높은 모델은 **{budget_slopes_above}/3**입니다.
추가 학습이 값 오차를 줄인 사실과 beat별 움직임을 충분히 복원했는지는 구분해야 합니다.

![학습 곡선](01_learning.png)
![값과 흐름 비교](02_budget_quality.png)

## 확대 예산 내 최선 모델의 검증 결과

{header}{table(BUDGET,'best_extended')}

같은 hidden 위치에서 평가한, 보이는 값만 사용하는 학습 없는 기준선입니다.

{header}{table(('visible_mean','linear_interpolation'),'baseline')}

![선택 및 종료 epoch](03_epoch_selection.png)
![추가 계산 비용](04_compute_cost.png)
![작품별 변화](05_work_effects.png)
![feature별 변화](06_feature_effects.png)

값 MSE는 선택 기준이며, 확대 예산에는 이전 checkpoint도 포함됩니다. 따라서 최소값이 줄어드는
것만으로 충분한 개선이라고 판단하지 않습니다. 모양·변화량·방향과 작품별 결과를 함께 읽습니다.
80 epoch에 도달한 실행은 수렴했다고 단정하지 않습니다.

## 고정 사례

![Dynamics 실제 복원 사례](examples/02_dynamics.png)

"""
    for number in (1,2,3):budget_text+=f"- 사례 {number}: "+" · ".join(f"[{c}](examples/{number:02d}_{c}.png)" for c in CHANNELS)+"\n"
    budget_text+="\n[실험 조건·지표·실행별 epoch·한계](../README.md)\n"
    (out/"budget/README.md").write_text(budget_text,encoding="utf-8")
    root=f"""# 후속 실험 1·3: 학습 예산과 BiLSTM 압축 차원

요청한 **1번(최대 80 epoch 학습)**과 **3번(BiLSTM 64/128/256차원 비교)**의 실제 실행 결과입니다.
같은 조건의 3 seed를 사용하며, 복원 오차와 시간 변화의 복원을 함께 비교합니다.

- [1번: 학습 예산 확대](budget/README.md)
"""
    if set(BOTTLENECK).issubset(available):
        root+="- [3번: BiLSTM 임베딩 차원 비교](bottleneck/README.md)\n"
        best_dimension=min(BOTTLENECK,key=lambda a:selected.loc[a,"mse"])
        best_slope=min(BOTTLENECK,key=lambda a:selected.loc[a,"slope_mse"])
        dimension_slopes_above=sum(selected.loc[a,"slope_mse"]>mean_baseline.slope_mse for a in BOTTLENECK)
        dimension_findings=[]
        reference=summary[(summary.model=="bilstm64")&(summary.condition=="best_extended")].set_index("seed")
        for arm in BOTTLENECK[1:]:
            bigger=summary[(summary.model==arm)&(summary.condition=="best_extended")].set_index("seed")
            gain=100*(1-bigger.mse/reference.mse)
            dimension_findings.append(f"- **{LABELS[arm]} vs 64D**: 값 MSE 감소율 평균 {gain.mean():+.2f}% "
                f"(seed 범위 {gain.min():+.2f}~{gain.max():+.2f}%, 개선 {int((gain>0).sum())}/3). "
                f"모양 상관 차이 {(bigger.shape_correlation-reference.shape_correlation).mean():+.3f}, "
                f"방향 일치 차이 {(bigger.direction_percent-reference.direction_percent).mean():+.2f}퍼센트포인트.")
        bottleneck_text=f"""# 3번: BiLSTM 임베딩 64·128·256차원 비교

이번 검증 split에서 seed 평균 **값 MSE가 가장 낮은 크기는 {LABELS[best_dimension]}**, 
**변화량 MSE가 가장 낮은 크기는 {LABELS[best_slope]}**입니다. 각 크기의 가중치 선택은 값 MSE를 기준으로 했습니다.
이 순위가 새로운 작품에서 재현되는지는 확인하지 않았습니다.

{chr(10).join(dimension_findings)}

인접 변화량 MSE가 보이는 값 평균 기준선보다 높은 크기는 **{dimension_slopes_above}/3**입니다.
값 오차가 가장 낮은 크기와 세밀한 시간 변화까지 잘 복원하는 모델은 같은 의미가 아닙니다.

![바꾼 구조](00_structure.png)

BiLSTM 폭과 층 수, 입력, 학습률, mask와 split을 유지했습니다. 인코더 가중치는 고정한 채 쓰는 것이 아니라,
**같은 초기 인코더에서 시작해 각 크기의 압축층·복원기와 함께 학습**합니다. 64D는 1번 실행을 재사용합니다.

![학습 곡선](01_learning.png)
![복원 품질](02_bottleneck_quality.png)

{header}{table(BOTTLENECK,'best_extended')}

![64차원 대비 seed별 변화](03_paired_effects.png)
![파라미터 수](04_parameters.png)
![작품별 개선](05_work_effects.png)
![feature별 개선과 악화](06_feature_effects.png)

차원 확대는 압축층과 복원기 입력층의 파라미터·초기화를 함께 바꿉니다. 복원기의 중간 폭과 출력층 구조는 동일합니다.
변화 폭이 100%에 가까워지는 것만으로 정확한 복원이라고 판단하지 않습니다.

## 고정 사례

![Dynamics 실제 복원 사례](examples/02_dynamics.png)

"""
        for number in (1,2,3):bottleneck_text+=f"- 사례 {number}: "+" · ".join(f"[{c}](examples/{number:02d}_{c}.png)" for c in CHANNELS)+"\n"
        bottleneck_text+="\n[공통 지표·종료 규칙·한계](../README.md)\n"
        (out/"bottleneck/README.md").write_text(bottleneck_text,encoding="utf-8")
    root+=f"""

## 공통 조건

train 39작품·395연주·5803구간, validation 8작품·93연주·935구간.
64-beat 구간·stride 32·기존 7채널의 정규화와 cohort reference를 그대로 사용했습니다.
seed 20261006/07/08, AdamW lr 0.001, batch 32, dropout 0.1, clipping 1, CPU 2 threads.
학습 시 유효 위치 약 20%를 최대 4-beat 연속 블록으로 숨깁니다. 결측·padding은 정답에서 제외합니다.
최소 20 epoch 이후 검증 값 MSE가 10회 연속 엄격히 개선되지 않으면 종료하며 최대 80 epoch입니다.
검증 mask는 모든 실행에서 고정합니다. **test는 평가·모델 선택에 사용하지 않았습니다.**

## 실제 선택과 종료

| 모델 | seed | 최선 epoch | 학습 종료 epoch | 종료 사유 |
|---|---:|---:|---:|---|
{run_table}

## 비교를 어떻게 보장했는가?

각 seed의 원래 DataLoader 순서와 mask를 미리 생성해 모든 모델에 재사용했습니다.
실제 각 epoch에서 사용한 batch index와 hidden mask를 다시 hash하여 같은지 검사했습니다.
첫 20 epoch schedule의 SHA-256은 기존 실험과 정확히 같아야 합니다.
64D 모델의 첫 20 epoch 검증 곡선은 원래 실행과 최대 절대 차이 2×10⁻⁶ 이내인지 검사합니다.
초기·20 epoch 내 최선·20 epoch·확대 예산 최선·마지막 checkpoint를 별도로 보존합니다.
중단 후에는 optimizer와 dropout RNG를 복원하며, 중단 없이 실행한 결과와 같은지 자동 테스트했습니다.

## 지표와 figure 해석

- **임베딩·압축 차원**: 인코더가 한 64-beat 구간을 요약해 복원기로 전달하는 숫자의 개수입니다.
  64D는 64개, 128D는 128개, 256D는 256개입니다. 원래 입력의 beat 수나 feature 수를 늘린 것이 아닙니다.
- **epoch**: 학습용 5803구간을 한 번씩 사용한 단위입니다. 검증 작품은 gradient 계산에 사용하지 않습니다.
- **조기 종료·checkpoint**: 고정 검증 오차가 10회 연속 낮아지지 않으면 학습을 멈춥니다.
  checkpoint는 특정 epoch의 저장 가중치이며, 마지막 가중치 대신 최저 검증 오차의 가중치를 선택합니다.
- **값 MSE**: hidden이면서 유효한 위치의 제곱 오차. 낮을수록 좋습니다.
- **모양 상관**: 각 64-beat 구간의 hidden 평균을 실제·예측에서 각각 제거한 뒤 상관 계산.
  서로 떨어진 hidden 블록도 같은 64-beat 구간에 있으면 함께 중심을 제거합니다.
  채널별로 중심 제거된 모든 구간의 hidden 값을 합쳐 하나의 상관을 구합니다. 구간별 상관의 단순 평균이 아닙니다.
  범위는 −1~1이며, 1은 같은 방향의 선형 변화, 0 부근은 약한 선형 관계, −1은 반대 방향입니다.
  예측이 일정해 표준편차가 0이면 상관을 정의하지 않습니다. 모양 상관만으로 변화 크기의 일치를 뜻하지 않습니다.
- **변화 폭 비율**: 같은 중심 제거 값의 예측 표준편차 / 실제 표준편차 ×100. 100%는 크기의 일치입니다.
  20%라면 실제의 약 1/5 변화 폭이며, 100%를 넘으면 실제보다 큰 변화 폭입니다.
- **변화량 MSE**: 원래 이웃한 두 beat가 모두 hidden·유효할 때 실제와 예측의 차분 오차.
- **방향 일치**: 그 인접 쌍 중 |실제 변화|>10⁻⁶인 쌍에서 부호가 같은 비율. 50%는 균등 무작위 부호 기준입니다.
- **보이는 값 평균 기준선**: 각 구간·채널에서 가리지 않은 유효 값의 평균으로 hidden 위치를 모두 예측합니다.
- **직선 보간 기준선**: 그 관측점들을 원래 beat 위치에서 직선으로 연결합니다. 양 끝 바깥은 최근접 관측값,
  관측점이 전혀 없는 경우는 0을 사용합니다. 두 기준선 모두 가린 정답을 계산에 사용하지 않습니다.

채널별 지표를 먼저 구하고 Tempo/Rubato/Dynamics/Articulation/Pedaling의 5그룹을 동일 비중으로
평균합니다. 페달 세 채널은 한 그룹입니다. 점은 seed 평균, 범위는 최소~최대이며 신뢰구간이 아닙니다.
그림에 사용하는 고정 사례는 기존 검증 작품 이름순 첫·중간·마지막의 중앙 연주·중앙 구간입니다.
위 명칭은 복원 결과를 비교하는 통계량이며, 각 계산 방법을 함께 기록했습니다.

## 범위와 한계

이미 관찰한 하나의 validation split에서 개발한 결과입니다. 새로운 독립 holdout 결과가 아닙니다.
최선 epoch 선택과 결과 요약에 같은 validation을 사용하므로 확정적인 일반화 성능으로 주장하지 않습니다.
작품별 공통 패턴·scale은 기존의 해당 작품 reference cohort 전체 연주로 계산한 값을 재사용합니다.
따라서 새 작품의 한 연주만 들어오는 상황에서의 성능을 검증한 것은 아닙니다.
early stopping 때문에 실제 epoch 수와 계산량이 다릅니다. 마지막 epoch까지 계속 개선되면 예산 상한의 영향이 남습니다.
차원 확대는 head 파라미터 수와 초기화를 함께 바꾸므로 순수한 정보 용량만의 효과로 단정하지 않습니다.
학습 인코더가 좋아져도 구간 평균·표준편차로 전곡을 요약하면 전곡의 구간 배치 순서는 잃습니다.
이 실험은 복원 진단이며 추천 품질·사람의 청취 유사도를 입증하지 않습니다.

## 파일과 재현

`summary.csv`, `channel_metrics.csv`, `work_metrics.csv`, `history.csv`, `example_beats.csv`,
`training_runs.json`, `protocol.json`과 PNG/SVG figure를 제공합니다.
{test_record} 산출물 검증은 아래 명령으로
`audit.json`에 기록합니다.
`layout_audit.json`은 제목·축·범례·주석이 캔버스 안에 들어오는지 검사합니다.
글자 간 겹침을 모두 검출하지는 않으므로 대표 그림의 직접 확인을 함께 수행했습니다.
대용량 mask schedule과 checkpoint는 `Classicfy/datasets/extended_sequences_run01`에 보존합니다.
기존 20 epoch 산출물은 덮어쓰지 않습니다. protocol이 같으면 완료 실행을 재사용하고 중단 실행은 이어갑니다.

```bash
XDG_CACHE_HOME=/tmp/classicfy-cache MPLCONFIGDIR=/tmp/classicfy-mpl \\
classicfy-ai/.venv/bin/python classicfy-ai/scripts/extend_sequence_experiments.py --stage all
XDG_CACHE_HOME=/tmp/classicfy-cache MPLCONFIGDIR=/tmp/classicfy-mpl \\
classicfy-ai/.venv/bin/python classicfy-ai/scripts/check_extended_figure_layout.py
classicfy-ai/.venv/bin/python classicfy-ai/scripts/audit_extended_sequences.py
```

[이전 20 epoch 모델 비교](../sequence_autoencoders/README.md)
"""
    (out/"README.md").write_text(root,encoding="utf-8")


def render(out):
    out=Path(out);setup_style();plt.rcParams.update({"svg.fonttype":"path","font.size":11})
    summary=pd.read_csv(out/"summary.csv");history=pd.read_csv(out/"history.csv")
    works=pd.read_csv(out/"work_metrics.csv");examples=pd.read_csv(out/"example_beats.csv")
    channels=pd.read_csv(out/"channel_metrics.csv")
    runs=json.loads((out/"training_runs.json").read_text());available={r["arm"] for r in runs}
    directory=out/"budget";directory.mkdir(exist_ok=True)
    learning(directory,history,BUDGET,"1번 · 20 epoch 이후에도 학습이 개선되는가?")
    budget_summary(directory,summary)
    stopped_and_cost(directory,[r for r in runs if r["arm"] in BUDGET],history,BUDGET)
    work_heatmap(directory,works,BUDGET,"best_extended")
    feature_effects(directory,channels,BUDGET)
    example_figures(directory,examples,BUDGET)
    if set(BOTTLENECK).issubset(available):
        directory=out/"bottleneck";directory.mkdir(exist_ok=True)
        bottleneck_structure(directory,runs)
        learning(directory,history,BOTTLENECK,"3번 · 임베딩 크기를 바꾸면 학습이 달라지는가?")
        bottleneck_quality(directory,summary)
        paired_bottleneck_effects(directory,summary)
        parameter_counts(directory,runs)
        work_heatmap(directory,works,BOTTLENECK[1:],"best_extended",reference_arm="bilstm64")
        feature_effects(directory,channels,BOTTLENECK[1:],reference_arm="bilstm64")
        example_figures(directory,examples,BOTTLENECK)
    report(out,summary,runs,available)
    manifest={"plot_source_sha256":sha(Path(__file__)),
              "figures":{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*.png"))},
              "tables":{p.name:sha(p) for p in sorted(out.glob("*.csv"))}}
    (out/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    render(parser.parse_args().out)
