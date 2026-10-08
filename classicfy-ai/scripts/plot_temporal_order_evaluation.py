"""Explain the perturbation task and report fixed-protocol embedding results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

from validate_tempo import setup_style, plt
from analyze_temporal_failures import sha


ARMS = ("cnn64", "bilstm64", "transformer64")
NAMES = {"cnn64":"1D CNN", "bilstm64":"BiLSTM", "transformer64":"Transformer",
         "raw":"원본 시계열 RMS", "summary14":"기존 14D 요약"}
STAGES = ("initial", "best20", "best_extended")
STAGE_NAMES = {"initial":"학습 전", "best20":"20 epoch 내 최선", "best_extended":"확대 예산 최선"}
ORDERS = ("shuffle", "block8", "reverse", "adjacent")
ORDER_NAMES = {"shuffle":"전체 순서 섞기", "block8":"8-beat 블록 재배치", "reverse":"시간 역순", "adjacent":"이웃 beat 쌍 교환"}
CHANNELS = ("Tempo · 빠르기", "Rubato · 속도 편차", "Dynamics · 강약",
            "Articulation · 건반 유지", "Pedal depth · 깊이", "Pedal ratio · 사용 비율", "Pedal changes · 전환")
COLORS = {"initial":"#A7ADB5", "best20":"#5496C8", "best_extended":"#187B67",
          "cnn64":"#327BA5", "bilstm64":"#27856A", "transformer64":"#D07539"}


def render(out):
    out = Path(out)
    setup_style()
    table = pd.read_csv(out / "summary.csv")
    protocol = json.loads((out / "protocol.json").read_text())
    manifest, violations = [], []

    def save(fig, name):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        for text in fig.findobj(match=lambda item: isinstance(item, Text)):
            if not text.get_visible() or not text.get_text().strip():
                continue
            # Tick labels outside view limits are not painted by the axes.
            if text.axes is not None and text not in text.axes.texts:
                box = text.get_window_extent(renderer)
                if box.x1 < 0 or box.x0 > fig.bbox.width or box.y1 < 0 or box.y0 > fig.bbox.height:
                    continue
            box = text.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width + 2 or box.y1 > fig.bbox.height + 2:
                violations.append(dict(figure=name, text=text.get_text(), bounds=list(box.extents)))
        for extension in ("png", "svg"):
            path = out / f"{name}.{extension}"
            fig.savefig(path, dpi=150)
            manifest.append(dict(file=path.name, sha256=sha(path)))
        plt.close(fig)

    def header(fig, title, subtitle, note):
        fig.suptitle(title, x=.06, y=.975, ha="left", fontsize=23, fontweight="bold")
        fig.text(.06, .925, subtitle, fontsize=12, va="top", color="#56616A")
        fig.text(.06, .025, note, fontsize=10, va="bottom", color="#56616A", linespacing=1.5)

    example = pd.read_csv(out / "example_beats.csv")
    example = example[(example.split == "validation") & (example.example == 2)]
    fig, axes = plt.subplots(7, 2, figsize=(16, 18))
    first = example.iloc[0]
    fig.suptitle("무엇을 평가하나? 작은 흔들림과 순서 변경을 구분하기", x=.06, y=.98, ha="left", fontsize=22, fontweight="bold")
    fig.text(.06, .95, f"고정 검증 사례: {first['piece']} / {Path(first['key']).stem} / 시작 beat {first['start']}\n검정 A=원본 · 파랑 B=작은 잡음 · 주황 C=같은 값들의 순서 변경", fontsize=11, va="top")
    for ci in range(7):
        data = example[example.channel == ci]
        for col, (variant,color) in enumerate((("noise","#327BA5"),("shuffle","#D07539"))):
            ax = axes[ci,col]
            ax.plot(data.beat, data.original, color="#323B42", lw=1.7, label="A 원본")
            ax.plot(data.beat, data[variant], color=color, lw=1.4, alpha=.85, label="B 작은 잡음" if col == 0 else "C 순서 변경")
            ax.set_ylabel(CHANNELS[ci], fontsize=10)
            ax.grid(alpha=.25)
            if ci == 0:
                ax.set_title("A–B: 흐름은 거의 유지", fontsize=16) if col == 0 else ax.set_title("A–C: 채널별 분포는 같고 순서만 변경", fontsize=16)
                ax.legend(fontsize=10, loc="upper right")
            if ci == 6:
                ax.set_xlabel("구간 내 beat")
    fig.text(.06,.018,"정답으로 설정한 관계: 거리(A,B) < 거리(A,C). 잡음 RMS는 각 채널의 원본 구간 표준편차의 5%.\n7채널을 함께 재배치합니다. 이 그림은 실제 원본 feature와 인위적 변형이며, 서로 다른 실제 연주 3개가 아닙니다.", fontsize=11)
    fig.subplots_adjust(left=.11,right=.97,top=.88,bottom=.075,hspace=.42,wspace=.27)
    save(fig,"00_task_example")

    primary = table[(table.noise == .05) & (table.order == "shuffle")]
    methods = [("summary14","baseline"),("raw","baseline")] + [(a,s) for a in ARMS for s in STAGES]
    fig, axes = plt.subplots(1,2,figsize=(18,10))
    for ax,split in zip(axes,("validation","test")):
        labels=[]
        for i,(arm,stage) in enumerate(methods):
            values=primary[(primary.split==split)&(primary.arm==arm)&(primary.stage==stage)].score_percent.to_numpy()
            avg=values.mean()
            color="#6E7781" if stage=="baseline" else COLORS[stage]
            ax.barh(i,avg,color=color,height=.7)
            ax.errorbar(avg,i,xerr=[[avg-values.min()],[values.max()-avg]],fmt="none",color="#28343B",capsize=3)
            ax.text(min(avg+1.3,102),i,f"{avg:.1f}%",va="center",fontsize=11)
            labels.append(NAMES[arm]+("" if stage=="baseline" else " · "+STAGE_NAMES[stage]))
        ax.set_yticks(range(len(methods)),labels,fontsize=10)
        ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.grid(axis="x",alpha=.25)
        support=next(s for s in protocol['support'] if s['split']==split)
        ax.set_title(f"{'검증' if split=='validation' else '기존 test'} · {support['selected_works']}작품 / {support['selected_windows']}구간",fontsize=16,pad=15)
        ax.set_xlabel("B를 C보다 가깝게 배치한 점수 (%) ↑")
    header(fig,"작은 잡음보다 순서 변경을 더 멀게 배치하는가?",
           "주평가: 잡음 5% vs 전체 순서 섞기 | 복원기 없이 64D 구간 임베딩의 cosine 거리만 사용",
           "B가 더 가까우면 1점, 동률은 0.5점. 구간 → 연주 → 작품 순으로 평균하고 작품을 동일 비중으로 집계합니다.\n신경망 막대=3 model seed 평균, 선=최소~최대(신뢰구간 아님). 원본 RMS·학습 전 모델도 높으면 이 과제만으로 학습의 추가 가치를 입증하기 어렵습니다.")
    fig.subplots_adjust(left=.19,right=.95,top=.84,bottom=.14,wspace=.65)
    save(fig,"01_primary_comparison")

    fig, axes = plt.subplots(1,2,figsize=(19,10))
    for ax,split in zip(axes,("validation","test")):
        matrix=[];labels=[]
        for arm,stage in methods:
            values=[]
            for order in ORDERS:
                subset=table[(table.split==split)&(table.arm==arm)&(table.stage==stage)&(table.order==order)&(table.noise==.05)]
                values.append(subset.score_percent.mean())
            matrix.append(values);labels.append(NAMES[arm]+("" if stage=="baseline" else " · "+STAGE_NAMES[stage]))
        matrix=np.asarray(matrix)
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap="YlGnBu",aspect="auto")
        ax.set_yticks(range(len(methods)),labels,fontsize=10)
        ax.set_xticks(range(len(ORDERS)),[ORDER_NAMES[o] for o in ORDERS],fontsize=10,rotation=15)
        for i in range(len(methods)):
            for j in range(len(ORDERS)):
                ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",fontsize=11,color="white" if matrix[i,j]>65 else "#23333D")
        ax.set_title("검증" if split=="validation" else "기존 test",fontsize=16,pad=14)
    fig.colorbar(im,ax=axes,shrink=.7,pad=.02,label="B를 더 가깝게 배치한 점수 (%)")
    header(fig,"순서를 바꾸는 방법이 달라도 결과가 유지되는가?",
           "고정 민감도 검사: 잡음 5% | 서로 다른 4가지 순서 변경 | 각 칸=작품 동일 비중·3 seed 평균",
           "8-beat 블록은 블록 내부 순서를 유지합니다. 역순·이웃 교환도 채널별 값 분포를 보존합니다.\n원본과 순서 변경본이 사실상 동일한 구간은 해당 조건에서 모든 방법에 공통으로 제외합니다. 인위적 변형의 난이도는 조건마다 다릅니다.")
    fig.subplots_adjust(left=.18,right=.86,top=.84,bottom=.17,wspace=.68)
    # Colorbar is positioned explicitly after layout so it cannot overlap axes.
    fig.axes[-1].set_position([.90,.23,.018,.54])
    save(fig,"02_order_conditions")

    fig,axes=plt.subplots(1,2,figsize=(17,9))
    for ax,split in zip(axes,("validation","test")):
        for arm in ("summary14","raw"):
            rows=table[(table.split==split)&(table.arm==arm)&(table.order=="shuffle")].sort_values("noise")
            ax.plot(rows.noise*100,rows.score_percent,marker="o",color="#7C669A" if arm=="summary14" else "#323B42",label=NAMES[arm])
        for arm in ARMS:
            for stage in ("initial","best_extended"):
                rows=table[(table.split==split)&(table.arm==arm)&(table.stage==stage)&(table.order=="shuffle")]
                means=rows.groupby("noise").score_percent.mean()
                ax.plot(means.index*100,means.values,marker="o",ls="--" if stage=="initial" else "-",color=COLORS[arm],label=NAMES[arm]+" · "+STAGE_NAMES[stage])
        ax.set(xlabel="잡음 RMS / 원본 구간 표준편차 (%)",ylabel="B를 더 가깝게 배치한 점수 (%)",ylim=(-3,103),xticks=[2,5,10])
        ax.set_title("검증" if split=="validation" else "기존 test",fontsize=16);ax.grid(alpha=.25)
    axes[1].legend(loc="upper left",bbox_to_anchor=(1.02,1),fontsize=10)
    header(fig,"작은 잡음의 크기를 바꾸면 판단이 달라지는가?",
           "전체 순서 섞기 고정 | 점선=학습 전 · 실선=확대 예산 최선 | 같은 잡음 방향을 크기만 바꾸어 사용",
           "2%·5%·10%는 원본 구간/채널 표준편차 기준이며 실제 음악적 허용 오차를 뜻하지 않습니다. 같은 점수의 선은 겹쳐 보입니다.\n조건을 결과에 맞춰 선택하지 않았습니다. 높은 점수가 포화되면 더 어려운 실제 연주 비교가 필요합니다.")
    fig.subplots_adjust(left=.08,right=.78,top=.83,bottom=.17,wspace=.30)
    save(fig,"03_noise_sensitivity")

    fig,axes=plt.subplots(1,2,figsize=(17,9))
    effects=[]
    for ax,split in zip(axes,("validation","test")):
        for ai,arm in enumerate(ARMS):
            for oi,order in enumerate(ORDERS):
                rows=table[(table.split==split)&(table.arm==arm)&(table.order==order)&(table.noise==.05)]
                wide=rows.pivot(index="seed",columns="stage",values="score_percent")
                delta=wide.best_extended-wide.initial
                x=oi+(ai-1)*.20
                avg=delta.mean()
                ax.errorbar(x,avg,yerr=[[avg-delta.min()],[delta.max()-avg]],fmt="o",color=COLORS[arm],capsize=4,label=NAMES[arm] if oi==0 else None)
                effects.append(dict(split=split,arm=arm,order=order,mean_pp=avg,min_pp=delta.min(),max_pp=delta.max()))
        ax.axhline(0,color="#4A5158",ls="--");ax.set_xticks(range(4),[ORDER_NAMES[o] for o in ORDERS],rotation=15,fontsize=10)
        ax.set_title("검증" if split=="validation" else "기존 test",fontsize=16);ax.set_ylabel("학습 후 − 학습 전 (퍼센트포인트)");ax.grid(axis="y",alpha=.25);ax.legend(fontsize=10)
        if max(abs(row['mean_pp']) for row in effects if row['split']==split)<1:
            ax.set_ylim(-1,1)
    header(fig,"이 과제에서 학습 자체가 추가로 도움이 되었는가?",
           "잡음 5% | 같은 architecture·같은 seed의 확대 예산 최선 − 초기 모델 | 양수=학습 후 점수 상승",
           "점=3 seed의 짝지은 차이 평균, 선=최소~최대이며 신뢰구간이 아닙니다.\n초기 모델부터 점수가 높으면 향상 여지가 작습니다. 음수도 표현 전체의 악화를 뜻하지 않고 이 특정 합성 과제에서의 변화만 뜻합니다.")
    fig.subplots_adjust(left=.10,right=.96,top=.83,bottom=.18,wspace=.30)
    save(fig,"04_training_effect")
    pd.DataFrame(effects).to_csv(out/"training_effects.csv",index=False)
    (out/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"report_generation.json").write_text(json.dumps(dict(
        plot_source_sha256=sha(Path(__file__)), evaluation_protocol_sha256=sha(out/"protocol.json"),
        summary_sha256=sha(out/"summary.csv"),
        note="Evaluation protocol retains its original pre-evaluation source hashes. This records final figure/report code after editorial revisions; numerical evaluation is unchanged."),
        ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"layout_audit.json").write_text(json.dumps(dict(status="passed" if not violations else "failed",figures=5,violations=violations,
        limitation="Text canvas-bound check complements visual review; it does not detect every overlap."),ensure_ascii=False,indent=2),encoding="utf-8")
    write_report(out,table,protocol)
    if violations:
        raise ValueError("Figure text extends outside canvas: see layout_audit.json")


def write_report(out,table,protocol):
    lines=["# 시간 순서 차이가 임베딩 거리에 반영되는가?", "",
        "저장된 1D CNN·BiLSTM·Transformer의 **64D 구간 임베딩**을 복원기 없이 평가했다. 새 학습·모델 선택은 하지 않았다.", "",
        "**이 실험은 작은 값 잡음보다 시간 순서 변경을 더 멀게 배치하는지 검사한다. 실제 연주 간 음악적 유사성·사용자 선호·전곡 검색 성능 평가는 아니다.**", "",
        "## 평가 문제", "", "A=실제 원본 feature 구간, B=작은 잡음을 더한 구간, C=같은 값의 순서를 바꾼 구간이다.",
        "`거리(A,B) < 거리(A,C)`이면 1점, 동률은 0.5점, 반대면 0점이다. C는 채널별 전체 값 분포를 보존한다.", "",
        "![실제 feature의 원본과 변형](00_task_example.png)", "", "## 주평가 결과", "",
        "주조건은 사전에 고정한 **잡음 5% vs 전체 순서 섞기**다. 점수는 구간→연주→작품 순으로 평균하며 작품에 같은 비중을 준다.", "",
        "| 방법 | 검증 점수 | 기존 test 점수 |", "|---|---:|---:|"]
    primary=table[(table.noise==.05)&(table.order=="shuffle")]
    methods=[("summary14","baseline"),("raw","baseline")]+[(a,s) for a in ARMS for s in STAGES]
    for arm,stage in methods:
        label=NAMES[arm]+("" if stage=="baseline" else " · "+STAGE_NAMES[stage])
        values=[]
        for split in ("validation","test"):
            v=primary[(primary.split==split)&(primary.arm==arm)&(primary.stage==stage)].score_percent
            values.append(f"{v.mean():.2f}%"+(f" ({v.min():.2f}~{v.max():.2f})" if len(v)>1 else ""))
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines += ["", "괄호는 세 model seed의 최소~최대이며 신뢰구간이 아니다. 50%를 모든 방법의 무작위 성능으로 해석하지 않는다. 동률만 발생하면 점수가 50%다.", ""]
    controls=table[table.arm!='summary14']
    if (controls.score_percent==100).all():
        lines += ["**실제 결과: 세 모델의 학습 전·20 epoch·확대 예산 모델과 원본 시계열 RMS가 모두 100%였다. 네 순서 변경 × 세 잡음 크기의 12조건에서도 동일했다.**", "",
            "이 검사는 순서 차이가 임베딩 거리에 남아 있음을 확인했지만, 학습 전 모델도 모두 풀 수 있을 만큼 쉬워 학습의 추가 가치를 구분하지 못했다. 학습 전후 차이는 모든 조건에서 0퍼센트포인트다. 시계열 학습이 요약 표현보다 유용한 연주 정보를 더 배웠다는 결론은 내릴 수 없다.", ""]
    lines += [
        "![주평가 비교](01_primary_comparison.png)", "", "## 해석", "",
        "- 요약 벡터는 A와 C가 같으므로 B를 더 가깝게 둘 수 없다. 이는 설계상 예상되는 결과이며 학습 모델의 우수성을 단독 입증하지 않는다.",
        "- 원본 시계열 RMS와 학습 전 모델도 높은 점수를 얻는지 함께 본다. 큰 순서 변경과 작은 잡음의 구분은 학습 없이도 가능하다.",
        "- 학습의 추가 기여는 같은 architecture·seed의 학습 전후 점수 차이로 확인한다. 높은 점수가 포화되면 학습 효과를 판별하기 어렵다.",
        "- 학습 후 점수가 떨어져도 다른 표현 능력 전체가 악화됐다고 해석하지 않는다. 복원 학습이 이 변형 구분 능력을 얼마나 유지했는지에 대한 결과다.", "",
        "![순서 변경 조건별 결과](02_order_conditions.png)", "", "![잡음 크기별 결과](03_noise_sensitivity.png)", "", "![학습 전후 차이](04_training_effect.png)", "",
        "## 데이터와 조건", "", "| split | 원래 작품/연주/구간 | 선택 작품/연주/구간 |", "|---|---:|---:|"]
    for s in protocol['support']:
        lines.append(f"| {s['split']} | {s['total_works']}/{s['total_performances']}/{s['total_windows']} | {s['selected_works']}/{s['selected_performances']}/{s['selected_windows']} |")
    lines += ["", "64 beat·7채널이 모두 유효한 구간만 선택하고, 연주별로 시간순 greedy 선택해 구간 겹침을 제외했다. 결측 위치를 붙이거나 padding을 재배치하지 않았다.",
        "학습 작품은 평가에 포함하지 않았다. 다만 validation과 기존 test는 이전 분석에서 이미 관찰한 작품이며 새로운 독립 holdout으로 주장하지 않는다.",
        "작품별 reference cohort·공통 패턴 제거·scale은 기존 정책을 유지한다. reference 없는 단독 새 연주 평가는 아니다.", "",
        "- 잡음 RMS: 원본 구간/채널 표준편차의 2%·5%·10%. 구간 내 평균을 제거하고 RMS를 맞춘 Gaussian 방향을 크기만 바꿔 사용한다. 상수 채널은 그대로 둔다.",
        "- 순서 변경: 전체 순열, 8-beat 블록 재배치, 역순, 이웃 beat 쌍 교환. 7채널을 같은 순열로 움직이므로 동시 채널 관계와 값 분포는 유지한다.",
        "- 초기·20 epoch 내 최선·확대 예산 최선: 기존 복원 validation MSE로 이미 선택한 checkpoint를 재사용한다.",
        "- 신경망 거리: 64D cosine. 기존 14D 요약 거리: 5 feature 동일 비중 RMS. 원본 시계열: 5 feature 동일 비중·64 beat RMS. 서로 다른 표현의 거리 수치를 직접 비교하지 않는다.",
        "- 구간마다 한 perturbation seed를 사용하며 모든 모델에 같은 변형을 제공한다. model seed 범위는 변형 seed에 대한 불확실성을 뜻하지 않는다.",
        "- 원본과 순서 변경본의 feature-weighted RMS 차이가 1e-10 이하이면 해당 구간·조건을 모든 방법에서 제외한다.", "",
        "## 한계와 다음 질문", "",
        "이는 시간 순서에 대한 기초 민감도 검사다. 순서 변경본은 실제 연주가 아니며, 더 멀게 배치하는 것이 모든 음악적 상황에서 바람직하다는 뜻도 아니다.",
        "특히 원본·초기 모델에서도 점수가 높다면 이 검사만으로 시계열 학습의 추가 가치를 입증하기 어렵다. 다음에는 실제 구간의 시간 특징을 고정 임베딩에서 읽어내는 검사와, 요약 성향이 유사한 실제 연주 간 비교가 필요하다.",
        "완전 유효 구간만 사용한 선택 편향이 있다. test에서 빠진 작품: " + ", ".join(next(s for s in protocol['support'] if s['split']=='test')['excluded_works']) + ".",
        "64D 구간을 평가했으므로 구간 평균·표준편차로 만든 128D 전곡 벡터의 시간 정보 보존은 별도 문제다.", "",
        "## 산출물과 재현", "",
        "- [summary.csv](summary.csv): split·방법·seed·순서 변경·잡음 조건별 작품 동일 비중 점수",
        "- [work_metrics.csv](work_metrics.csv): 작품별 점수·거리·지원 수",
        "- [training_effects.csv](training_effects.csv): 확대 예산 학습 후−학습 전 차이(퍼센트포인트)",
        "- [selected_windows.csv](selected_windows.csv), [example_beats.csv](example_beats.csv): 선택과 고정 사례",
        "- [protocol.json](protocol.json), [audit.json](audit.json), [layout_audit.json](layout_audit.json), [figure_manifest.json](figure_manifest.json), [report_generation.json](report_generation.json): 설정·hash·수치·레이아웃·최종 보고서 생성 기록",
        "- [independent_audit.json](independent_audit.json): 저장 입력·임베딩에서 모든 거리·점수·작품별 집계를 다시 계산한 독립 검증",
        "- 대용량 입력·순열·임베딩·구간별 평가: 저장소 밖 `Classicfy/datasets/temporal_order_evaluation_run01`",
        "", "저장소 루트에서 실행한다. 기존 출력이 있으면 덮어쓰지 않는다.", "", "```bash",
        "XDG_CACHE_HOME=/tmp/classicfy-cache MPLCONFIGDIR=/tmp/classicfy-mpl \\",
        "classicfy-ai/.venv/bin/python classicfy-ai/scripts/evaluate_temporal_order.py",
        "", "# 저장 CSV에서 그림과 보고서만 다시 생성", "classicfy-ai/.venv/bin/python classicfy-ai/scripts/evaluate_temporal_order.py --report-only",
        "", "# 모든 저장 거리·점수·집계를 독립 재계산", "classicfy-ai/.venv/bin/python classicfy-ai/scripts/audit_temporal_order.py", "```", "",
        "핵심 단위 검증 5개: 순열의 채널 묶음/분포 보존, 기존 요약 정의, 잡음 RMS·상수 채널, 동률·정의되지 않는 cosine, 결측·겹침 제외.", ""]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    render(parser.parse_args().out)
