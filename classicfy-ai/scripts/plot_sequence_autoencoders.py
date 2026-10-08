"""Figures and Korean explanations from the recorded sequence-model experiment.

No metric is inferred from an image. CSV/JSON tables are the source for every
plot, and every figure includes its population, direction and interpretation.
"""

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch

from validate_tempo import plt, setup_style
from analyze_temporal_failures import canvas, finish, SHORT, NAMES, sha
from embedding.data import CHANNELS

LABELS = {"cnn": "1D CNN", "bilstm": "BiLSTM", "transformer": "Transformer",
          "visible_mean": "보이는 값의 평균", "linear_interpolation": "직선 보간", "zero": "항상 0"}
COLORS = {"cnn": "#637585", "bilstm": "#2678B8", "transformer": "#CA6B36",
          "visible_mean": "#9273AE", "linear_interpolation": "#318C71", "zero": "#BBBBBB"}
SPECS = (("mse", "숨긴 값의 복원 오차", "MSE ↓", None),
         ("shape_correlation", "평균을 빼도 모양이 비슷한가?", "구간 중심 제거 상관 ↑", 0),
         ("shape_amplitude_ratio", "변화 폭을 얼마나 재현했나?", "예측/실제 표준편차 (%) · 100이 일치", 100),
         ("slope_mse", "인접 beat 변화량의 복원 오차", "변화량 MSE ↓", None),
         ("direction_percent", "상승·하강 방향을 맞혔나?", "방향 일치 (%) ↑", 50))


def piece_label(piece):
    parts = piece.removesuffix("/midi_score.mid").split("/")
    if parts[0] == "Bach":
        return f"Bach {parts[-1].replace('bwv_', 'BWV')} {parts[1]}"
    if parts[0] == "Beethoven":
        return f"Beethoven Sonata {parts[-1]}"
    if parts[0] == "Chopin" and parts[1].startswith("Etudes_op_"):
        return f"Chopin Op.{parts[1].split('_')[-1]} No.{parts[-1]}"
    return " / ".join(parts)


def range_plot(ax, table, models, metric, *, offset=0, label=None, color=None):
    scale = 100 if metric == "shape_amplitude_ratio" else 1
    for i, model in enumerate(models):
        values = table[table.model == model][metric].dropna().to_numpy() * scale
        if not len(values):
            continue
        mean = values.mean()
        ax.errorbar(i + offset, mean, yerr=[[mean-values.min()], [values.max()-mean]],
                    fmt="o", ms=8, capsize=5, lw=2, color=color or COLORS[model],
                    label=label if i == 0 else None)
        ax.annotate(f"{mean:.3f}" if metric in ("mse", "slope_mse", "shape_correlation") else f"{mean:.1f}%",
                    (i+offset, mean), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(np.arange(len(models)), [LABELS[m] for m in models], fontsize=11)
    ax.margins(x=.15, y=.24)


def architectures(out, models, runs):
    descriptions = {
        "cnn": "21채널 → 1×1 Conv → 64채널\nResidual Conv 4블록 · dilation 1/2/4/8\n주변 beat를 합성해 각 위치의 표현 생성",
        "bilstm": "21차원 → Linear → 64차원\n양방향 LSTM 2층 · 방향당 상태 32차원\n앞에서 읽은 32D + 뒤에서 읽은 32D",
        "transformer": "21차원 → Linear → 64차원 + 위치 인코딩\nAttention 2층 · 4 heads · FFN 128\n각 beat가 다른 유효 beat의 표현을 참조"}
    for model in models:
        directory = out / model
        directory.mkdir(exist_ok=True)
        fig, axes = canvas(f"{LABELS[model]} 오토인코더는 무엇을 입력받고 복원하는가?",
            "같은 입력 · 같은 64D 압축 벡터 · 같은 복원기 | 인코더 종류를 바꾸는 실험", size=(16, 10))
        ax = axes[0, 0]
        ax.set_axis_off()
        blocks = [("입력  [7채널 × 64 beat]", "템포·루바토·강약·아티큘레이션·페달 3채널\n유효 beat의 약 20%를 가리고 정답으로 보관", "#EAF0F3"),
                  ("모델에 전달  [21채널 × 64 beat]", "가린 값=0 + 유효 여부 7채널 + 가림 여부 7채널\n유효한 0 / 결측 / 일부러 가린 위치를 구별", "#EAF0F3"),
                  (f"{LABELS[model]} 인코더", descriptions[model], "#E4EFF7"),
                  ("순서 보존 압축 → 64차원 벡터", "64 beat × beat별 64차원을 채널 순서로 펼쳐 Linear(4096→64)\n모든 복원 정보는 이 벡터를 통과 · 우회 연결 없음", "#F6EDDF"),
                  ("동일한 복원기 → [7 × 64]", "Linear(64→128) → GELU → Linear(128→448)\n오차는 가렸으며 유효한 위치에서만 계산", "#E6F1EB")]
        for i, (title, body, color) in enumerate(blocks):
            y = .96 - i * .195
            ax.add_patch(FancyBboxPatch((.13, y-.17), .74, .17, boxstyle="round,pad=0.012", fc=color, ec="#CAD3D9", transform=ax.transAxes))
            ax.text(.5, y-.027, title, ha="center", va="top", fontsize=14, fontweight="bold", transform=ax.transAxes)
            ax.text(.5, y-.068, body, ha="center", va="top", fontsize=11, linespacing=1.25, transform=ax.transAxes)
            if i < 4:
                ax.annotate("", xy=(.5, y-.19), xytext=(.5, y-.175), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#7A8790", lw=2))
        count = next(r["parameters"] for r in runs if r["model"] == model)
        detail = "BiLSTM은 결측도 한 time step으로 남기고 마지막에 결측 위치 표현을 0으로 만듭니다." if model == "bilstm" else (
            "위치 인코딩: beat 번호별 고정 sin/cos 벡터. 결측은 Attention key에서 제외하되 가린 유효 beat는 유지합니다." if model == "transformer" else "CNN은 인과적 제한 없이 앞뒤 문맥을 읽습니다. 결측 위치의 출력 표현은 압축 전에 0으로 만듭니다.")
        finish(fig, directory / "00_architecture",
            f"파라미터 {count:,}개. 같은 seed에서는 CNN과 압축층·복원기의 초기 가중치도 동일합니다.\n{detail}\n"
            "앞뒤 문맥으로 숨긴 특징을 복원하는 모델이며 미래 예측 실험은 아닙니다. 임베딩은 모델이 학습한 구간의 숫자 표현입니다.", top=.86, bottom=.17)


def reading_diagram(out, models):
    fig, axes = canvas("CNN · BiLSTM · Transformer는 beat 순서를 어떻게 읽는가?",
        "인코더의 작동 방식 개념도 | 실제 Attention 가중치나 학습 결과를 시각화한 그림은 아닙니다", len(models), 1, (16, 4+3*len(models)))
    for ax, model in zip(axes.flat, models):
        ax.set_axis_off(); ax.set_xlim(-2.4, 10); ax.set_ylim(-.8, 1.4)
        ax.text(-2.3, .45, LABELS[model], fontsize=16, fontweight="bold", color=COLORS[model], va="center")
        x = np.arange(6)
        for i in x:
            ax.add_patch(FancyBboxPatch((i-.24, .24), .48, .4, boxstyle="round,pad=.03", fc="#F7E7D7" if i == 2 else "#E8EFF3", ec="#AAB7BF"))
            ax.text(i, .44, "가림" if i == 2 else str(i+1), ha="center", va="center", fontsize=11)
        if model == "bilstm":
            for i in range(5):
                ax.annotate("", (i+1, 1.0), (i, 1.0), arrowprops=dict(arrowstyle="->", color=COLORS[model], lw=2))
                ax.annotate("", (i, -.1), (i+1, -.1), arrowprops=dict(arrowstyle="->", color="#D28143", lw=2))
            explanation = "앞→뒤 상태와 뒤→앞 상태를 합칩니다.\n정답은 가려진 상태로 두 방향이 읽습니다."
        elif model == "transformer":
            for i in (0, 1, 3, 4, 5):
                ax.annotate("", (2, .8), (i, 1.1), arrowprops=dict(arrowstyle="->", color=COLORS[model], alpha=.7, lw=1.5))
            explanation = "각 beat가 다른 beat의 표현을 가중합합니다.\nbeat 번호를 나타내는 위치 벡터를 더합니다."
        else:
            for i in (1, 3):
                ax.annotate("", (2, .8), (i, 1.15), arrowprops=dict(arrowstyle="->", color=COLORS[model], lw=2))
            explanation = "주변 위치에 같은 필터를 적용합니다.\n간격을 벌린 필터와 여러 층으로 문맥을 넓힙니다."
        ax.text(6, .45, explanation, fontsize=11, va="center", linespacing=1.6)
    finish(fig, out / "00_encoder_reading",
        "숫자=원래 beat 위치, ‘가림’=복원 정답의 값은 모델 입력에서 0으로 숨김. 실제 입력은 64 beat × 7개 feature와 두 종류의 mask입니다.\n"
        "모든 모델은 앞뒤 관측 정보를 사용합니다. 이 개념도는 국소 처리·양방향 상태 전달·Attention의 차이를 설명하며, 특정 음악 구절 인식을 주장하지 않습니다.", top=.85, bottom=.13, hspace=.35)


def overview(out, summary, models):
    best = summary[(summary.split == "validation") & (summary.condition == "best")]
    displayed = [*models, "visible_mean", "linear_interpolation"]
    fig, axes = canvas("인코더를 바꾸면 값과 시간 흐름의 복원이 함께 좋아지는가?",
        "검증 8작품 · 93연주 · 935구간 | 검증 MSE로 선택한 checkpoint | 점=3 seed 평균 · 선=최소~최대", 2, 3, (20, 11))
    for ax, (metric, title, ylabel, reference) in zip(axes.flat, SPECS):
        range_plot(ax, best, displayed, metric)
        ax.set_title(title, loc="left", fontsize=14, pad=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        if reference is not None:
            ax.axhline(reference, color="#555", ls="--", lw=1)
        if metric == "direction_percent":
            ax.set_ylim(0, 100)
        if metric == "shape_amplitude_ratio":
            ax.set_ylim(bottom=0)
    axes.flat[-1].set_axis_off()
    model_best = best[best.model.isin(models)].groupby("model").mean(numeric_only=True)
    findings = "\n".join(f"{LABELS[m]} 값 MSE: {abs(100*(1-model_best.loc[m,'mse']/model_best.loc['cnn','mse'])):.1f}% "
        f"{'낮음' if model_best.loc[m,'mse'] < model_best.loc['cnn','mse'] else '높음'}" for m in models if m != "cnn")
    axes.flat[-1].text(.02, .96, f"이번 설정의 결과\n\n{findings}\n\n방향 일치와 변화량 오차는 별도 확인.\n\n평균 예측의 모양 상관은 정의되지 않아\n점이 없습니다.\n100% 변화 폭만으로 정확한 복원은 아닙니다.", va="top", fontsize=12, linespacing=1.5)
    finish(fig, out / "01_validation_comparison",
        "7채널 지표를 계산한 뒤 템포·루바토·강약·아티큘레이션·페달의 5그룹을 같은 비중으로 평균합니다. 페달 3채널은 한 그룹입니다.\n"
        "모양: 각 64-beat 구간에서 hidden 평균을 제거한 상관. 변화 폭: 같은 중심 제거 값의 표준편차 비율. 변화량: 두 인접 beat 모두 hidden인 경우.\n"
        "방향: 실제 변화 절댓값 >10⁻⁶인 쌍만 평가. 50% 선은 균등 무작위 부호의 기대값. seed 범위는 신뢰구간이 아닙니다.", top=.82, bottom=.19, hspace=.65)


def per_model(out, summary, channels, history, works, model):
    directory = out / model
    compare = ("cnn", model)
    names = ["cnn", model, "visible_mean", "linear_interpolation"]
    fig, axes = canvas(f"{LABELS[model]}는 학습하면서 얼마나 좋아졌는가?",
        "같은 학습 순서·가림 위치·20 epoch | 실선=검증 MSE · 점선=학습 batch 손실 · 3개 seed 각각 표시", 1, 2, (17, 8))
    for ax, name in zip(axes.flat, compare):
        for seed, group in history[history.model == name].groupby("seed"):
            ax.plot(group.epoch, group.validation_value_mse, label=f"검증 seed {str(seed)[-2:]}", lw=2)
            ax.plot(group.epoch, group.train_objective, ls="--", alpha=.5)
            i = group.validation_value_mse.idxmin()
            ax.scatter(group.loc[i, "epoch"], group.loc[i, "validation_value_mse"], marker="*", s=140, color="#D77731", zorder=5)
        ax.set(title=LABELS[name], xlabel="학습 epoch (0=학습 전)", ylabel="5 feature 동일 비중 MSE ↓")
        ax.legend(fontsize=10)
    finish(fig, directory / "01_learning",
        "★는 seed별 최선 검증 epoch. 최선 checkpoint가 주 비교 대상이고 20 epoch 최종값도 CSV에 보관합니다.\n"
        "학습 손실은 매 batch의 평균이며 가림이 바뀝니다. 검증은 고정 가림의 전체 합계로 계산하므로 곡선 높이를 그대로 일반화 차이로 해석하지 않습니다.", top=.80, bottom=.18)

    best = summary[(summary.split == "validation") & (summary.condition == "best")]
    fig, axes = canvas(f"{LABELS[model]}의 개선이 20 epoch 최종 모델에서도 유지되는가?",
        "검증 오차로 선택한 최선 vs 같은 학습 예산의 마지막 epoch | 점=3 seed 평균 · 선=최소~최대", 1, 2, (16, 8))
    for ax, metric, ylabel in zip(axes.flat, ("mse", "slope_mse"), ("값 MSE ↓", "변화량 MSE ↓")):
        for condition, offset, color, label in (("best", -.1, "#2678B8", "검증 최선"), ("final", .1, "#D7813B", "20 epoch 최종")):
            sub = summary[(summary.split == "validation") & (summary.condition == condition)]
            range_plot(ax, sub, compare, metric, offset=offset, label=label, color=color)
        ax.set_ylabel(ylabel); ax.legend()
    finish(fig, directory / "02_best_and_final", "모든 모델이 동일하게 20 epoch를 학습했습니다. 최선 epoch는 검증 값 MSE로만 고릅니다. 변화량 MSE나 test 성능으로 고르지 않습니다.", top=.80, bottom=.16)

    selected = channels[(channels.split == "validation") & (channels.condition == "best")]
    for number, metric, title, ylabel in ((3, "mse", "어느 feature의 숨긴 값을 더 잘 복원하는가?", "채널별 MSE ↓"),
            (4, "shape_correlation", "어느 feature의 구간 내 변화 모양을 따라가는가?", "구간 평균 제거 상관 ↑"),
            (5, "direction_percent", "어느 feature의 상승·하강을 맞히는가?", "방향 일치 (%) ↑")):
        fig, axes = canvas(f"{LABELS[model]} · {title}", "검증 전체 구간 | 막대=3 seed 평균 · 선=최소~최대 | baseline은 결정적 예측", size=(17, 9))
        ax = axes[0, 0]; x = np.arange(7); width = .19
        for ni, name in enumerate(names):
            table = selected[selected.model == name].groupby("channel")[metric].agg(["mean", "min", "max"]).reindex(CHANNELS)
            if table["mean"].isna().all():
                continue
            ax.bar(x+(ni-1.5)*width, table["mean"], width, color=COLORS[name], label=LABELS[name],
                yerr=np.array([table["mean"]-table["min"], table["max"]-table["mean"]]), capsize=3)
        ax.set_xticks(x, NAMES, fontsize=11); ax.set_ylabel(ylabel); ax.legend(ncol=2, fontsize=11)
        if metric == "shape_correlation":
            ax.axhline(0, color="#555", lw=1); ax.set_ylim(-.15, 1)
        if metric == "direction_percent":
            ax.axhline(50, color="#555", ls="--", label="무작위 50%"); ax.set_ylim(0, 105)
        finish(fig, directory / f"{number:02d}_{metric}",
            "페달 세 채널도 따로 표시합니다. 상관이 정의되지 않는 상수 평균 예측은 생략합니다.\n"
            "방향은 실제 변화가 0인 쌍을 제외하고 인접한 두 위치가 모두 hidden인 경우만 계산합니다. 값 오차와 시간 변화의 복원은 별도로 읽어야 합니다.", top=.82, bottom=.18)

    fig, axes = canvas(f"{LABELS[model]} · 관측 beat의 순서를 섞으면 복원이 달라지는가?",
        "숨긴 정답·가림 위치·관측 값의 집합은 고정 | 보이는 7채널 묶음을 같은 순열로 재배치", 1, 2, (16, 8))
    diagnostic = summary[(summary.split == "validation") & summary.model.isin(compare)]
    for ax, metric, ylabel in zip(axes.flat, ("mse", "shape_correlation"), ("값 MSE ↓", "구간 중심 제거 상관 ↑")):
        clean = diagnostic[diagnostic.condition == "best"]
        # Average the three shuffles inside each training seed before displaying
        # a range across training seeds: nine correlated observations != n=9.
        shuffled = diagnostic[diagnostic.condition.str.startswith("shuffle")].groupby(["model", "seed"], as_index=False)[metric].mean()
        range_plot(ax, clean, compare, metric, offset=-.12, color="#2678B8", label="원래 관측 순서")
        range_plot(ax, shuffled, compare, metric, offset=.12, color="#D7813B", label="관측 순서 섞기")
        ax.set_ylabel(ylabel); ax.legend(fontsize=10)
    finish(fig, directory / "06_order_diagnostic",
        "모델 학습 후 입력만 바꾼 진단입니다. 원래 관측 순서와 정답 위치의 관계를 깨뜨리며, 학습 때 보지 않은 입력일 수 있습니다.\n"
        "seed마다 3개 고정 순열의 지표를 평균한 뒤 3개 학습 seed의 최소~최대를 표시합니다. 순서를 쓰지 않는 모델과의 인과적 비교는 아닙니다.", top=.80, bottom=.19)

    fig, axes = canvas(f"{LABELS[model]} · 더 긴 구간을 가려도 복원되는가?",
        "검증 데이터에만 다른 가림을 적용 · 가리는 비율은 약 20% 유지 · 재학습 없음", 1, 2, (17, 8))
    for ax, metric, ylabel in zip(axes.flat, ("mse", "slope_mse"), ("값 MSE ↓", "변화량 MSE ↓")):
        for name in names:
            means, lows, highs = [], [], []
            for condition in ("gap1", "best", "gap8", "gap16"):
                vals = summary[(summary.split == "validation") & (summary.condition == condition) & (summary.model == name)][metric].dropna()
                means.append(vals.mean()); lows.append(vals.min()); highs.append(vals.max())
            means, lows, highs = map(np.array, (means, lows, highs))
            ax.errorbar(np.arange(4), means, yerr=[means-lows, highs-means], color=COLORS[name], label=LABELS[name], marker="o", capsize=4)
        ax.set_xticks(np.arange(4), ["1 beat", "4 beat\n학습 조건", "8 beat", "16 beat"])
        ax.set(xlabel="가림 생성기의 블록 크기 설정", ylabel=ylabel); ax.legend(fontsize=10)
    finish(fig, directory / "07_mask_blocks",
        "블록 설정은 실제 연속 가림 길이와 다릅니다. 선택한 블록이 합쳐지거나 결측·가림 목표 수에 의해 짧아질 수 있습니다.\n"
        "1-beat 설정도 우연히 인접한 beat를 가릴 수 있습니다. 모든 모델에 조건별 동일 mask를 적용하며, 조건 간 숨긴 위치 자체는 달라집니다.\n"
        "따라서 이 그림은 가림 생성 조건별 민감도 진단이며, 가림 길이만의 순수 효과를 분리한 실험은 아닙니다.", top=.80, bottom=.28)

    fig, axes = canvas(f"{LABELS[model]} · 개선이 여러 검증 작품에서 반복되는가?",
        "동일 seed·동일 작품의 CNN과 비교 | +는 MSE 감소, −는 증가 | 점=3 seed 평균 · 선=최소~최대", 1, 2, (18, 9))
    sub = works[(works.split == "validation") & (works.condition == "best") & works.model.isin(compare)]
    for ax, metric, title in zip(axes.flat, ("mse", "slope_mse"), ("값 복원", "인접 변화량 복원")):
        pivot = sub.pivot(index=["seed", "piece"], columns="model", values=metric)
        effect = (100*(1-pivot[model]/pivot.cnn)).groupby("piece").agg(["mean", "min", "max"])
        pos = np.arange(len(effect))
        ax.errorbar(effect["mean"], pos, xerr=[effect["mean"]-effect["min"], effect["max"]-effect["mean"]], fmt="o", capsize=4, color=COLORS[model])
        ax.set_yticks(pos, [piece_label(p) for p in effect.index], fontsize=10)
        ax.axvline(0, color="#555", ls="--"); ax.set(xlabel="CNN 대비 MSE 감소율 (%)", title=title)
    finish(fig, directory / "08_work_effects", "감소율 = 100 × (1 − 새 모델 MSE / 같은 seed CNN MSE). 각 작품에서 7채널을 5feature 동일 비중으로 요약합니다.\n작품별 구간 수가 달라 전체 pooled 결과와 작품 동일 비중 평균은 다를 수 있습니다. 같은 작품의 겹치는 구간을 독립 표본으로 세지 않습니다.", top=.82, bottom=.17, wspace=.75)


def examples(out, table, models):
    for model in models:
        if model == "cnn":
            continue
        directory = out / model / "examples"
        directory.mkdir(exist_ok=True)
        for number, group in table.groupby("example"):
            for channel in CHANNELS:
                g = group[group.channel == channel].sort_values("beat")
                fig, axes = canvas(f"사례 {number} · {LABELS[model]}는 {SHORT[CHANNELS.index(channel)]} 변화를 어떻게 복원했나?",
                    f"검증 고정 사례 · seed 20261006 | {g.key.iloc[0]}", size=(16, 8))
                ax = axes[0, 0]
                target = np.where(g.valid, g.target, np.nan)
                ax.plot(g.beat, target, color="#252B31", lw=1.8, label="실제 값 (숨긴 정답 포함)")
                for name in ("cnn", model, "linear_interpolation", "visible_mean"):
                    ax.plot(g.beat, np.where(g.hidden, g[name], np.nan), color=COLORS[name],
                        lw=2 if name == model else 1.3, marker="o", ms=4, label=LABELS[name])
                ax.scatter(g.beat[g.hidden], g.target[g.hidden], marker="x", s=40, color="#252B31", label="모델에 가린 정답", zorder=8)
                ax.set(xlabel="원래 score beat 구간 번호", ylabel="작품 공통 패턴을 제거하고 표준화한 feature")
                ax.legend(ncol=3, fontsize=10)
                finish(fig, directory / f"{int(number):02d}_{channel}",
                    "검정 선은 실제 곡선입니다. 색 선은 가린 위치의 예측만 표시하며 서로 떨어진 가림 구간을 연결하지 않습니다. ×가 복원 정답입니다.\n"
                    "사례는 검증 작품 이름순 첫·중간·마지막 작품, 중앙 순번 연주, 중앙에 가장 가까운 구간으로 선정합니다. 성능을 보고 고르지 않았습니다.", top=.80, bottom=.19)


def retrieval_plot(out, models):
    path = out / "retrieval_queries.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    fig, axes = canvas("복원이 좋아지면 다른 작품 검색도 시간 변화가 더 닮아지는가?",
        "기존 test 8작품 · 82연주를 query로 사용 | 같은 작품 제외 · cosine 최근접 1개 | feature 기반 관찰", size=(18, 9))
    ax = axes[0, 0]
    labels = {"representative": "대표 성향", "width": "변화 폭", "step_rms": "인접 beat 변화 크기"}
    metric = "closer_than_peer_share"
    for i, model in enumerate(models):
        sub = table[table.model == model]
        # Work macro first, then seed; do not let large works dominate.
        aggregate = sub.groupby(["seed", "descriptor", "query_piece"])[metric].mean().groupby(["seed", "descriptor"]).mean().mul(100).reset_index()
        mean = aggregate.groupby("descriptor")[metric].mean().reindex(labels)
        low = aggregate.groupby("descriptor")[metric].min().reindex(labels)
        high = aggregate.groupby("descriptor")[metric].max().reindex(labels)
        x = np.arange(3)+(i-(len(models)-1)/2)*.22
        ax.bar(x, mean, .22, color=COLORS[model], label=LABELS[model], yerr=[mean-low, high-mean], capsize=4)
        for xx, yy, hh in zip(x, mean, high):
            ax.text(xx, hh+1.8, f"{yy:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(np.arange(3), list(labels.values())); ax.set(ylabel="같은 후보 작품의 다른 연주보다 feature가 가까운 비율 (%) ↑", ylim=(0, 105))
    ax.axhline(50, color="#555", ls="--", label="후보 작품 내 무작위 선택 기대값 50%"); ax.legend(ncol=2)
    finish(fig, out / "04_retrieval_observation",
        "선택된 후보의 작품 안에서, 그 후보보다 query와 feature 거리가 먼 다른 연주의 비율을 계산합니다. 동률=0.5점. 예: 10명 중 9명보다 가까우면 90%.\n"
        "대표 성향=채널 평균(Rubato는 절댓값 중앙값). 변화 폭=p95−p5. 인접 변화 크기=원래 이웃한 유효 beat 차이의 RMS. 거리에는 5feature 동일 비중.\n"
        "query별 비율→작품별 평균→8작품 평균→seed 평균. 범위는 3 seed 최소~최대. 청취 정답에 대한 정확도가 아니며 변화 순서의 유사성도 측정하지 않습니다.", top=.82, bottom=.21)


def resource_and_test(out, models, runs, summary):
    fig, axes = canvas("같은 학습 예산에서 모델 크기와 실행 비용은 어떻게 다른가?",
        "CPU · 2 threads · 20 epoch · batch 32 | 막대=3 seed 평균 · 선=최소~최대", 1, 2, (16, 8))
    for ax, key, title, ylabel in zip(axes.flat, ("parameters", "seconds"), ("학습 가능한 파라미터", "20 epoch 학습 시간"), ("파라미터 수", "초")):
        for i, model in enumerate(models):
            values = np.array([r[key] for r in runs if r["model"] == model])
            ax.bar(i, values.mean(), color=COLORS[model], yerr=[[values.mean()-values.min()], [values.max()-values.mean()]], capsize=4)
            ax.text(i, values.max()*1.02, f"{values.mean():,.0f}", ha="center", fontsize=11)
        ax.set_xticks(np.arange(len(models)), [LABELS[m] for m in models]); ax.set(title=title, ylabel=ylabel); ax.margins(y=.2)
    finish(fig, out / "03_resources", "CNN은 이전 통제 실험의 저장 모델과 시간을 재사용합니다. 실행 날짜·시스템 부하가 달라 시간은 참고값이며 엄밀한 동시 벤치마크가 아닙니다.\n임베딩 크기(64D)와 복원기(66,112개 파라미터)는 동일하지만 인코더 파라미터 수는 다릅니다. 이 비교는 동일 파라미터 예산 실험이 아닙니다.", top=.80, bottom=.18)
    if not (summary.split == "test").any():
        return
    fig, axes = canvas("검증에서 고른 모델은 기존 test 작품에서도 개선되는가?",
        "모든 설정을 고정한 뒤 평가 | 파랑=검증 · 주황=test | 점=3 seed 평균 · 선=최소~최대", 1, 2, (16, 8))
    for ax, metric, ylabel in zip(axes.flat, ("mse", "slope_mse"), ("값 MSE ↓", "변화량 MSE ↓")):
        for split, offset, color, label in (("validation", -.1, "#2678B8", "검증 8작품·935구간"), ("test", .1, "#D7813B", "test 8작품·1670구간")):
            sub = summary[(summary.split == split) & (summary.condition == "best")]
            range_plot(ax, sub, models, metric, offset=offset, color=color, label=label)
        ax.set_ylabel(ylabel); ax.legend(fontsize=10)
    finish(fig, out / "02_validation_and_test", "test는 checkpoint 선택·모델 설정 변경에 사용하지 않습니다. 이미 이전 CNN 분석에서 관찰한 동일 test split이며 새로운 독립 holdout은 아닙니다.\n두 split의 작품·분포가 다르므로 검증보다 test MSE가 낮다는 사실 자체를 더 좋은 일반화의 근거로 삼지 않습니다.", top=.80, bottom=.19)


def report(out, models, summary, works, runs):
    best = summary[(summary.split == "validation") & (summary.condition == "best")]
    table = best.groupby("model")[[s[0] for s in SPECS]].mean()
    lines = []
    for model in (*models, "visible_mean", "linear_interpolation"):
        r = table.loc[model]
        corr = "정의 안 됨" if pd.isna(r.shape_correlation) else f"{r.shape_correlation:.3f}"
        lines.append(f"| {LABELS[model]} | {r.mse:.4f} | {corr} | {100*r.shape_amplitude_ratio:.1f}% | {r.slope_mse:.4f} | {r.direction_percent:.1f}% |")
    introduction = []
    for model in models:
        if model == "cnn":
            continue
        pivot = best[best.model.isin(("cnn", model))].pivot(index="seed", columns="model", values="mse")
        gain = 100*(1-pivot[model]/pivot.cnn)
        movement = "감소" if gain.mean() > 0 else "증가"
        introduction.append(f"- **{LABELS[model]}**: 같은 seed의 CNN 대비 검증 값 MSE 평균 **{abs(gain.mean()):.2f}% {movement}** "
            f"(세 seed의 감소율 범위 {gain.min():+.2f}~{gain.max():+.2f}%; 음수=악화).")
        r = table.loc[model]
        introduction[-1] += f" 모양 상관 **{r.shape_correlation:.3f}**, 변화 폭 **{100*r.shape_amplitude_ratio:.1f}%**, 방향 일치 **{r.direction_percent:.1f}%**."
    conclusions = []
    if "bilstm" in models:
        r, cnn = table.loc["bilstm"], table.loc["cnn"]
        conclusions.append(f"BiLSTM은 값 오차와 모양 상관({cnn.shape_correlation:.3f}→{r.shape_correlation:.3f})이 개선됐지만, "
            f"변화 폭은 {100*r.shape_amplitude_ratio:.1f}%에 머물고 방향 일치는 CNN {cnn.direction_percent:.1f}%보다 낮은 {r.direction_percent:.1f}%입니다. "
            "값 복원의 소폭 개선을 beat별 움직임 복원 성공으로 확대해 해석하지 않습니다.")
    if "transformer" in models:
        r, cnn = table.loc["transformer"], table.loc["cnn"]
        conclusions.append(f"Transformer는 이번 설정에서 CNN보다 값 오차가 크고 모양 상관도 낮습니다({cnn.shape_correlation:.3f}→{r.shape_correlation:.3f}). "
            f"변화 폭도 {100*r.shape_amplitude_ratio:.1f}%로 더 작습니다. 변화량 MSE가 약간 낮아도, "
            "상수 예측의 변화량 MSE가 더 낮다는 사실을 함께 보면 그 숫자만으로 실제 움직임을 잘 복원했다고 할 수 없습니다.")
    model_links = "\n".join(f"- [{LABELS[m]} 구현·실험 설명]({m}/README.md)" for m in models if m != "cnn")
    report_title = "시계열 오토인코더: 1D CNN · BiLSTM · Transformer 비교" if "transformer" in models else "BiLSTM 먼저 완료한 검증 비교 (Transformer 학습 전)"
    sequence_text = "BiLSTM의 세 seed 학습과 검증 figure를 먼저 완료한 뒤 Transformer를 실행했습니다." if "transformer" in models else "이 보고서는 Transformer 실행 전 BiLSTM 학습과 검증 결과를 담은 중간 기록입니다."
    report_text = f"# {report_title}\n\n" + "\n".join(introduction) + f"""

이 결과는 아래 고정 설정에서의 비교입니다. 모델 계열 전체의 우열을 뜻하지 않습니다.
{sequence_text}

## 먼저 볼 그림

![인코더가 순서를 읽는 방법](00_encoder_reading.png)

![값과 흐름 복원 비교](01_validation_comparison.png)

위 그림에서 값 MSE 감소와 시간 변화 복원을 따로 확인하세요. 평균에 가까운 값을 잘 맞혀도
급격한 변화·방향·변화 폭이 잘 복원된다고 단정할 수 없습니다.

""" + "\n\n".join(conclusions) + "\n\n" + model_links + """

## 실제 검증 결과

3개 seed 평균입니다. 표준편차 비율의 목표는 100%이며 높을수록 무조건 좋은 지표가 아닙니다.

| 모델 | 값 MSE ↓ | 모양 상관 ↑ | 예측/실제 변화 폭 | 변화량 MSE ↓ | 방향 일치 ↑ |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(lines) + """

## 무엇을 구현했는가?

기존 beat별 7채널 feature, 64-beat 구간, 64차원 임베딩, Linear 복원기를 유지하고 인코더만 바꿨습니다.
학습 정답은 입력에서 가린 feature 값입니다. 연주 취향 라벨이나 유사도 라벨은 사용하지 않았습니다.
앞뒤 문맥을 사용하는 masked autoencoder이며 미래를 예측하는 모델은 아닙니다.

- **BiLSTM**: 양방향 LSTM 2층. 방향마다 hidden 32개를 연결해 각 beat를 64차원으로 표현합니다.
- **Transformer**: 폭 64, Attention 4 heads, 2층, FFN 폭 128, pre-LayerNorm, GELU.
  원래 beat 번호를 구분하도록 고정 sin/cos 위치 벡터를 더합니다. causal mask는 사용하지 않습니다.
- **공통 압축층**: 각 위치의 표현을 순서대로 펼쳐 4096→64 Linear로 압축합니다.
- **공통 복원기**: 64→128→448 Linear, 중간 GELU. 임베딩을 우회하는 연결은 없습니다.

PyTorch 공식 [LSTM 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)와
[TransformerEncoderLayer 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)를 기준으로 구현했습니다.

## 비교 조건과 데이터

원래 CNN의 동일 데이터·분할을 사용합니다. train 39작품·395연주·5803구간,
validation 8작품·93연주·935구간, test 8작품·82연주·1670구간입니다.
작품 단위 split이며 64-beat 구간, stride 32입니다. 결측 beat를 삭제하거나 이웃으로 붙이지 않습니다.
이번 실험은 ASAP/nASAP의 기존 cohort이며 추가 ATEPP 데이터를 섞지 않았습니다.

입력은 가린 값을 0으로 만든 7채널 + 유효 여부 7채널 + 가림 여부 7채널입니다.
약 20%의 유효 beat를 블록 설정 4로 가립니다. 실제 블록 길이는 결측·겹침·마지막 선택에 따라 다릅니다.
가린 정답과 결측의 값을 바꿔도 출력이 변하지 않는 테스트를 수행했습니다.
BiLSTM은 결측을 한 step으로 남깁니다. Transformer는 결측을 Attention key에서 제외합니다.
LSTM의 hidden state(내부 상태)와 hidden target(가린 정답)은 서로 다른 의미입니다.

세 학습 seed는 20261006/07/08입니다. 모든 모델은 AdamW, lr 0.001, batch 32,
dropout 0.1, gradient clipping 1, 20 epoch를 사용합니다. 학습 batch 순서와 가림의 SHA-256이
같은 seed의 CNN과 정확히 일치하는지 확인합니다. 검증 가림은 모든 모델·seed에서 고정합니다.
같은 seed에서 압축층·복원기의 초기 가중치도 동일합니다.
검증 값 MSE가 최소인 checkpoint로 비교하고 마지막 20 epoch 결과도 별도로 제공합니다.

작품 공통 패턴 제거·scale은 기존 방식입니다. 검증/test도 해당 작품의 전체 reference 연주를
포함해 정규화합니다. reference 없이 단독 새 MIDI를 처리하는 상황과는 다릅니다.

## 지표를 읽는 법

1. **값 MSE**: 가렸으며 유효한 위치에서 (예측−실제)² 평균. 낮을수록 좋습니다.
2. **모양 상관**: 각 64-beat 구간에서 hidden 값의 평균을 실제·예측 각각에서 제거한 후 Pearson 상관.
   떨어진 여러 hidden 블록이 한 구간에 있으면 함께 중심을 제거합니다. 각 연속 블록별 상관과 다릅니다.
   구간의 평균적인 높이만 맞추는 효과를 줄입니다. 상수 평균 예측은 정의되지 않아 생략합니다.
3. **변화 폭 비율**: 같은 중심 제거 값의 예측 표준편차 / 실제 표준편차 ×100.
   100%는 크기가 같은 것이며, 같은 모양·방향이라는 뜻은 아닙니다.
4. **변화량 MSE**: 인접한 두 beat가 모두 hidden이고 유효할 때 Δ예측과 Δ실제의 MSE.
   관측 정답을 그대로 붙여 경계 변화가 좋아 보이는 효과를 제외합니다.
5. **방향 일치**: 같은 hidden 인접 쌍 중 |Δ실제|>10⁻⁶인 쌍에서 상승·하강 부호가 같은 비율.
   실제 변화 0은 제외하고 예측 변화 0은 정답 부호와 다르면 실패입니다. 50%는 균등 무작위 부호의 기대값입니다.

전체 요약은 채널별 지표를 구한 뒤 Tempo/Rubato/Dynamics/Articulation/Pedaling 5그룹을 같은 비중으로
평균합니다. 페달 세 채널은 평균해서 한 그룹으로 둡니다. 작품별 결과는 별도 CSV·그림으로 제공합니다.
**seed 최소~최대는 신뢰구간이 아닙니다.** 같은 연주의 겹치는 구간을 독립 샘플로 간주하지 않습니다.

## 실험 순서와 figure 구성

각 모델 폴더에는 구조 → 학습 → best/final → 채널별 값/모양/방향 → 관측 순서 섞기 → 가림 블록 조건 → 작품별 비교가 있습니다.
검증 사례는 작품 이름순 첫·중간·마지막 작품의 중앙 순번 연주, 중앙 구간으로 고정했습니다.
각 사례의 7채널을 모두 표시하며, 좋은 사례만 고르지 않았습니다.

순서 섞기는 학습 후 입력 변형 진단입니다. 세 고정 순열을 seed 안에서 평균한 후 seed 간 범위를 표시합니다.
순서 없는 모델을 따로 학습한 비교가 아니므로 시간 순서 사용의 인과적 증명은 아닙니다.
가림 블록 실험은 설정 1/4/8/16에서 같은 약 20%를 가립니다. 조건별 정답 위치도 달라지므로
길이만의 효과를 분리하지 않습니다. 검증 데이터에만 이 진단을 적용합니다.

## 모델 크기와 실행 비용

![크기와 시간](03_resources.png)

압축 벡터와 복원기는 동일하지만 인코더 파라미터 수는 다릅니다. 파라미터 수를 맞춘 실험은 아닙니다.
CNN은 이전 저장 모델을 재사용합니다. 실행 날짜·부하가 달라 학습 시간은 참고값입니다.

## 범위와 한계

한 작품 split, 3개 seed, 20 epoch, 하나의 optimizer 설정입니다. 각 모델에 최적인 설정을 탐색한 것은 아닙니다.
음악적으로 관련된 작품군 전체가 완전히 분리됐다고 보장하지 않습니다.
복원 개선만으로 추천 품질·사람이 느끼는 연주 해석 유사성을 입증하지 않습니다.
전곡 검색 벡터는 구간 임베딩 평균과 표준편차를 연결한 128D이며 구간 간의 전곡 배치 순서를 잃습니다.
여기서 ‘변화 폭’ 등은 그림에 적힌 통계량이며 새로운 음악 이론 개념이나 성향 정답을 정의한 것이 아닙니다.

## 산출물과 재현

- `summary.csv`: 모델·seed·split·진단 조건별 요약
- `channel_metrics.csv`: 7채널 지표와 평가 target/인접 쌍 수
- `work_metrics.csv`: 작품별 결과
- `history.csv`, `training_runs.json`: 학습 곡선·선택 epoch·규모·시간·schedule hash
- `example_beats.csv`: 모든 고정 사례의 실제·예측 수치
- `protocol.json`: 입력 hash·소스 hash·설정·분할 규모·한계
- figure는 PNG와 SVG를 모두 제공합니다. 체크포인트·임베딩 NPZ는 저장소 밖 datasets에 둡니다.

저장소 루트에서 실행합니다. 완료된 실행은 protocol과 checkpoint hash가 일치할 때만 재사용합니다.

```bash
XDG_CACHE_HOME=/tmp/classicfy-cache MPLCONFIGDIR=/tmp/classicfy-mpl \\
classicfy-ai/.venv/bin/python classicfy-ai/scripts/compare_sequence_autoencoders.py --stage all
```

기존 캐시는 ATEPP 전용 수집 파일이 추가되기 전에 생성됐습니다. 원래 reference commit의 feature·preprocessing
파일 **전체가 현재 파일과 byte 단위로 일치**하는지 먼저 검사하고, 원래 수집기 파일 목록으로 원래 소스 hash와
데이터 revision을 검증했습니다. 캐시나 provenance를 임의 수정하지 않았습니다.
"""
    if (summary.split == "test").any():
        report_text += """

## 고정 설정의 기존 test 결과와 검색 관찰

![검증과 test](02_validation_and_test.png)

test는 검증으로 checkpoint를 고른 뒤 고정 설정으로 평가했습니다. 이미 이전 CNN 분석에 사용한 test split이므로
새 독립 holdout으로 주장하지 않습니다. test 결과를 본 뒤 이번 모델 설정을 조정하지 않았습니다.

![검색의 feature 일치](04_retrieval_observation.png)

검색 후보의 작품 안에서 다른 연주보다 얼마나 feature가 가까운지 평가합니다. 예를 들어 후보 B가 같은 작품의
다른 연주 10명 중 9명보다 query A에 가까우면 90%입니다. 같은 거리에는 0.5점을 줍니다.
대표 성향=채널 평균(Rubato는 절댓값 중앙값), 변화 폭=p95−p5, 인접 변화 크기=차이의 RMS입니다.
query별 비율을 작품별 평균한 뒤 8개 query 작품을 같은 비중으로 평균합니다.
청취 정답, 검색 정확도, 시간 변화의 순서 일치를 측정한 수치가 아닙니다.
후보는 train/validation/test를 포함한 전체 570연주에서 query와 같은 작품의 연주를 제외합니다.
검색 시 특징을 가리지 않으며 dropout을 끕니다. 학습 때와 같은 reference cohort를 사용하는 검색 관찰입니다.
`neighbors.csv`는 모델·seed별 Top 5와 query/candidate split을, `retrieval_queries.csv`는 query별 비교 근거를 제공합니다.
"""
        if (out / "bilstm_first_validation/README.md").exists():
            report_text += "\n[Transformer 시작 전에 완료한 BiLSTM 검증 기록](bilstm_first_validation/README.md)도 보존했습니다. 동일 실험의 중간 기록이며 추가 독립 실험으로 세지 않습니다.\n"
    (out / "README.md").write_text(report_text, encoding="utf-8")
    for model in models:
        if model == "cnn":
            continue
        r = table.loc[model]
        cnn = table.loc["cnn"]
        gain = 100*(1-r.mse/cnn.mse)
        change_gain = 100*(1-r.slope_mse/cnn.slope_mse)
        work_table = works[(works.split == "validation") & (works.condition == "best") & works.model.isin(("cnn", model))]
        pivot = work_table.pivot(index=["piece", "seed"], columns="model", values="mse")
        improvements = ((pivot[model] < pivot.cnn).groupby("piece").mean() == 1).sum()
        model_runs = [v for v in runs if v["model"] == model]
        selected_epochs = ", ".join(f"{v['seed']}: {v['best_value_epoch']}" for v in model_runs)
        mechanism = ("""BiLSTM은 LSTM 두 개로 같은 구간을 앞→뒤와 뒤→앞으로 읽습니다. 각 LSTM의 hidden state는
지금까지 읽은 정보를 요약하는 벡터이고, cell state는 게이트로 정보를 유지·지우며 전달하는 내부 기억입니다.
이번 구현은 각 방향 32차원의 출력을 합쳐 beat마다 64차원으로 만듭니다. 첫 층의 출력 전체를 둘째 층이 다시 읽습니다.
한 방향의 마지막 state만 쓰지 않고 모든 beat의 출력을 순서대로 펼쳐 압축하므로 위치별 정보를 유지합니다.
뒤에서 읽는 방향도 **관측된 입력과 mask만** 읽습니다. 숨긴 정답을 읽는 것이 아닙니다.
오른쪽 padding과 중간 결측을 삭제하지 않아서 원래 beat 간 간격이 유지됩니다. 다만 결측 step도 상태 전달에 영향을 줄 수 있습니다.
""" if model == "bilstm" else """Transformer의 self-attention은 각 beat의 표현에서 query(Q), key(K), value(V)를 만들고,
Q와 다른 beat의 K가 얼마나 맞는지로 가중치를 구해 V를 가중합하는 연산입니다. 이 연산을 4개 head가 나누어 수행합니다.
head는 입력 표현의 다른 부분을 학습하는 경로이며, 특정 head가 템포나 페달 같은 음악 개념을 담당한다고 지정하지 않았습니다.
FFN은 각 위치에 적용하는 64→128→64 신경망이고, LayerNorm은 각 위치의 표현을 정규화합니다.
pre-LayerNorm은 Attention/FFN 전에 정규화하는 배치입니다. encoder layer 2개를 순서대로 통과합니다.
위치 인코딩은 beat 번호별 sin/cos 숫자 벡터입니다. Attention만으로는 입력 순서를 직접 구별하지 못하므로 위치 정보를 더합니다.
결측 위치는 key에서 제외하고 유효한 hidden beat는 남깁니다. 그 위치의 실제 값은 0으로 가려져 있으며 mask를 통해 가림 사실만 알 수 있습니다.
마지막에 64위치 전체를 순서대로 펼쳐 압축합니다. causal mask를 쓰지 않아 앞뒤 관측 문맥을 모두 볼 수 있습니다.
""")
        diagnostics = summary[(summary.model == model) & (summary.split == "validation")]
        shuffled_mse = diagnostics[diagnostics.condition.str.startswith("shuffle")].groupby("seed").mse.mean().mean()
        mask_table = []
        for condition, label in (("gap1", "1"), ("best", "4 (학습 조건)"), ("gap8", "8"), ("gap16", "16")):
            current = diagnostics[diagnostics.condition == condition].mean(numeric_only=True)
            mask_table.append(f"| {label} | {current.mse:.4f} | {current.slope_mse:.4f} |")
        movement = "감소" if gain > 0 else "증가"
        change_movement = "감소" if change_gain > 0 else "증가"
        content = f"""# {LABELS[model]} 구현과 실험 결과

**검증 값 MSE {r.mse:.4f}, CNN {cnn.mse:.4f}. 평균 MSE의 비율로 보면 {abs(gain):.2f}% {movement}입니다.**
인접 변화량 MSE는 {r.slope_mse:.4f} (CNN 대비 {abs(change_gain):.2f}% {change_movement}), 모양 상관 {r.shape_correlation:.3f},
예측/실제 변화 폭 {100*r.shape_amplitude_ratio:.1f}%, 방향 일치 {r.direction_percent:.1f}%입니다.
음수 감소율은 악화를 뜻합니다. 모델 종류 전체의 성능을 단정하지 않습니다.

![구현 구조](00_architecture.png)

구조의 숫자는 이번 구현 설정입니다. 64D 벡터와 복원기는 CNN과 동일합니다.
전체 지표 정의·입력·동일 조건·한계는 [공통 실험 설명](../README.md)에 있습니다.

{mechanism}

## 학습이 실제로 수행됐는가?

![학습 곡선](01_learning.png)

세 seed 모두 실제 학습·저장·재로드를 수행했습니다. seed별 선택 epoch: **{selected_epochs}**.
가린 정답 누출 방지, 결측·padding 처리, 인코더 gradient, 저장 후 동일 출력 테스트를 통과했습니다.

![최선과 최종](02_best_and_final.png)

## 값 오차와 시간 변화는 각각 어떻게 달라졌는가?

![채널별 값 오차](03_mse.png)
![채널별 모양](04_shape_correlation.png)
![채널별 방향](05_direction_percent.png)

값 MSE 감소는 평균적인 수준을 잘 맞춘 결과일 수도 있습니다. 모양·방향·변화 폭을 함께 읽어야 합니다.
변화 폭 비율은 100%가 크기의 일치일 뿐, 모양이 같다는 뜻이 아닙니다.

## 시간 순서와 가림 조건을 바꾸면?

![관측 순서 섞기](06_order_diagnostic.png)
![가림 생성 블록](07_mask_blocks.png)

순서 섞기는 추론 중 관측 값과 원래 위치의 관계를 깨는 진단입니다. 학습 분포 밖의 입력일 수 있습니다.
블록 크기 설정은 실제 gap 길이와 다르며, 조건별 숨긴 위치도 달라집니다. 이 두 실험을 인과적 증명으로 해석하지 않습니다.

원래 순서의 값 MSE는 **{r.mse:.4f}**, 세 순열을 seed 안에서 평균한 순서 섞기 MSE는 **{shuffled_mse:.4f}**입니다.
가림 생성 설정별 3 seed 평균은 다음과 같습니다.

| 블록 설정 (beat) | 값 MSE ↓ | 변화량 MSE ↓ |
|---|---:|---:|
""" + "\n".join(mask_table) + f"""

## 일부 작품에만 맞는 결과인가?

![작품별 비교](08_work_effects.png)

검증 8작품 중 **{improvements}작품**에서 세 seed 모두 CNN보다 값 MSE가 작았습니다.
전체 pooled 결과와 작품 동일 비중 평균은 구간 수 차이로 달라질 수 있습니다.

## 고정된 실제 복원 사례

검정=실제, ×=숨긴 정답, 색 선=가린 위치의 예측입니다. 서로 다른 gap을 이어 그리지 않습니다.
작품 이름순 첫·중간·마지막 작품의 중앙 연주·중앙 구간으로 고정했으며 성능을 보고 선택하지 않았습니다.

| 사례 | Tempo | Rubato | Dynamics | Articulation | Pedal depth | Pedal ratio | Pedal changes |
|---|---|---|---|---|---|---|---|
"""
        for number in (1, 2, 3):
            content += f"| {number} | " + " | ".join(f"[figure](examples/{number:02d}_{c}.png)" for c in CHANNELS) + " |\n"
        content += "\n![고정 사례의 Tempo 복원](examples/01_tempo.png)\n\n나머지 20개 사례 figure도 위 표에서 확인할 수 있습니다.\n"
        (out / model / "README.md").write_text(content, encoding="utf-8")


def render(out, models):
    out = Path(out)
    setup_style()
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 11})
    summary = pd.read_csv(out / "summary.csv")
    channels = pd.read_csv(out / "channel_metrics.csv")
    history = pd.read_csv(out / "history.csv")
    works = pd.read_csv(out / "work_metrics.csv")
    runs = json.loads((out / "training_runs.json").read_text())
    architectures(out, models, runs)
    reading_diagram(out, models)
    overview(out, summary, models)
    resource_and_test(out, models, runs, summary)
    for model in models:
        if model != "cnn":
            per_model(out, summary, channels, history, works, model)
    examples(out, pd.read_csv(out / "example_beats.csv"), models)
    retrieval_plot(out, models)
    report(out, models, summary, works, runs)
    manifest = {"plotting_source_sha256": sha(Path(__file__)),
                "figures": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*.png"))},
                "tables": {p.name: sha(p) for p in sorted(out.glob("*.csv"))}}
    (out / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "render_environment.json").write_text(json.dumps({"python": sys.version,
        "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "figure_backend": plt.get_backend()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["cnn", "bilstm", "transformer"])
    args = parser.parse_args()
    render(args.out, args.models)
