"""Diagnose a saved CNN by local shape, actual mask length, and train/validation.

Read-only model evaluation. Figures and machine-readable results only; no README
generation, retraining, checkpoint selection, or test-set tuning.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.data import CHANNELS, WindowDataset, restore_sequences
from embedding.evaluation import baseline_predictions, temporal_metrics
from embedding.training import TrainingConfig, fixed_masks, load_model
from validate_embedding import collect_groups
from validate_feature_normalization import read_cache
from validate_temporal_embedding import predictions, feature_average
from validate_tempo import setup_style, plt


NAMES = ("Tempo\n빠르기", "Rubato\n속도 편차", "Dynamics\n강약", "Articulation\n건반 유지",
         "Pedal depth\n페달 깊이", "Pedal ratio\n페달 사용 비율", "Pedal changes\n페달 전환")
SHORT = ("Tempo", "Rubato", "Dynamics", "Articulation", "Pedal depth", "Pedal ratio", "Pedal changes")
PATTERNS = ("small", "gradual", "abrupt", "mixed")
PATTERN_NAMES = ("작은 변화", "완만한 한 방향", "급격한 변화", "방향 전환·혼합")
METHODS = ("cnn", "visible_mean", "linear_interpolation")
METHOD_NAMES = {"cnn": "CNN", "visible_mean": "보이는 값의 평균", "linear_interpolation": "직선 보간"}
COLORS = {"cnn": "#2478C4", "visible_mean": "#8F6AAE", "linear_interpolation": "#D98324"}
SPLIT_COLORS = {"train": "#2A8D78", "validation": "#D26B41"}
SPLIT_NAMES = {"train": "학습에 쓴 연주", "validation": "검증 연주"}
TOLERANCE = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def runs(mask):
    """Half-open true runs, preserving invalid gaps and original beat indices."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)))


def classify(span, max_step, monotonicity, thresholds):
    if span <= thresholds["range_q25"]:
        return "small"
    if max_step > thresholds["step_q75"]:
        return "abrupt"
    if monotonicity >= .8:
        return "gradual"
    return "mixed"


def collect_gaps(split, dataset, arrays):
    target, valid, hidden, by_method = arrays
    rows = []
    for wi, (si, window_start) in enumerate(dataset.items):
        sequence = dataset.sequences[si]
        for ci, channel in enumerate(CHANNELS):
            for start, stop in runs(hidden[wi, ci]):
                two_sided = (start > 0 and stop < dataset.window_size
                             and valid[wi, ci, start - 1] and valid[wi, ci, stop]
                             and not hidden[wi, ci, start - 1] and not hidden[wi, ci, stop])
                truth = target[wi, ci, start:stop].astype(float)
                local = target[wi, ci, start - 1:stop + 1].astype(float) if two_sided else None
                delta = np.diff(truth)
                moving = np.abs(delta) > TOLERANCE
                row = {"split": split, "window": wi, "piece": sequence.piece, "key": sequence.key,
                       "channel": channel, "channel_index": ci, "start": start, "stop": stop,
                       "beat": window_start + start, "length": stop - start,
                       "length_bin": str(stop - start) if stop - start <= 4 else "5+",
                       "two_sided": bool(two_sided), "targets": len(truth), "pairs": len(delta),
                       "moving_pairs": int(moving.sum()), "actual_step_ss": float(np.sum(delta ** 2))}
                if two_sided:
                    local_delta = np.diff(local)
                    variation = np.abs(local_delta).sum()
                    row.update(span=float(np.ptp(local)), max_step=float(np.max(np.abs(local_delta))),
                               monotonicity=float(abs(local[-1] - local[0]) / variation) if variation > TOLERANCE else 1.)
                else:
                    row.update(span=np.nan, max_step=np.nan, monotonicity=np.nan)
                for method, prediction in by_method.items():
                    estimate = prediction[wi, ci, start:stop].astype(float)
                    predicted_delta = np.diff(estimate)
                    row.update({f"{method}_sse": float(np.sum((estimate - truth) ** 2)),
                                f"{method}_step_sse": float(np.sum((predicted_delta - delta) ** 2)),
                                f"{method}_step_ss": float(np.sum(predicted_delta ** 2)),
                                f"{method}_direction_hits": int(np.sum(np.sign(predicted_delta[moving]) == np.sign(delta[moving])))})
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame, keys):
    result = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        identifiers = dict(zip(keys, values))
        counts = {name: int(group[name].sum()) for name in ("targets", "pairs", "moving_pairs")}
        actual_step_ss = group.actual_step_ss.sum()
        mean_mse = group.visible_mean_sse.sum() / counts["targets"]
        mean_step_mse = group.visible_mean_step_sse.sum() / counts["pairs"] if counts["pairs"] else np.nan
        for method in METHODS:
            mse = group[f"{method}_sse"].sum() / counts["targets"]
            step_mse = group[f"{method}_step_sse"].sum() / counts["pairs"] if counts["pairs"] else np.nan
            result.append({**identifiers, "method": method, "gaps": len(group),
                           "works": group.piece.nunique(), "performances": group.key.nunique(), **counts,
                           "mse": mse, "rmse": np.sqrt(mse), "step_mse": step_mse,
                           "value_gain_percent": 100 * (1 - mse / mean_mse) if mean_mse > 1e-12 else np.nan,
                           "step_gain_percent": 100 * (1 - step_mse / mean_step_mse) if mean_step_mse > 1e-12 else np.nan,
                           "step_amplitude_percent": 100 * np.sqrt(group[f"{method}_step_ss"].sum() / actual_step_ss) if actual_step_ss > 1e-12 else np.nan,
                           "direction_percent": 100 * group[f"{method}_direction_hits"].sum() / counts["moving_pairs"] if counts["moving_pairs"] else np.nan})
    return pd.DataFrame(result)


def canvas(title, subtitle, nrows=1, ncols=1, size=(15, 9)):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.suptitle(title, x=.055, y=.97, ha="left", fontsize=23, fontweight="bold", color="#23333D")
    fig.text(.055, .925, subtitle, fontsize=12, color="#56616A", va="top")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.5)
    return fig, axes


def finish(fig, path, note, top=.84, bottom=.13, hspace=.6, wspace=.28, right=.965):
    fig.text(.055, .035, note, fontsize=10.5, color="#56616A", va="bottom", linespacing=1.55)
    fig.subplots_adjust(left=.085, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)
    fig.savefig(path.with_suffix(".png"), dpi=165)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def heatmap(ax, matrix, labels, *, limit=60):
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(4), PATTERN_NAMES, fontsize=11)
    ax.set_yticks(np.arange(7), SHORT, fontsize=11)
    for y in range(7):
        for x in range(4):
            value = matrix[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:+.1f}%\n{labels[y, x]}", ha="center", va="center", fontsize=10,
                        color="white" if abs(value) > .6 * limit else "#243039")
    return image


def pattern_figures(table, length_table, out):
    sub = table[(table.split == "validation") & (table.method == "cnn")]
    fig, axes = canvas("어떤 변화에서 CNN이 평균 예측보다 나아지는가?",
                       "검증 데이터 · 가린 구간의 양끝이 보이는 경우만 분류 · +는 CNN 개선, −는 평균 예측이 우세", size=(16, 9))
    matrix = np.full((7, 4), np.nan)
    labels = np.full((7, 4), "", dtype=object)
    for ci, channel in enumerate(CHANNELS):
        for pi, pattern in enumerate(PATTERNS):
            row = sub[(sub.channel == channel) & (sub.pattern == pattern)]
            if len(row):
                row = row.iloc[0]
                matrix[ci, pi] = row.value_gain_percent
                labels[ci, pi] = f"{int(row.gaps):,}구간"
    ax = axes[0, 0]
    im = heatmap(ax, matrix, labels)
    fig.colorbar(im, ax=ax, fraction=.035, pad=.025, label="평균 예측 대비 값 MSE 감소율 (%)", extend="both")
    finish(fig, out / "02_feature_by_pattern",
           "작은 변화: 학습 구간의 값 범위 하위 25% 이하. 급격한 변화: 최대 인접 변화가 학습 구간 상위 25% 초과.\n"
           "나머지 중 한 방향 변화 비율 ≥80%는 ‘완만한 한 방향’, 그 외는 ‘방향 전환·혼합’. 실제 정답으로 사후 분류하며 예측에는 사용하지 않음.", top=.84)

    fig, axes = canvas("상대 개선율과 별개로, 실제 오차는 얼마나 큰가?",
                       "검증 데이터 · CNN의 값 RMSE · 같은 행에서 유형별 비교 · 낮을수록 실제 값에 가까움", size=(16, 9))
    errors = np.full((7, 4), np.nan)
    for ci, channel in enumerate(CHANNELS):
        for pi, pattern in enumerate(PATTERNS):
            row = sub[(sub.channel == channel) & (sub.pattern == pattern)]
            if len(row):
                errors[ci, pi] = row.iloc[0].rmse
    ax = axes[0, 0]
    im = ax.imshow(errors, cmap="YlOrBr", aspect="auto", vmin=0, vmax=np.nanmax(errors))
    ax.grid(False)
    ax.set_xticks(np.arange(4), PATTERN_NAMES)
    ax.set_yticks(np.arange(7), SHORT)
    for ci in range(7):
        for pi in range(4):
            value = errors[ci, pi]
            if np.isfinite(value):
                ax.text(pi, ci, f"{value:.2f}", ha="center", va="center", fontsize=14,
                        color="white" if value > .65*np.nanmax(errors) else "#29353D")
    fig.colorbar(im, ax=ax, fraction=.035, pad=.025, label="CNN 값 RMSE")
    finish(fig, out / "02b_absolute_error_by_pattern",
           "정규화된 상대 feature 단위이며 채널별 분포는 다름. 유형별 구간 수와 평균 대비 개선율은 02 figure에서 확인.\n"
           "예: 페달 전환의 ‘작은 변화’는 평균 대비 개선율이 음수여도, 절대 오차 자체는 급격한 변화보다 작을 수 있음.")

    fig, axes = canvas("가린 길이가 길어지면 복원이 더 어려운가?",
                       "검증 데이터 · 실제 연속 hidden 구간 길이 · 아래 숫자는 구간 수 · 같은 가림에 세 방법을 적용", 2, 4, (17, 10))
    bins = ("1", "2", "3", "4", "5+")
    for ci, channel in enumerate(CHANNELS):
        ax = axes.flat[ci]
        for method in METHODS:
            sub = length_table[(length_table.split == "validation") & (length_table.channel == channel) & (length_table.method == method)]
            values = sub.set_index("length_bin").reindex(bins)
            ax.plot(np.arange(5), values.rmse, "o-", lw=2, ms=5, color=COLORS[method], label=METHOD_NAMES[method])
        counts = values.gaps.fillna(0).astype(int)
        ax.set_xticks(np.arange(5), [f"{b} beat\n(n={n:,})" for b, n in zip(bins, counts)], fontsize=9)
        ax.set_title(SHORT[ci], fontsize=14, loc="left", fontweight="bold")
        ax.set_ylabel("값 RMSE", fontsize=10)
        ax.set_ylim(bottom=0)
    axes.flat[7].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[7].legend(handles, labels, loc="upper left", frameon=False, fontsize=13)
    axes.flat[7].text(0, .52, "낮을수록 실제 값에 가까움\n\n5+는 가림 블록이 붙어서 생긴 구간\n채널마다 세로축 범위가 다름", transform=axes.flat[7].transAxes, fontsize=12, linespacing=1.7, va="top")
    finish(fig, out / "03_actual_mask_length",
           "길이별 곡선은 난이도·작품 구성도 달라지는 관찰 결과이며, 길이만 바꾼 통제 실험은 아님.\n"
           "RMSE = 숨긴 값의 평균 제곱 오차에 제곱근을 취한 값. 모든 값은 정규화된 상대 feature 단위.", top=.82, bottom=.16, hspace=.62)


def train_validation_figures(overall, matched, out):
    fig, axes = canvas("학습에 쓴 연주에서도 흐름을 놓치는가?",
                       "같은 18 epoch 모델 · eval 모드(dropout 끔) · 같은 20% 가림 규칙 · split마다 고정 마스크", 2, 2, (17, 11))
    specifications = (("value_gain_percent", "가린 값: 평균 예측보다 얼마나 개선됐나?", "값 MSE 감소율 (%)", 0),
                      ("step_gain_percent", "인접 변화량: 평균 예측보다 얼마나 개선됐나?", "변화량 MSE 감소율 (%)", 0),
                      ("step_amplitude_percent", "인접 변화 크기: 실제의 몇 %를 예측했나?", "예측 변화 RMS / 실제 변화 RMS (%)", 100),
                      ("direction_percent", "인접 변화 방향: 상승·하강을 얼마나 맞췄나?", "방향 일치 (%)", 50))
    x = np.arange(7)
    for ax, (metric, title, ylabel, reference) in zip(axes.flat, specifications):
        for split, offset in (("train", -.18), ("validation", .18)):
            data = overall[(overall.split == split) & (overall.method == "cnn")].set_index("channel").reindex(CHANNELS)
            ax.bar(x + offset, data[metric], .34, color=SPLIT_COLORS[split], label=SPLIT_NAMES[split])
        ax.axhline(reference, color="#555", ls="--", lw=1)
        ax.set_title(title, fontsize=14, loc="left", fontweight="bold", pad=12)
        ax.set_xticks(x, NAMES, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        if metric == "direction_percent":
            ax.set_ylim(0, 100)
        elif metric == "step_amplitude_percent":
            ax.set_ylim(0, max(110, ax.get_ylim()[1]))
        ax.legend(fontsize=10, frameon=False, loc="upper right")
    finish(fig, out / "04_train_vs_validation",
           "상단: 0%보다 높아야 평균 예측보다 좋음. 좌하단: 실제 변화 크기는 100%. 우하단: 50%는 무작위 방향 비교선.\n"
           "인접 변화는 두 beat 모두 숨긴 유효 위치에서만 측정. 실제 변화가 0인 쌍은 방향 평가에서 제외.\n"
           "학습·검증의 작품과 feature 분포가 다르므로 두 집단의 차이만으로 모델 용량 부족·과적합을 확정할 수 없음.", top=.83, bottom=.18, hspace=.57)

    fig, axes = canvas("같은 변화 유형·4 beat 가림에서도 차이가 나는가?",
                       "양끝이 보이는 정확히 4 beat 구간만 비교 · 색: 평균 예측 대비 값 MSE 감소율 · 셀의 숫자: 감소율 / 구간 수", 1, 2, (18, 9))
    for ax, split in zip(axes.flat, ("train", "validation")):
        matrix = np.full((7, 4), np.nan)
        labels = np.full((7, 4), "", dtype=object)
        for ci, channel in enumerate(CHANNELS):
            for pi, pattern in enumerate(PATTERNS):
                row = matched[(matched.split == split) & (matched.channel == channel) & (matched.pattern == pattern) & (matched.method == "cnn")]
                if len(row):
                    row = row.iloc[0]
                    matrix[ci, pi], labels[ci, pi] = row.value_gain_percent, f"n={int(row.gaps):,}"
        im = heatmap(ax, matrix, labels)
        ax.set_title(SPLIT_NAMES[split], loc="left", fontsize=15, fontweight="bold", pad=15)
    colorbar_axis = fig.add_axes([.925, .16, .014, .67])
    fig.colorbar(im, cax=colorbar_axis, label="평균 대비 MSE 감소율 (%)", extend="both")
    finish(fig, out / "05_train_vs_validation_matched",
           "변화 유형의 기준은 학습 데이터에서 계산해 검증에도 그대로 적용. 흰 셀은 해당 구간 없음.\n"
           "길이와 유형을 맞춰도 작품·변화의 세기까지 같아지는 것은 아님. 색 범위 밖의 값은 같은 끝 색으로 표시하고 실제 숫자를 병기.", top=.83, right=.88)


def choose_examples(gaps):
    """Choose median ground-truth descriptors, never prediction success/failure."""
    chosen = []
    for split in ("train", "validation"):
        for channel in CHANNELS:
            for pattern in PATTERNS:
                subset = gaps[(gaps.split == split) & (gaps.channel == channel) & (gaps.pattern == pattern) & (gaps.length >= 2)]
                exact = subset[subset.length == 4]
                if len(exact):
                    subset = exact
                if subset.empty:
                    continue
                descriptors = subset[["span", "max_step", "monotonicity"]]
                scale = descriptors.quantile(.75) - descriptors.quantile(.25)
                distance = ((descriptors - descriptors.median()).abs() / scale.replace(0, 1)).sum(axis=1)
                row = subset.assign(selection_distance=distance).sort_values(["selection_distance", "key", "beat", "window"]).iloc[0].to_dict()
                chosen.append(row)
    return chosen


def plot_examples(chosen, datasets, arrays, out):
    directory = out / "examples"
    directory.mkdir(exist_ok=True)
    records = []
    for ci, channel in enumerate(CHANNELS):
        fig, axes = canvas(f"{SHORT[ci]}: 실제 흐름과 CNN 예측을 직접 비교",
                           "왼쪽: 학습에 쓴 연주 / 오른쪽: 검증 연주 · 회색 영역: 가린 위치 · 검정: 실제 값 · 파랑: CNN", 4, 2, (17, 15))
        for pi, pattern in enumerate(PATTERNS):
            for sj, split in enumerate(("train", "validation")):
                ax = axes[pi, sj]
                candidates = [r for r in chosen if r["split"] == split and r["channel"] == channel and r["pattern"] == pattern]
                if not candidates:
                    ax.axis("off")
                    ax.text(.05, .5, "해당 유형 없음", transform=ax.transAxes)
                    continue
                row = candidates[0]
                wi, start, stop = (int(row[k]) for k in ("window", "start", "stop"))
                dataset = datasets[split]
                si, window_start = dataset.items[wi]
                sequence = dataset.sequences[si]
                target, valid, hidden, by_method = arrays[split]
                # Only show the hidden run and its visible anchors; no hidden truth
                # from other runs is passed off as observed context.
                lo, hi = start - 1, stop + 1
                beats = window_start + np.arange(lo, hi)
                actual = target[wi, ci, lo:hi]
                ax.axvspan(window_start + start - .45, window_start + stop - .55, color="#E8ECEF", zorder=0)
                ax.plot(beats, actual, "o-", color="#27333B", lw=1.8, ms=4, label="실제 값")
                ax.scatter(beats[[0, -1]], actual[[0, -1]], s=85, color="#27333B", marker="s", zorder=5, label="보이는 양끝")
                for method in METHODS:
                    ax.plot(window_start + np.arange(start, stop), by_method[method][wi, ci, start:stop],
                            "o-" if method == "cnn" else "--", color=COLORS[method],
                            lw=2 if method == "cnn" else 1.4, ms=5, label=METHOD_NAMES[method])
                title = f"{PATTERN_NAMES[pi]} · {SPLIT_NAMES[split]} · {stop-start} beat 가림"
                ax.set_title(title, fontsize=12, loc="left", fontweight="bold", pad=22)
                key = sequence.key.replace("/midi_score.mid", "").replace("Piano_Sonatas/", "Sonata/")
                ax.text(0, 1.025, key, transform=ax.transAxes, fontsize=8.2, color="#657078")
                ax.set_xticks(beats)
                ax.set_xlabel("원래 beat 구간 번호", fontsize=9)
                ax.set_ylabel("정규화된 상대 값", fontsize=9)
                for b in range(start, stop):
                    records.append({"split": split, "channel": channel, "pattern": pattern, "key": sequence.key,
                                    "window": wi, "beat": window_start + b, "target": float(target[wi, ci, b]),
                                    **{m: float(by_method[m][wi, ci, b]) for m in METHODS}})
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .889), ncol=5, frameon=False, fontsize=11)
        finish(fig, directory / f"{ci+1:02d}_{channel}",
               "각 유형·split에서 실제 변화의 범위·최대 변화·한 방향 비율이 중앙에 가까운 사례를 선정. 4 beat 가림 우선.\n"
               + "예측 오차를 선정 기준에 사용하지 않음. 이 그림은 사례이며 전체 성능은 집계 figure로 판단.", top=.82, bottom=.105, hspace=.88)
    pd.DataFrame(records).to_csv(out / "example_beats.csv", index=False)


def introductory_figure(chosen, datasets, arrays, out):
    fig, axes = canvas("무엇을 가리고, 어떤 변화의 복원을 확인했나?",
                       "실제 검증 Dynamics 사례 · 각 패널의 가운데 4 beat를 가림 · 검정은 정답, 파랑은 CNN 예측", 1, 4, (17, 6.8))
    target, valid, hidden, by_method = arrays["validation"]
    dataset = datasets["validation"]
    for ax, pattern, name in zip(axes.flat, PATTERNS, PATTERN_NAMES):
        row = next(r for r in chosen if r["split"] == "validation" and r["channel"] == "dynamics" and r["pattern"] == pattern)
        wi, start, stop = (int(row[k]) for k in ("window", "start", "stop"))
        si, offset = dataset.items[wi]
        x = offset + np.arange(start-1, stop+1)
        ax.axvspan(offset+start-.45, offset+stop-.55, color="#E8ECEF", zorder=0)
        ax.plot(x, target[wi, 2, start-1:stop+1], "o-", color="#27333B", lw=2, ms=5, label="실제 값")
        ax.scatter(x[[0,-1]], target[wi, 2, [start-1, stop]], marker="s", s=70, color="#27333B", zorder=4, label="보이는 양끝")
        ax.plot(offset+np.arange(start,stop), by_method["cnn"][wi, 2, start:stop], "o-", color=COLORS["cnn"], lw=2, label="CNN 예측")
        ax.set_title(name, loc="left", fontsize=15, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xlabel("원래 beat 구간 번호", fontsize=10)
        ax.set_ylabel("정규화된 강약 값", fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.065,.82), ncol=3, frameon=False)
    finish(fig, out / "01_reading_the_diagnostics",
           "모델은 이 양끝뿐 아니라 64 beat 안에서 보이는 7개 채널 전체를 입력으로 사용. 그림은 한 가림 구간만 확대.\n"
           "변화 유형은 실제 곡선으로 사후 분류하며, 학습 데이터에서 정한 기준을 검증에도 적용.\n"
           "사례는 각 유형에서 실제 변화 특성이 중앙에 가까운 구간으로 선정. 모델이 잘 맞춘/못 맞춘 정도로 고르지 않음.", top=.70, bottom=.26, wspace=.33)


def source_audit(gaps, arrays, datasets, args, provenance):
    """Re-extract worst local failures; agreement proves consistency, not musical truth."""
    from preprocessing import ASAPLoader, load_midi, load_match
    from features import extract_dynamics, extract_pedaling, extract_articulation, extract_note_articulation

    records = {r["key"]: r for r in read_cache(args.cache, provenance)}
    loader = ASAPLoader(args.asap_root, args.nasap_root)
    checks, details = [], []
    for ci, channel in enumerate(CHANNELS):
        subset = gaps[(gaps.split == "validation") & gaps.two_sided & gaps.length.between(2, 4)].copy()
        subset = subset[subset.channel == channel]
        subset["gap_mse"] = subset.cnn_sse / subset.targets
        row = subset.sort_values(["gap_mse", "key", "beat"], ascending=[False, True, True]).iloc[0]
        sample = loader.get_sample(row.key)
        cached = records[row.key]
        lo, hi = int(row.beat) - 1, int(row.beat) + int(row.length) + 1
        beat_indices = np.arange(lo, hi)
        annotation_dt = np.diff(sample.performance_beats)
        np.testing.assert_array_equal(annotation_dt, cached["beat_seconds"])
        recomputed = None
        note_counts = None
        if channel in ("tempo", "rubato"):
            raw = np.log2(np.diff(sample.score_beats)[beat_indices] / annotation_dt[beat_indices])
            check_type = "score/performance annotation durations"
        else:
            midi = load_midi(sample.performance_path)
            if channel == "dynamics":
                feature = extract_dynamics(midi, sample.performance_beats)
                recomputed, note_counts = feature.sequence, feature.onset_counts
            elif channel == "articulation":
                alignment = load_match(sample.note_alignment_path)
                feature = extract_articulation(alignment, sample.performance_beats)
                recomputed, note_counts = feature.sequence, feature.note_counts
            else:
                feature = extract_pedaling(midi, sample.performance_beats)
                field = {"pedal_depth": "depth", "pedal_down_ratio": "down_ratio", "pedal_changes": "changes"}[channel]
                recomputed = getattr(feature, field)
            np.testing.assert_allclose(recomputed.values[beat_indices], cached[f"{channel}_values"][beat_indices], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(recomputed.mask[beat_indices], cached[f"{channel}_mask"][beat_indices])
            raw = recomputed.values[beat_indices]
            check_type = "original MIDI/match re-extraction versus raw cache"
        entry = {"channel": channel, "key": row.key, "piece": row.piece, "beat": int(row.beat),
                 "length": int(row.length), "pattern": row.pattern, "gap_mse": row.gap_mse,
                 "check_type": check_type, "cache_agrees": True, "robust_alignment": sample.robust_note_alignment,
                 "performance_midi_sha256": sha(sample.performance_path),
                 "match_sha256": sha(sample.note_alignment_path) if sample.note_alignment_path else None,
                 "context_beats": beat_indices.tolist(), "raw_values": raw.tolist(),
                 "beat_seconds": annotation_dt[beat_indices].tolist(),
                 "note_counts": note_counts[beat_indices].tolist() if note_counts is not None else None}
        if channel == "articulation":
            note_values = extract_note_articulation(alignment)
        for beat in beat_indices:
            a, b = sample.performance_beats[beat:beat + 2]
            item = {"channel": channel, "key": row.key, "beat": int(beat), "start_seconds": a, "end_seconds": b,
                    "raw": float(raw[beat - lo]), "hidden": int(row.beat) <= beat < int(row.beat) + int(row.length)}
            if channel == "dynamics":
                item["velocities"] = [n.velocity for n in midi.notes if a <= n.start < b]
            elif channel == "articulation":
                item["matched_notes"] = [{"log2_ratio": float(value),
                    "held_seconds": alignment.matches[index][1].offset - alignment.matches[index][1].onset,
                    "score_duration_beats": alignment.matches[index][0].offset_beats - alignment.matches[index][0].onset_beats}
                    for index, onset, value in zip(note_values.match_indices, note_values.onsets, note_values.values) if a <= onset < b]
            elif channel.startswith("pedal_"):
                prior = [p for p in midi.pedals if p.time < a]
                item["cc64_before"] = prior[-1].value if prior else 0
                item["cc64_events"] = [[p.time, p.value] for p in midi.pedals if a <= p.time < b]
            details.append(item)
        checks.append(entry)
    (args.out / "source_checks.json").write_text(json.dumps({"selection": "Largest per-gap CNN MSE per channel, validation, two visible anchors, length 2..4",
        "limit": "Re-extraction consistency and robust alignment flags do not certify musical validity or annotation accuracy.",
        "checks": checks, "events": details}, ensure_ascii=False, indent=2, default=json_scalar), encoding="utf-8")
    return checks


def source_figure(checks, arrays, datasets, out):
    fig, axes = canvas("오차가 가장 큰 구간: 원본 입력부터 다시 확인",
                       "검증 채널별 최대 오차 사례 · 2~4 beat 가림 / 양끝 관측 · 원본 MIDI·match 또는 annotation과 재대조", 2, 4, (18, 10))
    for ci, entry in enumerate(checks):
        ax = axes.flat[ci]
        x = np.array(entry["context_beats"])
        ax.plot(x, entry["raw_values"], "o-", color="#26343D", lw=2)
        ax.axvspan(entry["beat"]-.45, entry["beat"]+entry["length"]-.55, color="#E8ECEF", zorder=0)
        ax.set_title(SHORT[ci], loc="left", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("원래 beat 구간 번호", fontsize=9)
        ax.set_xticks(x)
        ax.tick_params(labelsize=9)
        ax.set_ylabel("log₂ 악보/연주 시간비" if ci < 2 else "추출 원시값", fontsize=10)
        note = (f"기여 음 수: {entry['note_counts']}" if entry["note_counts"] is not None else "annotation 시간 일치" if ci < 2 else "CC64 재추출 일치")
        ax.text(.01, .98, note, va="top", transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    axes.flat[7].axis("off")
    axes.flat[7].text(0, .98, "확인한 것\n\n5개 MIDI/match 채널: 캐시와 재추출 일치\nTempo/Rubato: 원본 beat 시간 일치\n7개 사례: robust note alignment\n\n아직 판단하지 못하는 것\n\n변화의 음악적 타당성\n원본 annotation 자체의 정확성",
                      transform=axes.flat[7].transAxes, va="top", fontsize=12, linespacing=1.55)
    finish(fig, out / "06_largest_errors_source_check",
           "극단적인 실패 원인을 점검하기 위해 최대 오차를 의도적으로 선정한 그림. 평균적인 사례가 아님.\n"
           "Tempo/Rubato 패널은 둘 다 원본 Tempo 비율이며, Rubato 잔차나 CNN 입력값이 아님. 나머지 패널도 공통 제거·정규화 전 값.\n"
           "원자료 일치는 캐시 손상·계산 불일치 검사이며 추출 설계의 음악적 타당성을 보증하지 않음. 상세 경로·이벤트: source_checks.json.", top=.82, bottom=.18, hspace=.55)


def main():
    ai = Path(__file__).resolve().parents[1]
    data = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=data / "temporal_embedding_cluster_run01")
    parser.add_argument("--cache", type=Path, default=data / "feature_normalization_cluster_raw.npz")
    parser.add_argument("--asap-root", type=Path, default=data / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=data / "nASAP")
    parser.add_argument("--out", type=Path, default=ai / "analysis/temporal_diagnostics")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive threads required")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new/empty output directory")
    torch.set_num_threads(args.threads)
    args.asap_root, args.nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    stored = json.loads((args.run / "dataset.json").read_text())
    info = json.loads((args.run / "training.json").read_text())
    if sha(args.cache) != stored["cache_sha256"]:
        parser.error("Cache hash differs from saved training run")
    print("Rebuilding provenance-checked normalized inputs", flush=True)
    groups, audits, scales, provenance = collect_groups(SimpleNamespace(asap_root=args.asap_root,
        nasap_root=args.nasap_root, cache=args.cache, **stored["cohort_settings"]))
    if provenance != stored["source_provenance"] or scales != stored["scales"]:
        raise ValueError("Source provenance or scales differ from training run")
    sequences = restore_sequences(groups, audits)
    config = TrainingConfig(**info["training_config"])
    model, checkpoint = load_model(args.run / "best.pt")
    if checkpoint["epoch"] != info["best_epoch"]:
        raise ValueError("Checkpoint epoch differs from training metadata")
    args.out.mkdir(parents=True)
    datasets, arrays, frames, whole_metrics = {}, {}, [], []
    for split in ("train", "validation"):
        dataset = WindowDataset([s for s in sequences if stored["split"][s.piece] == split],
            window_size=info["model_config"]["window_size"], stride=stored["stride"])
        datasets[split] = dataset
        values, valid = zip(*[(v, m) for v, m, _ in DataLoader(dataset, batch_size=config.batch_size, shuffle=False)])
        values, valid = torch.cat(values), torch.cat(valid)
        masks = fixed_masks(dataset, config)
        target, validity, hidden = values.numpy(), valid.numpy(), masks.numpy()
        by_method = baseline_predictions(target, validity, hidden)
        by_method.pop("zero")
        by_method["cnn"] = predictions(model, values, valid, masks, batch_size=config.batch_size)
        arrays[split] = (target, validity, hidden, by_method)
        frames.append(collect_gaps(split, dataset, arrays[split]))
        for method in METHODS:
            for ci, channel in enumerate(CHANNELS):
                whole_metrics.append({"split": split, "method": method, "channel": channel,
                    **temporal_metrics(target[:, ci], by_method[method][:, ci], hidden[:, ci])})
        print(f"{split}: {len(dataset)} windows, {int(hidden[:, 0].sum())} hidden beats", flush=True)
    val_mse = feature_average([r for r in whole_metrics if r["split"] == "validation" and r["method"] == "cnn"], "mse")
    np.testing.assert_allclose(val_mse, info["best_validation"]["loss"], rtol=1e-6)
    gaps = pd.concat(frames, ignore_index=True)
    thresholds = {}
    for channel in CHANNELS:
        reference = gaps[(gaps.split == "train") & (gaps.channel == channel) & gaps.two_sided]
        thresholds[channel] = {"range_q25": float(reference.span.quantile(.25)), "step_q75": float(reference.max_step.quantile(.75))}
    gaps["pattern"] = [classify(r.span, r.max_step, r.monotonicity, thresholds[r.channel]) if r.two_sided else "unanchored" for r in gaps.itertuples()]
    overall = summarize(gaps, ["split", "channel"])
    pattern = summarize(gaps[gaps.two_sided], ["split", "channel", "pattern"])
    lengths = summarize(gaps, ["split", "channel", "length_bin"])
    matched = summarize(gaps[gaps.two_sided & (gaps.length == 4)], ["split", "channel", "pattern"])
    work = summarize(gaps, ["split", "piece", "channel"])
    work_pattern = summarize(gaps[gaps.two_sided], ["split", "piece", "channel", "pattern"])
    for filename, table in (("overall_metrics", overall), ("pattern_metrics", pattern), ("length_metrics", lengths),
                            ("matched_four_beat_metrics", matched), ("work_metrics", work), ("work_pattern_metrics", work_pattern),
                            ("window_centered_metrics", pd.DataFrame(whole_metrics))):
        table.to_csv(args.out / f"{filename}.csv", index=False)
    for split in ("train", "validation"):
        for ci, channel in enumerate(CHANNELS):
            row = overall[(overall.split == split) & (overall.channel == channel) & (overall.method == "cnn")].iloc[0]
            ref = next(r for r in whole_metrics if r["split"] == split and r["channel"] == channel and r["method"] == "cnn")
            np.testing.assert_allclose([row.mse, row.step_mse, row.direction_percent / 100], [ref["mse"], ref["slope_mse"], ref["direction_accuracy"]], rtol=1e-10)
    chosen = choose_examples(gaps)
    pd.DataFrame(chosen).to_csv(args.out / "selected_examples.csv", index=False)
    print("Rendering local failures and train/validation figures", flush=True)
    setup_style()
    plt.rcParams.update({"axes.titlesize": 14, "svg.fonttype": "path", "font.size": 11})
    pattern_figures(pattern, lengths, args.out)
    train_validation_figures(overall, matched, args.out)
    plot_examples(chosen, datasets, arrays, args.out)
    introductory_figure(chosen, datasets, arrays, args.out)
    checks = source_audit(gaps, arrays, datasets, args, provenance)
    source_figure(checks, arrays, datasets, args.out)
    manifest = {"run": str(args.run.resolve()), "best_epoch": checkpoint["epoch"],
        "checkpoint_sha256": sha(args.run / "best.pt"), "cache_sha256": sha(args.cache),
        "source_provenance": provenance, "script_sha256": sha(__file__),
        "mask": {"seed": config.seed + 2, "fraction": config.mask_fraction, "requested_block_size": config.mask_block_size,
                 "note": "Actual adjacent blocks can merge to >4 beats. Same fixed_masks procedure used separately per split. One fixed realization."},
        "evaluation": "Saved best model, eval mode, no retraining, no test use; target-pooled per channel; overlapping windows are not independent samples.",
        "splits": {split: {"works": len({s.piece for s in dataset.sequences}), "performances": len(dataset.sequences), "windows": len(dataset)} for split, dataset in datasets.items()},
        "validation_mse_reproduced": val_mse, "shape_thresholds_train_only": thresholds,
        "shape_definition": {"context": "one visible valid beat on each side of a maximal consecutive hidden run",
            "small": "range <= channel train 25th percentile", "abrupt": "not small and max absolute adjacent change > channel train 75th percentile",
            "gradual": "neither above; abs(endpoint difference)/sum(abs(adjacent changes)) >= 0.8", "mixed": "remaining anchored gaps",
            "limit": "Operational diagnostics in normalized residuals, not musical phrase or articulation labels; thresholds include overlapping windows."},
        "source_checks_limit": "Re-extraction agreement is consistency only, not proof that MIDI/annotation/features reflect musical intent.",
        "unanchored_gaps": gaps[gaps.channel == "tempo"].groupby("split").two_sided.agg(["size", "sum"]).to_dict(),
        "figure_files": sorted(str(p.relative_to(args.out)) for p in args.out.rglob("*.png")),
        "source_audit_passed": all(c["cache_agrees"] for c in checks)}
    (args.out / "stats.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_scalar), encoding="utf-8")
    print(overall[overall.method == "cnn"][["split", "channel", "value_gain_percent", "step_gain_percent", "step_amplitude_percent", "direction_percent"]].to_string(index=False), flush=True)
    print(f"Figures: {args.out}", flush=True)


if __name__ == "__main__":
    main()
