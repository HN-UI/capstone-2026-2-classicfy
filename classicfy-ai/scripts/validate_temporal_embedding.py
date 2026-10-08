"""Evaluate epoch growth, hidden-beat shape and order perturbations on fixed validation works."""

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
from embedding.evaluation import baseline_predictions, shuffle_visible, temporal_metrics
from embedding.training import TrainingConfig, fixed_masks, load_model
from validate_embedding import collect_groups
from validate_tempo import BLUE, ORANGE, GRAY, chart, plt, save, setup_style, write_csv


LABELS = {"initial": "학습 전", "epoch2": "2 epoch", "best": "최선 모델",
          "shuffled": "최선 모델 · 관측 순서 섞음", "zero": "0 예측",
          "visible_mean": "관측 구간 평균", "linear_interpolation": "직선 보간"}
NAMES = dict(zip(CHANNELS, ("템포", "루바토", "강약", "아티큘레이션", "페달 깊이", "페달 사용 비율", "페달 전환")))


def predictions(model, values, valid, hidden, *, batch_size, shuffle_seed=None):
    outputs = []
    generator = torch.Generator().manual_seed(shuffle_seed) if shuffle_seed is not None else None
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x, v, h = values[start:start + batch_size], valid[start:start + batch_size], hidden[start:start + batch_size]
            if generator is not None:
                x = shuffle_visible(x, v, h, generator=generator)
            outputs.append(model(x, v, h)[0].numpy())
    return np.concatenate(outputs)


def feature_average(rows, metric):
    values = {row["channel"]: row[metric] for row in rows}
    blocks = [values[name] for name in CHANNELS[:4]]
    pedal = [values[name] for name in CHANNELS[4:] if values[name] is not None]
    if pedal:
        blocks.append(float(np.mean(pedal)))
    return float(np.mean([x for x in blocks if x is not None]))


def plot_summary(history, metrics, aggregate, out, best_epoch):
    setup_style()
    fig, ax = chart("epoch이 늘어나면 복원 오차가 줄어들까?", "같은 검증 작품 · 같은 숨긴 위치 · 검증 오차가 낮을수록 좋음")
    ax.plot(history.epoch, history.train_loss, label="학습 (매번 다른 마스크)", color=GRAY)
    ax.plot(history.epoch, history.validation_loss, label="검증 (고정 마스크)", color=BLUE, marker="o", ms=3)
    ax.axvline(best_epoch, color=ORANGE, linestyle=":", label=f"최선 {best_epoch} epoch")
    ax.axhline(aggregate["visible_mean"]["mse"], color="#8956a8", linestyle="--", label="관측 구간 평균")
    ax.set(xlabel="학습 epoch", ylabel="5 feature 동일 비중 MSE")
    ax.legend(fontsize=9)
    save(fig, out / "01_epoch_loss.png", "초기 모델은 epoch 0. 학습·검증 데이터가 다르므로 두 곡선의 절대 크기를 직접 비교하지 않습니다.")
    for metric, title, ylabel, filename in (
            ("mse", "숨긴 값의 복원 오차", "MSE (낮을수록 좋음)", "02_feature_errors.png"),
            ("shape_correlation", "구간 평균을 빼도 변화 모양을 따라갈까?", "구간 내 중심을 제거한 상관 (높을수록 좋음)", "03_shape_correlation.png"),
            ("slope_mse", "숨긴 연속 beat의 변화량을 복원할까?", "변화량 MSE (낮을수록 좋음)", "04_change_errors.png")):
        fig, ax = chart(title, "모든 검증 구간 사용 · 페달 세 채널도 각각 표시")
        methods = [m for m in ("initial", "epoch2", "best", "visible_mean", "linear_interpolation") if m in aggregate]
        x = np.arange(7)
        width = .8 / len(methods)
        colors = (GRAY, "#72a8d1", BLUE, "#8956a8", ORANGE)
        for i, method in enumerate(methods):
            subset = {r["channel"]: r for r in metrics if r["method"] == method}
            y = [subset[channel][metric] if subset[channel][metric] is not None else 0 for channel in CHANNELS]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, y, width, color=colors[i], label=LABELS[method])
        ax.set_xticks(x, [NAMES[name] for name in CHANNELS], rotation=15)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        if metric == "shape_correlation":
            ax.set_ylim(-.25, 1)
        save(fig, out / filename, "평균 예측은 구간 내 상수라 모양 상관이 정의되지 않습니다. 해당 막대는 0으로 표시하고 CSV는 결측으로 유지합니다." if metric == "shape_correlation" else "변화량은 인접한 두 beat가 모두 숨긴 위치에서만 계산합니다. 값의 오프셋을 맞추는 것과 구분합니다.")
    methods = [m for m in ("best", "shuffled", "visible_mean", "linear_interpolation") if m in aggregate]
    fig, ax = chart("관측 순서를 섞으면 복원은 어떻게 바뀔까?", "값의 집합·마스크·정답은 고정하고, 보이는 7개 특징의 묶음을 시간축에서 재배치")
    x = np.arange(len(methods))
    values = [aggregate[m]["mse"] for m in methods]
    ax.bar(x, values, color=[BLUE, ORANGE, "#8956a8", GRAY][:len(methods)])
    ax.set_xticks(x, [LABELS[m] for m in methods], rotation=10)
    ax.set_ylabel("5 feature 동일 비중 MSE")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
    save(fig, out / "05_order_diagnostic.png", "추론 시 입력 변형 진단입니다. 순서를 사용하지 않는 모델을 따로 학습해 비교한 실험은 아닙니다.")
    for metric, title, ylabel, filename in (
            ("direction_accuracy", "숨긴 beat의 상승·하강 방향을 맞힐까?", "방향 일치 비율", "06_direction_accuracy.png"),
            ("shape_amplitude_ratio", "실제 변화의 크기를 얼마나 복원할까?", "예측 / 실제 변화 크기", "07_variation_amplitude.png")):
        fig, ax = chart(title, "구간 안의 숨긴 위치에서 평가 · 값의 전체 평균을 맞추는 효과와 구분")
        methods = ("best", "shuffled", "linear_interpolation")
        x, width = np.arange(7), .24
        for i, method in enumerate(methods):
            subset = {row["channel"]: row for row in metrics if row["method"] == method}
            ax.bar(x + (i - 1) * width, [subset[name][metric] for name in CHANNELS], width,
                   label=LABELS[method], color=(BLUE, GRAY, ORANGE)[i])
        reference = .5 if metric == "direction_accuracy" else 1.
        ax.axhline(reference, color="black", linestyle=":",
                   label="무작위 상승·하강 기대값 50%" if metric == "direction_accuracy" else "실제 변화 크기 100%")
        ax.set_xticks(x, [NAMES[name] for name in CHANNELS], rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        save(fig, out / filename, "방향: 인접한 hidden beat 중 실제 변화가 0이 아닌 경우. 크기: 구간별 hidden 평균을 제거한 예측/실제 표준편차 비율입니다.")


def select_examples(dataset):
    """Three works by fixed alphabetical positions, median performer and central window."""
    pieces = sorted({s.piece for s in dataset.sequences})
    chosen = sorted({0, len(pieces) // 2, len(pieces) - 1})
    indices = []
    for wi in chosen:
        owners = sorted({si for si, _ in dataset.items if dataset.sequences[si].piece == pieces[wi]},
                        key=lambda si: dataset.sequences[si].key)
        si = owners[len(owners) // 2]
        center = max(0, (dataset.sequences[si].values.shape[1] - dataset.window_size) / 2)
        index = min((i for i, (owner, _) in enumerate(dataset.items) if owner == si),
                    key=lambda i: (abs(dataset.items[i][1] - center), dataset.items[i][1]))
        indices.append(index)
    return indices


def plot_examples(dataset, indices, target, valid, hidden, predictions_by_method, out):
    directory = out / "examples"
    directory.mkdir()
    records = []
    for number, index in enumerate(indices, 1):
        si, start = dataset.items[index]
        sequence = dataset.sequences[si]
        beats = start + np.arange(dataset.window_size)
        for channel, name in enumerate(CHANNELS):
            fig, ax = chart(f"구간 흐름 복원 · {NAMES[name]}", f"{sequence.key} | beat {start}부터 {dataset.window_size}구간")
            actual = np.where(valid[index, channel], target[index, channel], np.nan)
            selected = hidden[index, channel]
            ax.plot(beats, actual, color="black", lw=1.5, label="실제 곡선 (숨긴 정답 포함)")
            for method, color, style in (("initial", GRAY, ":"), ("best", BLUE, "-"),
                                         ("visible_mean", "#8956a8", "--"), ("linear_interpolation", ORANGE, "--")):
                # NaN outside hidden intervals avoids connecting unrelated gaps.
                y = np.where(selected, predictions_by_method[method][index, channel], np.nan)
                ax.plot(beats, y, color=color, linestyle=style, marker="o", ms=3, label=LABELS[method])
            ax.scatter(beats[selected], actual[selected], marker="x", color="black", label="숨긴 위치")
            ax.set(xlabel="원래 score beat 구간 번호", ylabel="표준화된 상대 feature")
            ax.legend(fontsize=8)
            save(fig, directory / f"{number:02d}_{name}.png", "선정은 작품 이름순 3곳·중앙 연주·중앙 구간으로 고정합니다. 잘 복원한 사례를 골라낸 그림이 아닙니다.")
            for beat in np.flatnonzero(selected):
                records.append({"example": number, "key": sequence.key, "channel": name,
                                "beat": int(beats[beat]), "target": float(target[index, channel, beat]),
                                **{method: float(value[index, channel, beat]) for method, value in predictions_by_method.items()}})
    return records


def main():
    ai = Path(__file__).resolve().parents[1]
    datasets = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=datasets / "temporal_embedding_run01")
    parser.add_argument("--early-run", type=Path, default=datasets / "temporal_embedding_smoke")
    parser.add_argument("--out", type=Path, default=ai / "analysis/temporal_embedding")
    parser.add_argument("--asap-root", type=Path, default=datasets / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=datasets / "nASAP")
    parser.add_argument("--cache", type=Path, default=datasets / "feature_normalization_raw.npz")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or (args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir()))):
        parser.error("Choose a new/empty output directory and positive thread count")
    torch.set_num_threads(args.threads)
    stored = json.loads((args.run / "dataset.json").read_text())
    info = json.loads((args.run / "training.json").read_text())
    cache_hash = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if cache_hash != stored["cache_sha256"]:
        parser.error("Raw cache differs from training reference")
    groups, audits, _, provenance = collect_groups(SimpleNamespace(asap_root=args.asap_root.resolve(),
        nasap_root=args.nasap_root.resolve(), cache=args.cache, **stored["cohort_settings"]))
    if provenance != stored["source_provenance"]:
        parser.error("Feature provenance differs from training reference")
    sequences = restore_sequences(groups, audits)
    validation = WindowDataset([s for s in sequences if stored["split"][s.piece] == "validation"],
                               window_size=info["model_config"]["window_size"], stride=stored["stride"])
    config = TrainingConfig(**info["training_config"])
    masks = fixed_masks(validation, config)
    values, valid = [], []
    for x, v, _ in DataLoader(validation, batch_size=config.batch_size, shuffle=False):
        values.append(x)
        valid.append(v)
    values, valid = torch.cat(values), torch.cat(valid)
    target, validity, selected = values.numpy(), valid.numpy(), masks.numpy()
    by_method = baseline_predictions(target, validity, selected)
    checkpoints = {"initial": args.run / "initial.pt", "best": args.run / "best.pt"}
    if args.early_run is not None and (args.early_run / "best.pt").exists():
        early_stored = json.loads((args.early_run / "dataset.json").read_text())
        early_info = json.loads((args.early_run / "training.json").read_text())
        if (early_stored["split"] != stored["split"] or early_stored["cache_sha256"] != cache_hash
                or early_info["model_config"] != info["model_config"]
                or early_info["training_config"]["seed"] != config.seed
                or early_info["training_config"]["mask_fraction"] != config.mask_fraction
                or early_info["training_config"]["mask_block_size"] != config.mask_block_size):
            parser.error("Early run must have matching validation and masking settings")
        if early_info["best_epoch"] != 2:
            parser.error("The early-run comparison expects its best checkpoint at epoch 2")
        checkpoints["epoch2"] = args.early_run / "best.pt"
    for method, path in checkpoints.items():
        model, _ = load_model(path)
        by_method[method] = predictions(model, values, valid, masks, batch_size=config.batch_size)
    best, _ = load_model(checkpoints["best"])
    by_method["shuffled"] = predictions(best, values, valid, masks, batch_size=config.batch_size,
                                        shuffle_seed=config.seed + 11)
    metrics, work_rows, work_summary, aggregate = [], [], [], {}
    works = np.array([validation.sequences[si].piece for si, _ in validation.items])
    for method, prediction in by_method.items():
        current = []
        for channel, name in enumerate(CHANNELS):
            row = {"method": method, "channel": name,
                   **temporal_metrics(target[:, channel], prediction[:, channel], selected[:, channel])}
            metrics.append(row)
            current.append(row)
            for piece in sorted(set(works)):
                keep = works == piece
                work_rows.append({"method": method, "piece": piece, "channel": name,
                    **temporal_metrics(target[keep, channel], prediction[keep, channel], selected[keep, channel])})
        aggregate[method] = {metric: feature_average(current, metric) for metric in ("mse", "slope_mse")}
        for piece in sorted(set(works)):
            selected_rows = [row for row in work_rows if row["method"] == method and row["piece"] == piece]
            work_summary.append({"method": method, "piece": piece,
                                 **{metric: feature_average(selected_rows, metric) for metric in ("mse", "slope_mse")}})
    np.testing.assert_allclose(aggregate["best"]["mse"], info["best_validation"]["loss"], rtol=1e-6)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "metrics.csv", metrics)
    write_csv(args.out / "work_metrics.csv", work_rows)
    write_csv(args.out / "work_summary.csv", work_summary)
    history = pd.read_csv(args.run / "history.csv")
    history.to_csv(args.out / "history.csv", index=False)
    plot_summary(history, metrics, aggregate, args.out, info["best_epoch"])
    examples = select_examples(validation)
    records = plot_examples(validation, examples, target, validity, selected, by_method, args.out)
    write_csv(args.out / "example_beats.csv", records)
    summary = {"run": str(args.run.resolve()), "cache_sha256": cache_hash,
               "evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()
                    + (ai / "src/embedding/evaluation.py").read_bytes()).hexdigest(),
               "checkpoint_sha256": {method: hashlib.sha256(path.read_bytes()).hexdigest() for method, path in checkpoints.items()},
               "validation_works": len(set(works)), "validation_windows": len(validation),
               "initial_epoch": 0, "best_epoch": info["best_epoch"], "completed_epochs": info["completed_epochs"],
               "aggregate": aggregate, "selected_examples": [{"key": validation.sequences[validation.items[i][0]].key,
                     "start_beat": validation.items[i][1]} for i in examples],
               "direction_tolerance": 1e-6, "normalization": stored["normalization"],
               "shape_definition": "Per-window centering on hidden valid values; pooled Pearson correlation",
               "slope_definition": "Adjacent pairs with both beats hidden and valid; original beat indices retained",
               "shuffle_definition": "Inference only; joint permutation of seven visible channels; targets and masks unchanged",
               "shuffle_seed": config.seed + 11,
               "work_macro": {method: {metric: float(np.mean([row[metric] for row in work_summary if row["method"] == method]))
                               for metric in ("mse", "slope_mse")} for method in aggregate}}
    (args.out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
