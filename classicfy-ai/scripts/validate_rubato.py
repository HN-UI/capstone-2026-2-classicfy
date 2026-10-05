"""Draw one-chart-per-file Rubato examples from an aligned ASAP piece.

Run from the repository root:
    .venv/bin/python classicfy-ai/scripts/validate_rubato.py

Uses the existing extraction API without changing its values or masks. Percentage
axes are display conversions of log2 features, not another normalization step.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

# Share the Tempo example's dataset grouping, style and one-chart export helpers.
from validate_tempo import (
    BLUE, GRAY, ORANGE, chart, collect, masked_raw, percentage_delta,
    plt, save, setup_style, write_csv,
)
from matplotlib.ticker import FuncFormatter
from features import (
    TempoInput, extract_piece_rubato_features, extract_piece_tempo_features,
    summarize_rubato,
)

ROLES = ("편차 작은 편", "편차 중간", "편차 큰 편")
COLORS = (BLUE, GRAY, ORANGE)


def max_error(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    valid = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[valid] - expected[valid]))) if valid.any() else 0.


def validate(samples, tempo, rubato):
    rows, beat_rows, pairs = [], [], []
    absolute_matrix = []
    errors = {name: 0. for name in (
        "raw_log2", "absolute_log2", "common_log2", "relative_log2",
        "absolute_median_log2", "reconstructed_interval_seconds", "pair_difference_log2",
    )}
    for sample in samples:
        key = sample.performance_key
        tf, rf = tempo[key], rubato[key]
        score_dt, perf_dt = np.diff(sample.score_beats), np.diff(sample.performance_beats)
        raw_mask = np.array([interval.mask for interval in tf.intervals])
        raw = np.full(len(raw_mask), np.nan)
        raw[raw_mask] = np.log2(score_dt[raw_mask] / perf_dt[raw_mask])
        baseline = float(np.median(raw[raw_mask]))
        absolute = raw - baseline
        errors["raw_log2"] = max(errors["raw_log2"], max_error(masked_raw(tf), raw))
        errors["absolute_log2"] = max(errors["absolute_log2"], max_error(rf.absolute_rubato_sequence.values, absolute))
        np.testing.assert_array_equal(rf.absolute_rubato_sequence.mask, raw_mask)
        errors["absolute_median_log2"] = max(errors["absolute_median_log2"], abs(float(np.median(absolute[raw_mask]))))
        reconstructed = score_dt[raw_mask] / np.exp2(baseline + rf.absolute_rubato_sequence.values[raw_mask])
        errors["reconstructed_interval_seconds"] = max(
            errors["reconstructed_interval_seconds"], max_error(reconstructed, perf_dt[raw_mask]))
        absolute_matrix.append(absolute)
        summary = summarize_rubato(rf)
        rows.append({"key": key, "performance": Path(key).stem, "intervals": len(raw),
                     "absolute_valid": int(raw_mask.sum()), "relative_valid": int(rf.relative_rubato_sequence.mask.sum()),
                     **{f"status_{name}": sum(i.status == name for i in tf.intervals)
                        for name in ("regular", "special", "suspicious", "invalid")},
                     "tempo_baseline_log2": baseline, "tempo_baseline_vs_score_percent": float(100 * np.exp2(baseline)),
                     **summary,
                     "absolute_amount_percent_ratio_gap": float(percentage_delta(summary["absolute_rubato_amount"])),
                     "relative_amount_percent_ratio_gap": float(percentage_delta(summary["relative_rubato_amount"]))})
        for i, interval in enumerate(tf.intervals):
            beat_rows.append({"key": key, "beat_index": i, "display_interval": i + 1,
                              "score_interval_seconds": float(score_dt[i]), "performance_interval_seconds": float(perf_dt[i]),
                              "status": interval.status, "status_reason": interval.status_reason,
                              "raw_tempo_log2": interval.score_relative_tempo, "tempo_mask": bool(raw_mask[i]),
                              "tempo_baseline_log2": baseline,
                              "absolute_rubato_log2": float(rf.absolute_rubato_sequence.values[i]),
                              "absolute_mask": bool(rf.absolute_rubato_sequence.mask[i]),
                              "common_rubato_log2": float(rf.common_rubato_sequence.values[i]),
                              "common_mask": bool(rf.common_rubato_sequence.mask[i]),
                              "common_support": int(rf.common_rubato_support[i]),
                              "relative_rubato_log2": float(rf.relative_rubato_sequence.values[i]),
                              "relative_mask": bool(rf.relative_rubato_sequence.mask[i]),
                              "absolute_speed_delta_percent": float(percentage_delta(rf.absolute_rubato_sequence.values[i])),
                              "common_speed_delta_percent": float(percentage_delta(rf.common_rubato_sequence.values[i])),
                              "relative_speed_delta_percent": float(percentage_delta(rf.relative_rubato_sequence.values[i]))})
    matrix = np.array(absolute_matrix)
    support = np.isfinite(matrix).sum(axis=0)
    common = np.full(matrix.shape[1], np.nan)
    for i in np.flatnonzero(support >= 2):
        common[i] = np.median(matrix[np.isfinite(matrix[:, i]), i])
    for row_index, sample in enumerate(samples):
        rf = rubato[sample.performance_key]
        np.testing.assert_array_equal(rf.common_rubato_support, support)
        np.testing.assert_array_equal(rf.common_rubato_sequence.mask, support >= 2)
        expected_mask = np.isfinite(matrix[row_index]) & (support >= 2)
        np.testing.assert_array_equal(rf.relative_rubato_sequence.mask, expected_mask)
        relative = matrix[row_index] - common
        errors["common_log2"] = max(errors["common_log2"], max_error(rf.common_rubato_sequence.values, common))
        errors["relative_log2"] = max(errors["relative_log2"], max_error(rf.relative_rubato_sequence.values, relative))
    features = list(rubato.values())
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            shared = a.relative_rubato_sequence.mask & b.relative_rubato_sequence.mask
            if not shared.any():
                continue
            before = (a.absolute_rubato_sequence.values - b.absolute_rubato_sequence.values)[shared]
            after = (a.relative_rubato_sequence.values - b.relative_rubato_sequence.values)[shared]
            error = max_error(after, before)
            errors["pair_difference_log2"] = max(errors["pair_difference_log2"], error)
            d0, d1 = float(np.median(np.abs(before))), float(np.median(np.abs(after)))
            pairs.append({"a": a.performance_key, "b": b.performance_key, "shared_beats": int(shared.sum()),
                          "before_log2_distance": d0, "after_log2_distance": d1,
                          "before_percent_ratio_gap": float(percentage_delta(d0)),
                          "after_percent_ratio_gap": float(percentage_delta(d1)), "max_beat_error": error})
    if any(error > 1e-12 for error in errors.values()):
        raise AssertionError(errors)
    return rows, beat_rows, pairs, errors


def check_speed_invariance(samples, tempo, rubato, factor=1.2):
    inputs = []
    for sample in samples:
        original = TempoInput.from_asap_sample(sample)
        beats = np.asarray(original.performance_beats)
        faster_beats = beats[0] + (beats - beats[0]) / factor
        inputs.append(replace(original, performance_beats=faster_beats))
    shifted_tempo = extract_piece_tempo_features(inputs)
    shifted = extract_piece_rubato_features(shifted_tempo)
    errors = {name: 0. for name in ("tempo_shift_log2", "absolute_log2", "common_log2", "relative_log2")}
    for key, rf in rubato.items():
        original_raw, faster_raw = masked_raw(tempo[key]), masked_raw(shifted_tempo[key])
        np.testing.assert_array_equal([i.mask for i in tempo[key].intervals], [i.mask for i in shifted_tempo[key].intervals])
        np.testing.assert_array_equal([i.status for i in tempo[key].intervals], [i.status for i in shifted_tempo[key].intervals])
        np.testing.assert_array_equal(rf.common_rubato_support, shifted[key].common_rubato_support)
        errors["tempo_shift_log2"] = max(errors["tempo_shift_log2"], max_error(faster_raw, original_raw + np.log2(factor)))
        for name in ("absolute", "common", "relative"):
            a, b = getattr(rf, f"{name}_rubato_sequence"), getattr(shifted[key], f"{name}_rubato_sequence")
            np.testing.assert_array_equal(a.mask, b.mask)
            errors[f"{name}_log2"] = max(errors[f"{name}_log2"], max_error(a.values, b.values))
    return shifted, errors


def legend(fig, ax):
    fig.legend(*ax.get_legend_handles_labels(), fontsize=10, ncols=2,
               loc="lower center", bbox_to_anchor=(.55, .08))


def plot_examples(tempo, rubato, shifted, rows, pairs, out, label, curve_beats):
    ordered = sorted(rows, key=lambda row: (row["relative_rubato_amount"], row["key"]))
    selected = [ordered[int(round(q * (len(ordered) - 1)))] for q in (.25, .5, .75)]
    keys = [row["key"] for row in selected]
    names = {key: f"{role} · {Path(key).stem}" for key, role in zip(keys, ROLES)}
    reference = rubato[keys[1]]
    width = min(curve_beats, len(reference.absolute_rubato_sequence))
    start = (len(reference.absolute_rubato_sequence) - width) // 2
    stop, x = start + width, np.arange(start + 1, start + width + 1)
    section = slice(start, stop)

    valid_beats = np.flatnonzero(reference.absolute_rubato_sequence.mask)
    sorted_beats = valid_beats[np.argsort(reference.absolute_rubato_sequence.values[valid_beats], kind="stable")]
    beats = [int(sorted_beats[int(round(q * (len(sorted_beats) - 1)))]) for q in (.25, .5, .75)]
    baseline = selected[1]["tempo_baseline_log2"]
    values = [float(100 * np.exp2(reference.absolute_rubato_sequence.values[i])) for i in beats]
    fig, ax = chart("루바토는 자기 평소 속도에서 빨라지고 느려지는 정도다", f"{label} | {Path(keys[1]).stem} | 평소 속도 = 전체 유효 Tempo의 중앙값")
    labels = [f"구간 {i + 1}\n{role}" for i, role in zip(beats, ("평소보다 느린 구간", "평소와 비슷한 구간", "평소보다 빠른 구간"))]
    bars = ax.barh(labels, values, height=.48, color=COLORS)
    for bar, i, value in zip(bars, beats, values):
        interval = tempo[keys[1]].intervals[i]
        expected_duration = interval.score_duration / np.exp2(baseline)
        ax.text(value + .7, bar.get_y() + bar.get_height() / 2,
                f"평소 대비 {value:.1f}%  ·  실제 {interval.performance_duration:.3f}초", va="center", fontsize=12)
        # Express the reference time for the very same score interval, even if
        # score intervals are nonuniform in another piece selected by --work.
        ax.text(2, bar.get_y() + bar.get_height() / 2, f"평소 속도라면 {expected_duration:.3f}초",
                va="center", color="white", fontsize=11)
    ax.axvline(100, color="#252d28", ls="--", lw=1.5, label="자기 평소 속도 100%")
    ax.set(xlabel="자기 평소 속도 대비 해당 구간의 속도(%)", xlim=(0, max(values) * 1.5))
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=10)
    save(fig, out / "01_rubato_from_own_baseline.png", "한 연주의 전체 유효 구간을 속도순으로 정렬해 25·50·75% 위치를 골랐다. 곡의 시간순 25·50·75%가 아니다.")

    plotted = [percentage_delta(reference.common_rubato_sequence.values[section])]
    for key in keys:
        plotted.extend(percentage_delta(getattr(rubato[key], f"{name}_rubato_sequence").values[section])
                       for name in ("absolute", "relative"))
    valid_plotted = np.concatenate(plotted)
    valid_plotted = valid_plotted[np.isfinite(valid_plotted)]
    if not len(valid_plotted):
        raise ValueError("Central curve window has no valid beats; choose another work or increase --curve-beats")
    limit = max(4., float(np.max(np.abs(valid_plotted))) * 1.15)
    formatter = FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    for name, title, subtitle, filename in (
        ("absolute", "전체 빠르기를 빼면 구간별 속도 변화가 보인다",
         "0% = 각 연주 자신의 평소 속도", "02_absolute_rubato_curves.png"),
        ("relative", "작품 공통 변화를 빼도 연주별 루바토 차이는 남는다",
         "0% = 그 위치의 공통 루바토 패턴", "03_relative_rubato_curves.png"),
    ):
        fig, ax = chart(title, f"{label} | 가운데 {width}개 구간 확대 | {subtitle}")
        for key, color in zip(keys, COLORS):
            ax.plot(x, percentage_delta(getattr(rubato[key], f"{name}_rubato_sequence").values[section]),
                    color=color, lw=2, label=names[key])
        if name == "absolute":
            ax.plot(x, percentage_delta(reference.common_rubato_sequence.values[section]), color="#252d28", lw=2.3,
                    label=f"작품 공통 변화 · {len(rows)}개 연주의 중앙값")
        ax.axhline(0, color="#252d28", lw=1, ls="--", label="각 연주의 평소 속도" if name == "absolute" else "위치별 공통 패턴")
        ax.set(xlabel="악보 beat 구간 번호", ylabel="평소 대비 속도 편차" if name == "absolute" else "공통 변화 대비 속도 편차",
               xlim=(x[0], x[-1]), ylim=(-limit, limit))
        ax.yaxis.set_major_formatter(formatter)
        legend(fig, ax)
        save(fig, out / filename, "두 곡선 그림의 연주·구간·세로 눈금은 같다. 양수는 기준보다 빠름, 음수는 느림. 평활화·clipping 없음.", bottom=.26)

    fig, ax = chart("전체 연주에서 공통 제거 전후 루바토 양을 비교한다", f"{label} | 전체 유효 구간 사용 | 연결된 두 점은 같은 연주")
    before = [row["absolute_amount_percent_ratio_gap"] for row in ordered]
    after = [row["relative_amount_percent_ratio_gap"] for row in ordered]
    y = np.arange(len(ordered))
    for index, a, b in zip(y, before, after):
        ax.plot([a, b], [index, index], color="#c3c6bd", lw=2)
        ax.text(max(a, b) + .12, index, f"{a:.1f}% → {b:.1f}%", va="center", fontsize=11)
    ax.scatter(before, y, color=GRAY, s=65, label="공통 제거 전 · 절대 루바토")
    ax.scatter(after, y, color=ORANGE, s=65, label="공통 제거 후 · 상대 루바토")
    ax.set(yticks=y, yticklabels=[row["performance"] for row in ordered], xlabel="대표 변화 크기(속도 배율 차이, %)",
           xlim=(0, max(before + after) * 1.45), ylim=(-.6, len(ordered) - .4))
    legend(fig, ax)
    save(fig, out / "04_all_performance_rubato_amounts.png", "변화 크기 = median(|log2 루바토|)를 배율 차이로 변환. 부호는 없으며 연주의 좋고 나쁨을 뜻하지 않는다.", bottom=.23)

    fig, ax = chart("곡 전체를 20% 빠르게 해도 루바토 곡선은 유지된다", f"{label} | {Path(keys[1]).stem} | 실제 연주와 시간 간격을 1/1.2로 줄인 계산 실험")
    original = percentage_delta(reference.absolute_rubato_sequence.values[section])
    faster = percentage_delta(shifted[keys[1]].absolute_rubato_sequence.values[section])
    ax.plot(x, original, color=BLUE, lw=3, alpha=.8, label="실제 연주 · 절대 루바토")
    ax.plot(x, faster, color=ORANGE, lw=1.8, ls="--", label="전체 속도 1.2배 · 절대 루바토")
    ax.axhline(0, color="#252d28", ls="--", lw=1)
    ax.set(xlabel="악보 beat 구간 번호", ylabel="각 버전의 평소 대비 속도 편차", xlim=(x[0], x[-1]), ylim=(-limit, limit))
    ax.yaxis.set_major_formatter(formatter)
    legend(fig, ax)
    save(fig, out / "05_global_speed_invariance.png", "두 선이 겹치면 전체 빠르기와 국소 변화가 분리된다. 1.2배 버전은 새 실제 연주가 아니라 계산 실험이다.", bottom=.23)

    fig, ax = chart("공통 루바토를 빼도 연주 사이의 차이는 유지된다", f"{label} | 모든 {len(pairs)}개 연주 쌍 | 동일한 유효 위치끼리 비교")
    before = np.array([pair["before_percent_ratio_gap"] for pair in pairs])
    after = np.array([pair["after_percent_ratio_gap"] for pair in pairs])
    bound = max(1., float(before.max()) * 1.15)
    ax.plot([0, bound], [0, bound], color="#343b35", ls="--", lw=1.5, label="제거 전후 차이가 같은 위치")
    ax.scatter(before, after, color=BLUE, s=52, edgecolor="white", linewidth=.8, label="연주 쌍 하나")
    ax.set(xlabel="공통 제거 전의 루바토 배율 차이(%)", ylabel="공통 제거 후의 루바토 배율 차이(%)",
           xlim=(0, bound), ylim=(0, bound))
    ax.legend(loc="upper left", fontsize=10)
    save(fig, out / "06_pairwise_difference_preserved.png", "거리 = 같은 beat의 |절대 루바토 A − B| 중앙값을 배율 차이로 변환. 전체 빠르기는 이미 제거한 상태다.")
    return selected, beats, [start + 1, stop]


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asap-root", type=Path, default=ai.parent.parent / "datasets/ASAP")
    parser.add_argument("--nasap-root", type=Path, default=ai.parent.parent / "datasets/nASAP")
    parser.add_argument("--work", default="Bach/Fugue/bwv_848")
    parser.add_argument("--curve-beats", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ai / "analysis/rubato")
    args = parser.parse_args()
    if args.curve_beats < 2:
        parser.error("--curve-beats must be at least 2")
    root, nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    samples, tempo = collect(root, nasap_root, args.work)
    rubato = extract_piece_rubato_features(tempo)
    rows, beat_rows, pairs, errors = validate(samples, tempo, rubato)
    shifted, speed_errors = check_speed_invariance(samples, tempo, rubato)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    label = "Bach · Fugue BWV 848" if args.work == "Bach/Fugue/bwv_848" else args.work
    selected, beats, curve_range = plot_examples(tempo, rubato, shifted, rows, pairs, args.out, label, args.curve_beats)
    write_csv(args.out / "performance_summary.csv", rows)
    write_csv(args.out / "beat_features.csv", beat_rows)
    write_csv(args.out / "pairwise_distances.csv", pairs)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    inputs = [root / "asap_annotations.json", root / "metadata.csv", nasap_root / "metadata.csv",
              Path(__file__), Path(__file__).with_name("validate_tempo.py"),
              *sorted((ai / "src/features").glob("*.py")), *sorted((ai / "src/preprocessing").glob("*.py"))]
    reference = next(iter(rubato.values()))
    stats = {
        "work": args.work, "performances": len(rows), "intervals_per_performance": len(reference.absolute_rubato_sequence),
        "valid_absolute_values": sum(row["absolute_valid"] for row in rows),
        "valid_relative_values": sum(row["relative_valid"] for row in rows),
        "status_counts": dict(Counter(i.status for tf in tempo.values() for i in tf.intervals)),
        "common_support_min": int(reference.common_rubato_support.min()),
        "curve_selection_rule": "nearest ranks to 25th, 50th and 75th percentiles of relative_rubato_amount; no extreme selection",
        "selected_performances": [{"role": role, "key": row["key"], "relative_amount_percent_ratio_gap": row["relative_amount_percent_ratio_gap"]}
                                  for role, row in zip(ROLES, selected)],
        "example_key": selected[1]["key"], "example_display_intervals": [i + 1 for i in beats],
        "example_beat_rule": "nearest ranks to 25th, 50th and 75th percentiles of absolute_rubato within the median-relative-amount performer",
        "curve_display_interval_range": curve_range, "pair_count": len(pairs),
        "min_relative_amount_percent_ratio_gap": min(row["relative_amount_percent_ratio_gap"] for row in rows),
        "max_relative_amount_percent_ratio_gap": max(row["relative_amount_percent_ratio_gap"] for row in rows),
        "median_absolute_amount_percent_ratio_gap": float(np.median([row["absolute_amount_percent_ratio_gap"] for row in rows])),
        "median_relative_amount_percent_ratio_gap": float(np.median([row["relative_amount_percent_ratio_gap"] for row in rows])),
        "median_pairwise_percent_ratio_gap": float(np.median([pair["before_percent_ratio_gap"] for pair in pairs])),
        "min_pairwise_percent_ratio_gap": min(pair["before_percent_ratio_gap"] for pair in pairs),
        "max_pairwise_percent_ratio_gap": max(pair["before_percent_ratio_gap"] for pair in pairs),
        "common_rubato_amount_log2": float(np.median(np.abs(reference.common_rubato_sequence.values[reference.common_rubato_sequence.mask]))),
        "extraction_checks": errors, "speed_invariance_checks": {"speed_factor": 1.2, **speed_errors},
        "asap_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "inputs_and_code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest(),
        "clipping": None, "smoothing": None, "standardization": None,
        "validation_scope": "Beat-level numerical consistency, invariance to a uniform time scaling, and within-piece differences; not independent annotation accuracy, score-marking separation or recommendation evaluation",
    }
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
