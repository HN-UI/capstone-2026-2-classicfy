"""Draw one-chart-per-file Tempo examples from an actual aligned ASAP piece.

Run from the repository root:
    .venv/bin/python classicfy-ai/scripts/validate_tempo.py

Extraction remains unchanged. Percentage labels are invertible display conversions
of the existing log2 features. All performers enter the common pattern and checks.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import TempoInput, extract_piece_tempo_features, summarize_tempo
from preprocessing import ASAPLoader

BLUE, ORANGE, GREEN, GRAY = "#2878c8", "#dc6436", "#148766", "#92958f"
ROLE_COLORS = (BLUE, GRAY, ORANGE)
ROLE_NAMES = ("느린 편", "중간", "빠른 편")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to export: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup_style():
    for name in ("Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
        break
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 19, "axes.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "grid.color": "#e2e3dd"})


def chart(title, subtitle, *, height=5.8):
    fig, ax = plt.subplots(figsize=(10, height))
    ax.set_title(title, loc="left", pad=46, fontweight="bold")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, color="#555953", fontsize=11)
    ax.grid(alpha=.55)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, path, note, *, bottom=.07):
    fig.text(.02, .025, note, color="#555953", fontsize=10)
    fig.tight_layout(rect=(0, bottom, 1, .98))
    if len(fig.axes) != 1:
        raise AssertionError("Each output must contain exactly one chart")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def masked_raw(feature):
    return np.array([i.score_relative_tempo if i.mask else np.nan for i in feature.intervals])


def percentage_delta(log_values):
    return 100 * np.expm1(np.log(2) * log_values)


def collect(asap_root, nasap_root, work):
    if Path(work).is_absolute() or ".." in Path(work).parts:
        raise ValueError("--work must be a piece folder relative to ASAP")
    loader = ASAPLoader(asap_root, nasap_root)
    score = (loader.root / work / "midi_score.mid").resolve()
    # The adapter keeps no_repeat/extra_repeat performances out of the default group.
    samples = [s for s in loader.iter_samples(True) if s.score_path == score and (
        s.note_alignment_path is None or s.note_alignment_path.parent.name == s.performance_path.parent.name
    )]
    if len(samples) < 3:
        raise ValueError("At least three aligned performances with the same repeat structure are required")
    features = extract_piece_tempo_features([TempoInput.from_asap_sample(s) for s in samples])
    if any(not f.individual_tempo_sequence.mask.any() for f in features.values()):
        raise ValueError("Each performance must have valid individual Tempo beats")
    return samples, features


def validate(samples, features):
    beat_rows, rows, pair_rows = [], [], []
    reference_matrix = []
    log_error, duration_error, residual_error, pair_error = 0., 0., 0., 0.
    for sample in samples:
        feature = features[sample.performance_key]
        score_dt = np.diff(sample.score_beats)
        performance_dt = np.diff(sample.performance_beats)
        mask = np.array([i.mask for i in feature.intervals])
        reference = np.full(len(mask), np.nan)
        reference[mask] = np.log2(score_dt[mask] / performance_dt[mask])
        raw = masked_raw(feature)
        np.testing.assert_allclose(raw, reference, rtol=0, atol=1e-12)
        reconstructed = score_dt[mask] / np.exp2(raw[mask])
        log_error = max(log_error, float(np.max(np.abs(raw[mask] - reference[mask]))))
        duration_error = max(duration_error, float(np.max(np.abs(reconstructed - performance_dt[mask]))))
        relative = feature.individual_tempo_sequence
        residual_error = max(residual_error, float(np.max(np.abs(
            relative.values[relative.mask] - (raw - feature.common_tempo_sequence.values)[relative.mask]
        ))))
        reference_matrix.append(reference)
        summary = summarize_tempo(feature)
        median_relative = summary["overall_individual_tempo"]
        rows.append({"key": sample.performance_key, "performance": Path(sample.performance_key).stem,
                     "intervals": len(raw), "raw_valid": int(mask.sum()), "individual_valid": int(relative.mask.sum()),
                     **{f"status_{name}": sum(i.status == name for i in feature.intervals)
                        for name in ("regular", "special", "suspicious", "invalid")},
                     **summary, "representative_vs_common_percent": float(percentage_delta(median_relative)),
                     "representative_vs_score_percent": float(100 * np.exp2(summary["overall_score_relative_tempo"]))})
        for i, interval in enumerate(feature.intervals):
            beat_rows.append({"key": sample.performance_key, "beat_index": i, "display_interval": i + 1,
                              "score_start_seconds": sample.score_beats[i], "score_end_seconds": sample.score_beats[i + 1],
                              "performance_start_seconds": sample.performance_beats[i],
                              "performance_end_seconds": sample.performance_beats[i + 1],
                              "score_interval_seconds": float(score_dt[i]), "performance_interval_seconds": float(performance_dt[i]),
                              "status": interval.status, "status_reason": interval.status_reason,
                              "raw_log2": interval.score_relative_tempo, "raw_mask": interval.mask,
                              "common_log2": float(feature.common_tempo_sequence.values[i]),
                              "common_support": int(feature.common_tempo_support[i]),
                              "individual_log2": float(relative.values[i]), "individual_mask": bool(relative.mask[i]),
                              "speed_vs_score_percent": float(100 * np.exp2(raw[i])),
                              "speed_vs_common_percent": float(percentage_delta(relative.values[i]))})
    matrix = np.asarray(reference_matrix)
    support = np.isfinite(matrix).sum(axis=0)
    common = np.full(matrix.shape[1], np.nan)
    for i in np.flatnonzero(support >= 2):
        common[i] = np.median(matrix[np.isfinite(matrix[:, i]), i])
    for feature in features.values():
        np.testing.assert_array_equal(feature.common_tempo_support, support)
        np.testing.assert_allclose(feature.common_tempo_sequence.values, common, rtol=0, atol=1e-12)
    common_error = float(np.max(np.abs(features[samples[0].performance_key].common_tempo_sequence.values[support >= 2] - common[support >= 2])))
    values = list(features.values())
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            overlap = a.individual_tempo_sequence.mask & b.individual_tempo_sequence.mask
            if not overlap.any():
                continue
            raw_difference = (masked_raw(a) - masked_raw(b))[overlap]
            relative_difference = (a.individual_tempo_sequence.values - b.individual_tempo_sequence.values)[overlap]
            error = float(np.max(np.abs(raw_difference - relative_difference)))
            pair_error = max(pair_error, error)
            before, after = float(np.median(np.abs(raw_difference))), float(np.median(np.abs(relative_difference)))
            pair_rows.append({"a": a.performance_key, "b": b.performance_key, "shared_beats": int(overlap.sum()),
                              "before_log2_distance": before, "after_log2_distance": after,
                              "before_percent_ratio_gap": float(percentage_delta(before)),
                              "after_percent_ratio_gap": float(percentage_delta(after)), "max_beat_error": error})
    errors = {"max_raw_log2_error": log_error, "max_reconstructed_interval_seconds_error": duration_error,
              "max_common_log2_error": common_error, "max_individual_log2_error": residual_error,
              "max_pair_difference_log2_error": pair_error}
    if any(value > 1e-12 for value in errors.values()):
        raise AssertionError(errors)
    return rows, beat_rows, pair_rows, errors


def plot_examples(samples, features, rows, pairs, errors, out, label, curve_beats):
    ordered = sorted(rows, key=lambda r: (r["overall_individual_tempo"], r["key"]))
    selected = [ordered[int(round(q * (len(ordered) - 1)))] for q in (.25, .5, .75)]
    keys = [r["key"] for r in selected]
    names = {key: f"{role} · {Path(key).stem}" for key, role in zip(keys, ROLE_NAMES)}
    colors = dict(zip(keys, ROLE_COLORS))
    reference = features[samples[0].performance_key]
    shared = np.logical_and.reduce([features[k].individual_tempo_sequence.mask for k in keys])
    positions = np.flatnonzero(shared)
    if not len(positions):
        raise ValueError("Selected performances have no jointly valid example beat")
    durations_by_key = [np.array([i.performance_duration for i in features[k].intervals]) for k in keys]
    ordered_beats = shared & (durations_by_key[0] >= durations_by_key[1]) & (durations_by_key[1] >= durations_by_key[2])
    candidates = np.flatnonzero(ordered_beats) if ordered_beats.any() else positions
    beat = int(candidates[np.argmin(np.abs(candidates - (len(shared) - 1) / 2))])
    fig, ax = chart("같은 악보 구간도 연주마다 걸리는 시간이 다르다", f"{label} | 악보 구간 {beat + 1} | 실제 연주 시각에서 잰 간격")
    score_duration = reference.intervals[beat].score_duration
    durations = [score_duration] + [features[k].intervals[beat].performance_duration for k in keys]
    labels = ["악보 MIDI 기준"] + [names[k] for k in keys]
    bars = ax.barh(labels, durations, color=["#343b35", *ROLE_COLORS], height=.52)
    for i, (bar, duration) in enumerate(zip(bars, durations)):
        speed = 100 if i == 0 else 100 * np.exp2(features[keys[i - 1]].intervals[beat].score_relative_tempo)
        ax.text(duration + .012 * max(durations), bar.get_y() + bar.get_height() / 2,
                f"{duration:.3f}초  ·  악보 대비 속도 {speed:.1f}%", va="center", fontsize=12)
    ax.set(xlabel="한 beat 구간에 걸린 시간(초)", xlim=(0, max(durations) * 1.65))
    ax.invert_yaxis()
    save(fig, out / "01_same_beat_duration.png", "막대가 짧을수록 빠른 연주. 세 연주는 전체 빠르기 순위의 25·50·75%에서 선택했다.")

    fig, ax = chart("Tempo에서 되돌린 beat 간격이 원본 시각과 일치한다", f"{label} | 모든 연주의 유효 beat로 계산 확인")
    all_measured, all_restored = [], []
    for feature in features.values():
        for interval in feature.intervals:
            if interval.mask:
                all_measured.append(interval.performance_duration)
                all_restored.append(interval.score_duration / np.exp2(interval.score_relative_tempo))
    ax.scatter(all_measured, all_restored, s=14, color=BLUE, alpha=.3, label=f"유효 beat {len(all_measured):,}개")
    lo, hi = min(all_measured) * .95, max(all_measured) * 1.05
    ax.plot([lo, hi], [lo, hi], ls="--", color="#343b35", lw=1.5, label="완전히 일치하는 위치")
    ax.set(xlabel="원본 beat 시각으로 잰 간격(초)", ylabel="추출된 Tempo 값에서 역산한 간격(초)", xlim=(lo, hi), ylim=(lo, hi))
    ax.legend(loc="upper left")
    save(fig, out / "02_tempo_calculation_check.png", f"최대 오차 {errors['max_reconstructed_interval_seconds_error']:.2g}초. 계산 일치 확인이며 beat 정렬의 음악적 정답 검증은 아니다.")

    width = min(curve_beats, len(reference.intervals))
    start = (len(reference.intervals) - width) // 2
    stop = start + width
    section = slice(start, stop)
    x = np.arange(start + 1, stop + 1)
    fig, ax = chart("같은 작품의 세 연주, 빠르기 곡선이 다르다", f"{label} | 중간 {width}개 구간 확대 | 100% = 악보 MIDI와 같은 속도")
    for key in keys:
        ax.plot(x, 100 * np.exp2(masked_raw(features[key])[section]), lw=2, color=colors[key], label=names[key])
    ax.plot(x, 100 * np.exp2(reference.common_tempo_sequence.values[section]), color="#252d28", lw=2.3,
            label=f"작품 공통 패턴 · {len(features)}개 연주의 중앙값")
    ax.axhline(100, color=GREEN, lw=1.1, ls="--", label="악보 MIDI 기준")
    ax.set(xlabel="악보 beat 구간 번호", ylabel="악보 MIDI 대비 속도", xlim=(x[0], x[-1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.legend(*ax.get_legend_handles_labels(), fontsize=10, ncols=2, loc="lower center", bbox_to_anchor=(.55, .08))
    save(fig, out / "03_performance_tempo_curves.png", "중간 구간을 평활화 없이 확대했다. 전체 beat는 CSV에 보존하며 악보 MIDI는 정답 속도가 아니다.", bottom=.26)

    fig, ax = chart("공통 빠르기를 제거해도 연주별 차이는 남는다", f"{label} | 같은 중간 {width}개 구간 | 0% = 공통 패턴 | 양수: 더 빠름, 음수: 더 느림")
    for key in keys:
        ax.plot(x, percentage_delta(features[key].individual_tempo_sequence.values[section]),
                lw=2, color=colors[key], label=names[key])
    ax.axhline(0, color="#252d28", lw=1.8, label="위치별 공통 패턴")
    ax.set(xlabel="악보 beat 구간 번호", ylabel="공통 패턴 대비 속도 편차", xlim=(x[0], x[-1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
    fig.legend(*ax.get_legend_handles_labels(), fontsize=10, ncols=2, loc="lower center", bbox_to_anchor=(.55, .08))
    save(fig, out / "04_individual_tempo_curves.png", "individual_tempo를 100 × (2^값 − 1)로 표시했다. 0 근처에 있어도 좋은·나쁜 연주를 뜻하지 않는다.", bottom=.26)

    fig, ax = chart("전체 9개 연주를 비교하면 빠르기 차이가 보인다" if len(rows) == 9 else f"전체 {len(rows)}개 연주의 빠르기 차이",
                    f"{label} | 작품 전체의 유효 individual_tempo 중앙값을 속도 편차로 변환", height=6.4)
    for i, row in enumerate(ordered):
        value = row["representative_vs_common_percent"]
        color = colors.get(row["key"], "#56625b")
        ax.hlines(i, min(0, value), max(0, value), color=color, lw=2, alpha=.4)
        ax.scatter(value, i, color=color, s=70, zorder=3)
        ax.annotate(f"{value:+.1f}%", (value, i), xytext=(8 if value >= 0 else -8, 0), textcoords="offset points",
                    va="center", ha="left" if value >= 0 else "right", color=color, fontweight="bold")
    ax.axvline(0, color="#252d28", ls="--", lw=1.3)
    values = [r["representative_vs_common_percent"] for r in ordered]
    span = max(max(values) - min(values), 1)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[r["performance"] for r in ordered],
           xlabel="작품 공통 패턴 대비 대표 속도 편차(%)",
           xlim=(min(values) - span * .18, max(values) + span * .18))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
    save(fig, out / "05_all_performance_differences.png", "아래는 느린 편, 위는 빠른 편. 점은 전체 유효 beat의 중앙값이며 곡 전체 재생시간의 비율은 아니다.")

    fig, ax = chart("공통 패턴을 빼도 연주 사이의 거리가 유지된다", f"{label} | 모든 {len(pairs)}개 연주 쌍 | 공통 유효 beat끼리 비교")
    before = np.array([r["before_percent_ratio_gap"] for r in pairs])
    after = np.array([r["after_percent_ratio_gap"] for r in pairs])
    bound = max(float(before.max()) * 1.12, 1)
    ax.plot([0, bound], [0, bound], color="#343b35", ls="--", lw=1.5, label="제거 전후 차이가 같은 위치")
    ax.scatter(before, after, s=52, color=BLUE, edgecolor="white", linewidth=.8, label="연주 쌍 하나")
    ax.set(xlabel="공통 패턴 제거 전의 속도 배율 차이(%)", ylabel="공통 패턴 제거 후의 속도 배율 차이(%)",
           xlim=(0, bound), ylim=(0, bound))
    ax.legend(loc="upper left", fontsize=10)
    save(fig, out / "06_pairwise_difference_preserved.png", "거리 = 공통 beat의 |Tempo A − Tempo B| 중앙값을 배율 차이(%)로 변환. 대각선 위이면 보존된다.")
    return selected, beat, [start + 1, stop]


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asap-root", type=Path, default=ai.parent.parent / "datasets/ASAP")
    parser.add_argument("--nasap-root", type=Path, default=ai.parent.parent / "datasets/nASAP")
    parser.add_argument("--work", default="Bach/Fugue/bwv_848")
    parser.add_argument("--curve-beats", type=int, default=32, help="Number of central intervals shown in curve figures")
    parser.add_argument("--out", type=Path, default=ai / "analysis/tempo")
    args = parser.parse_args()
    if args.curve_beats < 2:
        parser.error("--curve-beats must be at least 2")
    root, nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    samples, features = collect(root, nasap_root, args.work)
    rows, beat_rows, pairs, errors = validate(samples, features)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    label = "Bach · Fugue BWV 848" if args.work == "Bach/Fugue/bwv_848" else args.work
    selected, beat, curve_range = plot_examples(samples, features, rows, pairs, errors, args.out, label, args.curve_beats)
    write_csv(args.out / "performance_summary.csv", rows)
    write_csv(args.out / "beat_features.csv", beat_rows)
    write_csv(args.out / "pairwise_distances.csv", pairs)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    src = ai / "src"
    inputs = [root / "asap_annotations.json", root / "metadata.csv", nasap_root / "metadata.csv",
              Path(__file__), *sorted((src / "features").glob("*.py")), *sorted((src / "preprocessing").glob("*.py"))]
    summaries = np.array([r["overall_individual_tempo"] for r in rows])
    stats = {
        "work": args.work, "performances": len(rows), "intervals_per_performance": len(features[samples[0].performance_key].intervals),
        "valid_raw_values": sum(r["raw_valid"] for r in rows), "valid_individual_values": sum(r["individual_valid"] for r in rows),
        "status_counts": dict(Counter(i.status for f in features.values() for i in f.intervals)),
        "common_support_min": int(features[samples[0].performance_key].common_tempo_support.min()),
        "curve_selection_rule": "nearest ranks to 25th, 50th and 75th percentiles of overall_individual_tempo",
        "selected_performances": [{"role": role, "key": row["key"], "representative_vs_common_percent": row["representative_vs_common_percent"]}
                                  for role, row in zip(ROLE_NAMES, selected)],
        "example_beat_index": beat, "example_display_interval": beat + 1, "pair_count": len(pairs),
        "example_beat_rule": "Nearest midpoint jointly valid interval with slow/median/fast duration ordering; otherwise nearest jointly valid midpoint",
        "curve_display_interval_range": curve_range,
        "representative_slowest_percent": float(percentage_delta(summaries.min())),
        "representative_fastest_percent": float(percentage_delta(summaries.max())),
        "representative_fastest_vs_slowest_ratio": float(np.exp2(summaries.max() - summaries.min())),
        "median_pairwise_percent_ratio_gap": float(np.median([r["before_percent_ratio_gap"] for r in pairs])),
        "min_pairwise_percent_ratio_gap": min(r["before_percent_ratio_gap"] for r in pairs),
        "max_pairwise_percent_ratio_gap": max(r["before_percent_ratio_gap"] for r in pairs),
        "extraction_checks": errors, "asap_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "inputs_and_code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest(),
        "clipping": None, "smoothing": None, "standardization": None,
        "validation_scope": "Numerical consistency with ASAP aligned beat times; not independent beat annotation accuracy or recommendation evaluation",
    }
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
