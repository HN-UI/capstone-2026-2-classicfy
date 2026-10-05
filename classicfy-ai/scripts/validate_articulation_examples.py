"""Readable single-chart Articulation examples; default output analysis/_articulation.

Run from the repository root:
    .venv/bin/python classicfy-ai/scripts/validate_articulation_examples.py

Production log2 extraction, beat medians and MAD scaling are unchanged.
Percentages are display conversions. Robust note alignment is required by
default; --include-non-robust explicitly relaxes that caller-side policy.
"""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

from validate_tempo import BLUE, GRAY, ORANGE, chart, plt, save, setup_style, write_csv
from validate_dynamics import check, collect_samples, legend, value_range
from features import (PieceFeatureInput, extract_articulation, separate_piece_feature,
                      standardize_piece_feature)
from features.articulation import build_tempo_map, extract_note_articulation
from preprocessing import load_match, load_midi

COLORS = (BLUE, GRAY, ORANGE)
ROLES = ("평균 길이 짧은 편", "평균 길이 중간", "평균 길이 긴 편")
# Match and MIDI clocks are discretized differently; same tolerance as the
# existing whole-corpus validate_articulation.py timing-agreement check.
MIDI_TIME_TOLERANCE = .002


def percent(log2_values):
    return 100 * np.exp2(log2_values)


def delta_percent(log2_values):
    return 100 * np.expm1(np.log(2) * log2_values)


def reference_map(alignment):
    """Scalar median + bisect LIS, with the production tie-breaking policy."""
    groups = defaultdict(list)
    for score, performed in alignment.matches:
        if not score.is_grace:
            groups[round(score.onset_beats, 6)].append(performed.onset)
    positions = sorted(groups)
    times = [statistics.median(groups[p]) for p in positions]
    tails, indices, previous = [], [], [-1] * len(times)
    for i, t in enumerate(times):
        j = bisect_left(tails, t)
        if j:
            previous[i] = indices[j - 1]
        if j == len(tails):
            tails.append(t)
            indices.append(i)
        else:
            tails[j], indices[j] = t, i
    kept = []
    i = indices[-1] if indices else -1
    while i >= 0:
        kept.append(i)
        i = previous[i]
    kept.reverse()
    return [positions[i] for i in kept], [times[i] for i in kept], len(positions)


def reference_notes(sample, alignment, positions, times):
    """Recompute expected durations and exclusions with scalar interpolation."""
    def to_time(position):
        right = bisect_left(positions, position)
        if right < len(positions) and positions[right] == position:
            return times[right]
        left = right - 1
        share = (position - positions[left]) / (positions[right] - positions[left])
        return times[left] + share * (times[right] - times[left])

    rows, values, exclusions = [], {}, Counter()
    ornaments = {"trillmark", "tremolo", "mordent", "invertedmordent", "turn"}
    for i, (score, performed) in enumerate(alignment.matches):
        expected, value = float("nan"), float("nan")
        reason = ""
        if score.is_grace:
            reason = "grace"
        elif len(positions) < 2:
            reason = "outside_tempo_map"
        elif ornaments.intersection(score.attributes):
            reason = "ornament"
        elif score.onset_beats < positions[0] or score.offset_beats > positions[-1]:
            reason = "outside_tempo_map"
        else:
            expected = to_time(score.offset_beats) - to_time(score.onset_beats)
            if expected <= 0 or performed.duration <= 0:
                reason = "non_positive"
            else:
                value = math.log2(performed.duration / expected)
                values[i] = value
        if reason:
            exclusions[reason] += 1
        beat = next((j for j, (left, right) in enumerate(zip(sample.performance_beats[:-1], sample.performance_beats[1:]))
                     if left <= performed.onset < right), -1)
        rows.append({"key": sample.performance_key, "match_index": i, "note_id": performed.note_id,
                     "pitch": performed.pitch, "score_onset_beats": score.onset_beats,
                     "score_offset_beats": score.offset_beats, "score_attributes": "|".join(score.attributes),
                     "onset_seconds": performed.onset, "offset_seconds": performed.offset,
                     "actual_duration_seconds": performed.duration, "expected_duration_seconds": expected,
                     "note_log2": value, "duration_vs_expected_percent": float(percent(value)),
                     "used_for_note_feature": not bool(reason), "excluded_reason": reason,
                     "beat_index": beat, "display_interval": beat + 1 if beat >= 0 else None,
                     "used_for_beat_feature": not reason and beat >= 0 and sample.performance_beats[beat + 1] > sample.performance_beats[beat]})
    return rows, values, exclusions


def analyze(samples):
    inputs, note_rows, beat_rows, rows, pairs = [], [], [], [], []
    errors = {n: 0. for n in ("tempo_map", "note_log2", "raw", "common", "relative", "scale",
                             "standardized", "pair_difference", "scaled_pair_difference")}
    audit = {}
    for sample in samples:
        key = sample.performance_key
        alignment = load_match(sample.note_alignment_path)
        positions, times, position_count = reference_map(alignment)
        production_positions, production_times = build_tempo_map(alignment)
        errors["tempo_map"] = max(errors["tempo_map"], check(production_positions, np.asarray(positions)),
                                  check(production_times, np.asarray(times)))
        notes = extract_note_articulation(alignment)
        events, direct_notes, excluded = reference_notes(sample, alignment, positions, times)
        assert set(direct_notes) == set(notes.match_indices.tolist())
        errors["note_log2"] = max(errors["note_log2"], check(notes.values, np.array([direct_notes[int(i)] for i in notes.match_indices])))
        for reason, count in notes.excluded_counts.items():
            assert count == excluded[reason]
        feature = extract_articulation(alignment, sample.performance_beats)
        direct = np.full(len(feature.sequence), np.nan)
        counts = np.zeros(len(direct), dtype=int)
        for i in range(len(direct)):
            vals = [e["note_log2"] for e in events if e["used_for_note_feature"] and e["beat_index"] == i]
            counts[i] = len(vals)
            if vals and sample.performance_beats[i + 1] > sample.performance_beats[i]:
                direct[i] = statistics.median(vals)
        errors["raw"] = max(errors["raw"], check(feature.sequence.values, direct))
        np.testing.assert_array_equal(feature.note_counts, counts)
        np.testing.assert_array_equal(feature.sequence.mask, (counts > 0) & (np.diff(sample.performance_beats) > 0))
        inputs.append(PieceFeatureInput.from_asap_sample(sample, feature.sequence))
        note_rows.extend(events)
        audit[key] = {"matches": len(alignment.matches), "deletions": len(alignment.deletions),
                      "insertions": len(alignment.insertions), "used_notes": len(direct_notes),
                      "excluded_notes": dict(notes.excluded_counts), "tempo_map_positions": len(positions),
                      "tempo_map_dropped_positions": position_count - len(positions), "counts": counts}
    separated = separate_piece_feature(inputs)
    normalized = standardize_piece_feature(list(separated.values()), feature_name="articulation")
    matrix = np.array([np.where(i.sequence.mask & np.isfinite(i.sequence.values) & (np.diff(i.score_beats) > 0),
                                i.sequence.values, np.nan) for i in inputs])
    support = np.isfinite(matrix).sum(axis=0)
    common = np.full(matrix.shape[1], np.nan)
    for i in np.flatnonzero(support >= 2):
        common[i] = np.median(matrix[np.isfinite(matrix[:, i]), i])
    residual = matrix - common
    pooled = residual[np.isfinite(residual)]
    if not len(pooled):
        raise ValueError("No valid Articulation residuals with support >= 2")
    candidates = {"mad": float(1.4826 * np.median(np.abs(pooled - np.median(pooled)))),
                  "iqr": float(np.diff(np.percentile(pooled, [25, 75]))[0] / 1.3489795003921634),
                  "std": float(np.std(pooled, ddof=0))}
    method = next((name for name in ("mad", "iqr", "std") if candidates[name] > 1e-12), "unit")
    expected_scale = candidates.get(method, 1.)
    ref = next(iter(normalized.values()))
    assert ref.scale.method == method
    errors["scale"] = check(np.array([ref.scale.value]), np.array([expected_scale]))
    np.testing.assert_allclose([ref.scale.mad, ref.scale.iqr, ref.scale.std], list(candidates.values()), atol=1e-12)
    for sample, source in zip(samples, inputs):
        key, nf = sample.performance_key, normalized[sample.performance_key]
        raw, mask = source.sequence, source.sequence.mask & np.isfinite(source.sequence.values) & (np.diff(source.score_beats) > 0) & (support >= 2)
        np.testing.assert_array_equal(nf.raw.values, raw.values)
        np.testing.assert_array_equal(nf.raw.mask, raw.mask)
        np.testing.assert_array_equal(nf.common_support, support)
        np.testing.assert_array_equal(nf.common.mask, support >= 2)
        np.testing.assert_array_equal(nf.relative.mask, mask)
        np.testing.assert_array_equal(nf.standardized.mask, mask)
        expected = np.where(mask, raw.values - common, np.nan)
        errors["common"] = max(errors["common"], check(nf.common.values, common))
        errors["relative"] = max(errors["relative"], check(nf.relative.values, expected))
        errors["standardized"] = max(errors["standardized"], check(nf.standardized.values, expected / expected_scale))
        rows.append({"key": key, "performance": Path(key).stem, "robust": sample.robust_note_alignment,
                     "intervals": len(raw), "raw_valid": int(raw.mask.sum()), "relative_valid": int(mask.sum()),
                     **{k: v for k, v in audit[key].items() if k not in ("counts", "excluded_notes")},
                     **{f"excluded_{k}": v for k, v in audit[key]["excluded_notes"].items()},
                     **{f"{stage}_mean" + ("_log2" if stage != "standardized" else ""): float(np.mean(getattr(nf, stage).values[getattr(nf, stage).mask]))
                        for stage in ("raw", "common", "relative", "standardized")},
                     "raw_representative_percent": float(percent(np.mean(raw.values[raw.mask]))),
                     "relative_representative_delta_percent": float(delta_percent(np.mean(expected[mask]))),
                     "raw_range_log2": value_range(raw), "relative_range_log2": value_range(nf.relative),
                     "relative_mean_abs_log2": float(np.mean(np.abs(expected[mask]))), "scale": nf.scale.value})
        for i in range(len(raw)):
            beat_rows.append({"key": key, "beat_index": i, "display_interval": i + 1,
                              "score_interval_seconds": float(np.diff(sample.score_beats)[i]),
                              "performance_start_seconds": sample.performance_beats[i],
                              "performance_end_seconds": sample.performance_beats[i + 1],
                              "score_beat_type": sample.score_beat_types[i], "performance_beat_type": sample.performance_beat_types[i],
                              "note_count": int(audit[key]["counts"][i]),
                              **{stage: float(getattr(nf, stage).values[i]) for stage in ("raw", "common", "relative", "standardized")},
                              **{f"{stage}_mask": bool(getattr(nf, stage).mask[i]) for stage in ("raw", "common", "relative", "standardized")},
                              "raw_duration_percent": float(percent(raw.values[i])),
                              "common_duration_percent": float(percent(common[i])),
                              "relative_delta_percent": float(delta_percent(expected[i])),
                              "common_support": int(support[i]), "scale": nf.scale.value})
    features = list(normalized.values())
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            mask = a.relative.mask & b.relative.mask
            if not mask.any():
                continue
            before, after = (a.raw.values - b.raw.values)[mask], (a.relative.values - b.relative.values)[mask]
            scaled = (a.standardized.values - b.standardized.values)[mask]
            error, scaled_error = check(after, before), check(scaled, before / expected_scale)
            errors["pair_difference"] = max(errors["pair_difference"], error)
            errors["scaled_pair_difference"] = max(errors["scaled_pair_difference"], scaled_error)
            pairs.append({"a": a.performance_key, "b": b.performance_key, "shared_beats": int(mask.sum()),
                          "before_mean_abs_log2_difference": float(np.mean(np.abs(before))),
                          "after_mean_abs_log2_difference": float(np.mean(np.abs(after))),
                          "multiplicative_gap_percent": float(delta_percent(np.mean(np.abs(before)))),
                          "standardized_mean_abs_difference": float(np.mean(np.abs(scaled))),
                          "max_error": error, "max_scaled_error": scaled_error})
    return normalized, rows, beat_rows, note_rows, pairs, errors


def plot_examples(normalized, rows, notes, pairs, out, label, curve_beats):
    ordered = sorted(rows, key=lambda r: (r["raw_mean_log2"], r["key"]))
    selected = [ordered[int(round(q * (len(ordered) - 1)))] for q in (.25, .5, .75)]
    keys = [r["key"] for r in selected]
    ref = normalized[keys[1]]
    valid = np.flatnonzero(ref.raw.mask)
    beat = int(valid[np.argmin(np.abs(valid - (len(ref.raw) - 1) / 2))])
    events = sorted([n for n in notes if n["key"] == keys[1] and n["beat_index"] == beat and n["used_for_beat_feature"]], key=lambda n: n["match_index"])
    note = min(events, key=lambda n: abs(n["note_log2"] - ref.raw.values[beat]))
    fig, ax = chart("악보의 기대 길이와 실제로 누른 시간을 비교한다",
                    f"{label} | {Path(keys[1]).stem} | 악보 구간 {beat + 1} | MIDI pitch {note['pitch']}")
    ms = [1000 * note["expected_duration_seconds"], 1000 * note["actual_duration_seconds"]]
    ax.barh([1, 0], ms, color=[GRAY, BLUE], height=.45)
    for y, value in zip([1, 0], ms):
        ax.text(value + max(ms) * .03, y, f"{value:.1f} ms", va="center", fontsize=14)
    ax.set(yticks=[1, 0], yticklabels=["악보 음가로 계산한 기대 시간", "실제로 건반을 누른 시간"], xlabel="길이(ms)", xlim=(0, max(ms) * 1.35), ylim=(-.6, 1.6))
    ax.text(.98, .92, f"실제 ÷ 기대 = {note['duration_vs_expected_percent']:.1f}%", transform=ax.transAxes, ha="right", fontsize=17, color=BLUE)
    save(fig, out / "01_actual_vs_expected_duration.png", "기대 길이는 정렬된 음 시작 시각의 tempo map으로 환산한다. 실제 길이는 note-off까지이며 페달로 남은 소리는 포함하지 않는다.")

    fig, ax = chart("beat 값은 그 구간에서 시작한 음들의 중앙값이다",
                    f"{label} | {Path(keys[1]).stem} | 악보 구간 {beat + 1} | 유효 음 {len(events)}개")
    ax.bar(np.arange(len(events)), [n["duration_vs_expected_percent"] for n in events], color=BLUE, width=.6, label="각 음의 실제/기대 길이")
    beat_pct = float(percent(ref.raw.values[beat]))
    ax.axhline(beat_pct, color=ORANGE, ls="--", lw=2, label=f"beat 중앙값 {beat_pct:.1f}%")
    ax.axhline(100, color=GRAY, ls=":", label="기대 길이와 동일 = 100%")
    ax.set(xticks=np.arange(len(events)), xticklabels=[f"음 {i + 1}\n(pitch {n['pitch']})" for i, n in enumerate(events)], xlabel="이 beat에서 시작한 정렬 음", ylabel="기대 길이 대비 실제 누른 시간(%)", ylim=(0, max(110., max(n["duration_vs_expected_percent"] for n in events) * 1.2)))
    legend(fig, ax)
    save(fig, out / "02_notes_to_beat_median.png", "log2(실제/기대)의 중앙값을 %로 되돌린 표시다. 음 개수가 짝수면 가운데 두 비율의 기하평균이다.", bottom=.23)

    width = min(curve_beats, len(ref.raw))
    start = (len(ref.raw) - width) // 2
    section = slice(start, start + width)
    x = np.arange(start + 1, start + width + 1)
    specs = [("raw", "03_raw_and_common.png", "같은 악보에서도 음을 누르는 길이는 다르다", "기대 길이 대비 실제 누른 시간(%)", percent),
             ("relative", "04_relative.png", "공통 제거 후: 다른 연주들보다 더 길거나 짧게", "공통 길이 비율 대비 차이(%)", delta_percent),
             ("standardized", "05_standardized.png", "MAD로 단위를 맞추어도 연주 차이는 남는다", "상대 Articulation ÷ 작품 MAD scale", lambda v: v)]
    for stage, filename, title, ylabel, convert in specs:
        fig, ax = chart(title, f"{label} | 전체 {len(rows)}연주 중 평균 길이 순위 25·50·75% 예시 | 동일한 중앙 {width}구간")
        if stage == "raw":
            ax.plot(x, percent(ref.common.values[section]), color="#242b27", lw=2.4, ls="--", label=f"작품 공통: {len(rows)}연주 위치별 중앙값")
        for key, role, color in zip(keys, ROLES, COLORS):
            seq = getattr(normalized[key], stage)
            values = np.where(seq.mask, seq.values, np.nan)[section]
            ax.plot(x, convert(values), color=color, lw=1.8, marker="o", ms=3.5, label=f"{role} · {Path(key).stem}")
        ax.axhline(100 if stage == "raw" else 0, color=GRAY, lw=1, ls=":")
        ax.set(xlabel="악보 beat 구간 번호(1부터)", ylabel=ylabel, xlim=(x[0], x[-1]))
        legend(fig, ax)
        note_text = {"raw": "100%는 기대 음가만큼 누름. 짧거나 긴 것은 음표 길이의 비율이며, 스타카토/레가토의 확정 판정이 아니다.",
                     "relative": "공통 60%, 원본 66%이면 상대 +10%다. 6%p가 아니다. 원래 log2 상대값에는 log2(1.1)이 저장된다.",
                     "standardized": f"scale={ref.scale.value:.5f} ({ref.scale.method.upper()}). 1은 scale 한 단위의 편차이며, 길이가 1% 길다는 뜻이 아니다."}[stage]
        save(fig, out / filename, note_text + "\n평활화·결측 보간·값 clipping 없음.", bottom=.25)

    y = np.arange(len(ordered))
    fig, ax = chart("곡 전체로 보면 누르는 길이의 성향이 구분된다", f"{label} | 전체 유효 beat | 대표값 = log2 beat 값 평균을 비율로 변환", height=max(5.8, 2.5 + .36 * len(rows)))
    values = [r["relative_representative_delta_percent"] for r in ordered]
    ax.hlines(y, 0, values, color="#c3c6bd", lw=2)
    ax.scatter(values, y, color=BLUE, s=65)
    span = max(max(values) - min(values), 1.)
    for yy, row in zip(y, ordered):
        ax.text(max(values) + .09 * span, yy, f"{row['relative_representative_delta_percent']:+.1f}%  (원본 {row['raw_representative_percent']:.1f}%)", va="center", fontsize=11)
    ax.axvline(0, color=GRAY, ls="--")
    ax.set(yticks=y, yticklabels=[r["performance"] for r in ordered], xlabel="작품 공통 대비 대표 길이 비율의 차이(%)", xlim=(min(min(values), 0) - .12 * span, max(max(values), 0) + .85 * span))
    save(fig, out / "06_all_performance_tendencies.png", "양수: 대체로 공통보다 길게, 음수: 대체로 짧게. log2 평균의 역변환은 기하평균이며, 음별 길이의 산술평균이 아니다.")

    fig, ax = chart("공통값을 빼도 연주끼리의 차이는 그대로다", f"{label} | {len(pairs)}연주 쌍 | 두 연주가 모두 유효한 beat만 비교")
    before = [p["before_mean_abs_log2_difference"] for p in pairs]
    after = [p["after_mean_abs_log2_difference"] for p in pairs]
    bound = max(before + after) * 1.2
    ax.plot([0, bound], [0, bound], ls="--", color=GRAY, label="차이가 같으면 이 선 위")
    ax.scatter(before, after, color=BLUE, s=65, alpha=.75, label="연주 쌍 하나")
    ax.set(xlabel="공통 제거 전 평균 절대 차이(log2)", ylabel="공통 제거 후 평균 절대 차이(log2)", xlim=(0, bound), ylim=(0, bound))
    ax.legend(loc="upper left", fontsize=10)
    save(fig, out / "07_pairwise_difference_preserved.png", "같은 위치에서 같은 공통값을 빼므로 연주 간 log2 차이는 보존된다. 이 그림은 추천 성능이나 청취 정확도 검증은 아니다.")

    fig, ax = chart("평균 길이와 곡 안에서 변하는 폭은 따로 보자", f"{label} | 전체 유효 beat | 변화 폭 = 95백분위 − 5백분위(log2)", height=max(5.8, 2.5 + .36 * len(rows)))
    raw_ranges = [r["raw_range_log2"] for r in ordered]
    relative_ranges = [r["relative_range_log2"] for r in ordered]
    for yy, a, b in zip(y, raw_ranges, relative_ranges):
        ax.plot([a, b], [yy, yy], color="#c3c6bd", lw=2)
    ax.scatter(raw_ranges, y, color=GRAY, s=65, label="원본 변화 폭")
    ax.scatter(relative_ranges, y, color=ORANGE, s=65, label="공통 제거 후 편차의 변화 폭")
    ax.set(yticks=y, yticklabels=[r["performance"] for r in ordered], xlabel="5~95백분위 폭(log2; 1 = 길이 비율 두 배)", xlim=(0, max(raw_ranges + relative_ranges) * 1.15))
    legend(fig, ax)
    save(fig, out / "08_all_performance_ranges.png", "원본은 악보 위치의 공통 길이도 포함한다. 상대 변화 폭은 공통에서 벗어나는 정도이며, 제거 후 반드시 작아지는 것은 아니다.", bottom=.23)
    return selected, {"key": keys[1], "display_interval": beat + 1, "notes": len(events),
                      "beat_raw_log2": float(ref.raw.values[beat]), "beat_duration_percent": beat_pct,
                      "note": note}, [start + 1, start + width]


def write_report(stats, rows, out):
    s, e = stats["scale"], stats["duration_example"]
    n, extreme = e["note"], stats["largest_standardized_beat"]
    ordered = sorted(rows, key=lambda r: (r["raw_mean_log2"], r["key"]))
    table = "\n".join(f"| {r['performance']} | {r['raw_representative_percent']:.1f}% | {r['relative_representative_delta_percent']:+.1f}% | {r['raw_range_log2']:.3f} | {r['relative_range_log2']:.3f} |" for r in ordered)
    text = f"""# Articulation 그림 읽기

`_articulation/`은 음 길이부터 공통 제거까지 쉽게 읽는 개별 예시 폴더다.
기존 [articulation/ 전체 데이터 보고서](../articulation/README.md)는 별도로 보존한다.
모든 PNG는 **파일 하나에 그래프 하나**이며, 표의 순서대로 보면 된다.

| 순서 | 그림 | 확인할 내용 |
|---|---|---|
| 1 | [실제 길이와 기대 길이](01_actual_vs_expected_duration.png) | Articulation이 무엇을 재는가 |
| 2 | [음표에서 beat 중앙값으로](02_notes_to_beat_median.png) | 여러 음을 beat 값 하나로 어떻게 묶는가 |
| 3 | [원본과 작품 공통 곡선](03_raw_and_common.png) | 같은 악보에서도 길이 비율이 다른가 |
| 4 | [공통 제거 후 상대 곡선](04_relative.png) | 공통보다 얼마나 길거나 짧게 누르는가 |
| 5 | [MAD 표준화 곡선](05_standardized.png) | 모델에 넣을 단위를 맞춰도 차이가 남는가 |
| 6 | [전체 연주 성향](06_all_performance_tendencies.png) | 곡 전체에서 대체로 길게/짧게 누르는가 |
| 7 | [연주 쌍 차이 보존](07_pairwise_difference_preserved.png) | 공통 제거가 연주 차이를 지우지 않았는가 |
| 8 | [전체 연주 변화 폭](08_all_performance_ranges.png) | 평균이 비슷해도 위치별 변화 폭이 다른가 |

## 1. 첫 그림: 기대 시간 중 어느 정도를 눌렀나

Articulation은 **실제로 건반을 누른 시간 / 그 자리의 빠르기로 환산한 악보 음가**다.
기대 시간은 악보 MIDI의 고정 tempo가 아니라 각 연주의 정렬 음 시작 시각에서 만든
tempo map으로 계산한다. 같은 악보 위치의 onset 중앙값을 묶고, 시간이 순증가하는
최장 부분열(LIS)을 남긴 뒤 악보 음의 시작·끝 위치를 보간한다.
빠르기와 루바토는 이 기대 길이에 반영된다.

이번 예시는 `{Path(e['key']).stem}`, 악보 구간 {e['display_interval']}, pitch {n['pitch']}다.
기대 길이 **{1000*n['expected_duration_seconds']:.1f} ms**, 실제 누른 길이
**{1000*n['actual_duration_seconds']:.1f} ms**이므로 비율은 **{n['duration_vs_expected_percent']:.1f}%**다.
100%는 기대 길이와 같고, 50%는 절반만큼 누른 것이다.
100%를 넘으면 기대 길이보다 오래 눌렀다는 뜻이다.

짧은 쪽·긴 쪽의 연주 성향을 나타내는 대리 지표로 해석한다.
이 값 하나로 스타카토·레가토를 확정하지 않는다. match의 note-off를 사용하므로
페달로 지속되는 실제 소리의 길이와도 다르다.

## 2. 두 번째 그림: 음마다 다르므로 beat 중앙값으로 묶는다

그 구간에서 시작한 유효 정렬 음 {e['notes']}개의 길이 비율을 보여준다.
주황 점선은 beat 대표값 **{e['beat_duration_percent']:.1f}%**다.
코드는 음마다 `log2(실제/기대)`를 구한 뒤 **log2 값의 중앙값**을 저장한다.
그림의 %는 이 값을 다시 비율로 변환한 표시다.
음이 짝수 개면 가운데 두 비율의 기하평균이며, 비율의 산술 중앙값과는 다를 수 있다.
각 beat는 요약 단계에서 같은 가중치를 갖고, 전체 음을 한꺼번에 평균내지 않는다.

## 3. 세 번째 그림: 공통 모양과 연주 차이를 함께 본다

원본 색 선은 기대 길이를 100%로 한 beat 값이고, 검정 점선은 위치별 작품 공통값이다.
색 선들이 같이 움직이는 곳은 데이터 안의 공통 패턴, 같은 위치에서 벌어지는 곳은
연주별 편차로 읽는다. 점선은 음악적 정답이 아니라 현재 비교 연주들의 중앙 패턴이다.

분석 작품은 `{stats['work']}`, {stats['performances']}연주·연주당 {stats['intervals_per_performance']}구간이다.
곡선은 원본 log2 평균의 순위 25·50·75%에 가까운 연주를 고르고,
곡 중앙 **{stats['curve_display_interval_range'][0]}~{stats['curve_display_interval_range'][1]}구간**을 표시한다.
차이가 가장 큰 구간이나 연주를 검색해서 고른 예시가 아니다.
평활화·결측 보간·clipping 없이 그렸다. 마스킹된 값은 선이 끊기게 표시한다.

## 4. 네 번째 그림: 기준이 100%에서 0으로 바뀐다

상대값은 기존 log2 공간에서 `raw - common`이다.
그림은 이해하기 쉽게 `100 × (2^relative - 1)`로 표시한다.
**0은 공통 길이 비율과 같음, +10%는 공통 비율의 1.1배, −10%는 0.9배**다.
예를 들어 원본 66%, 공통 60%면 상대 +10%다. 6%p 차이가 아니다.
연주마다 그 위치의 tempo가 다르므로 서로의 실제 ms 길이를 직접 나눈 값도 아니다.

큰 봉우리는 원본과 공통을 함께 확인한다. 이번 최대값은
`{Path(extreme['key']).stem}`의 {extreme['display_interval']}구간이다.
원본 **{extreme['raw_duration_percent']:.1f}%**, 공통 **{extreme['common_duration_percent']:.1f}%**이므로
상대 **{extreme['relative_delta_percent']:+.1f}%**다.
원본이 터무니없이 긴 것이 아니라 공통 비율이 작은 위치에서 차이가 커진 사례다.
이 beat의 유효 음 {extreme['note_count']}개를 원본 match/MIDI 길이까지 확인했다.
match와 MIDI의 시간 양자화 차이를 고려해 기존 전체 보고서와 같은
{1000*stats['midi_time_tolerance_seconds']:.1f} ms 허용 오차로 대조했다.
확인 음들의 최대 시각 차이는 {1000*max(c['max_timestamp_error_seconds'] for c in stats['example_and_extreme_midi_checks']):.3f} ms다.
이 수치 확인만으로 정렬이나 음악적 해석이 정확하다고 판정할 수는 없다.

## 5. 다섯 번째 그림: MAD로 모델 입력 단위를 맞춘다

전체 유효 residual을 모아 `scale = 1.4826 × median(|r - median(r)|)`를 계산한다.
이번 MAD **{s['mad']:.6f}**, IQR/1.34898 **{s['iqr']:.6f}**, SD **{s['std']:.6f}**다.
MAD와 IQR은 비슷하고 SD는 더 크므로 중앙 분포보다 꼬리의 영향이 있음을 볼 수 있다.
기존 정책인 **MAD 표준화**를 유지한다. Articulation은 이미 log2 비율이므로
추가 log 변환 없이 `standardized = relative / scale`을 사용한다.

1은 scale 한 단위의 편차다. 1% 또는 1ms라는 뜻이 아니다.
이번 scale 한 단위의 양수 편차는 공통 비율의 약 **{delta_percent(s['value']):.1f}% 증가**에 해당한다.
최대 |standardized|는 **{stats['standardized_max_abs']:.2f}**이며 이를 잘라내지 않았다.
MAD가 0에 가까우면 IQR → SD → 단위 scale 1의 기존 fallback을 사용한다.
표준화 때 residual 평균을 다시 빼지 않으므로, 상대 0은 그대로 공통과 같다는 뜻이다.

## 6. 여섯 번째 그림: 작품 안에서 성향이 어느 정도 다른가

전체 유효 beat의 log2 평균을 비율로 되돌린 **기하평균 대표값**이다.
원본 대표값은 **{stats['raw_representative_percent_range'][0]:.1f}~{stats['raw_representative_percent_range'][1]:.1f}%**,
공통 대비 상대 성향은 **{stats['relative_representative_delta_percent_range'][0]:+.1f}~{stats['relative_representative_delta_percent_range'][1]:+.1f}%**다.
공통 제거 후에도 연주별 성향이 모두 0으로 붕괴하지 않는다.
가운데에 가까운 평균이라도 특정 위치의 편차는 클 수 있다.

| 연주 | 원본 대표 길이 | 공통 대비 상대 성향 | 원본 5~95% 폭(log2) | 상대 폭(log2) |
|---|---:|---:|---:|---:|
{table}

## 7~8. 연주 차이 보존과 곡 안 변화 폭은 구분한다

7번은 {stats['pair_count']}연주 쌍의 같은 위치 차이 보존을 확인한다.
`(raw A - common) - (raw B - common) = raw A - raw B`이므로 점들이 대각선 위에 놓인다.
쌍별 평균 절대 log2 차이의 중앙값은 **{stats['median_pair_mean_abs_log2_difference']:.3f}**다.
비율로 되돌린 평균 배율 차이의 중앙값은 **{stats['median_pair_multiplicative_gap_percent']:.1f}%**다.
이는 `2^(mean(|log2 A - log2 B|)) - 1`의 변환이며,
원본 비율의 평균 %p 차이나 지각적으로 느끼는 차이의 크기는 아니다.
쌍별 차이 보존 최대 오차는 {stats['extraction_checks']['pair_difference']:.2e}다.

8번은 한 연주의 위치별 5~95백분위 범위를 비교한다.
폭 1(log2)은 양 끝의 길이 비율이 두 배라는 뜻이다.
공통 제거 후 상대 변화 폭이 남는다는 점을 확인한다.
연주끼리의 차이가 보존되는 것과 한 연주의 곡 안 변화 폭이 변하는 것은 서로 다른 관계다.
일반적으로 공통 제거 후 폭이 반드시 작아지는 것은 아니다.

## 데이터·검증 범위

기본값은 `robust_note_alignment=True`인 연주만 포함한다. 이번 {stats['robust_performances']}연주는 모두 robust다.
`--include-non-robust`로 호출자가 정책을 바꿀 수 있다. 정렬이 없는 연주는 제외하고
제외 목록을 JSON에 기록한다. 반복 구조가 다른 nASAP 변형은 같은 그룹에 묶지 않는다.

이번 원본/상대 유효값은 각각 {stats['valid_raw_values']}/{stats['valid_relative_values']}개,
beat별 공통 지원 수는 {stats['common_support_min']}~{stats['common_support_max']}다.
지원 수 2 미만은 common/relative/standardized를 NaN + mask=False로 처리한다.
음이 없거나 기대 길이를 계산할 수 없는 위치의 기존 mask도 유지한다.
이번 작품에 결측 beat가 없다는 것이 다른 작품도 그렇다는 뜻은 아니다.

정렬 음 {stats['matched_notes']}개 중 음 feature에 사용한 값은 {stats['used_note_values']}개다.
꾸밈음 {stats['excluded_note_counts']['grace']}개, 장식음 {stats['excluded_note_counts']['ornament']}개,
tempo map 범위 밖 {stats['excluded_note_counts']['outside_tempo_map']}개,
양수가 아닌 길이 {stats['excluded_note_counts']['non_positive']}개를 기존 정책대로 제외했다.
tempo map에서 역행 등으로 제외된 악보 onset 위치는 총 {stats['tempo_map_dropped_positions']}개다.

음 길이·tempo map·beat 중앙값을 별도 scalar 계산과 대조하고,
common/support/residual/MAD/standardized와 mask를 재계산했다.
기존 LIS 동점 선택 정책 안에서의 수치 일치 검증이다.
다른 LIS 후보, 성부별 표현, 정렬 ground truth, 청취 평가, 추천 성능은 이 예시로 검증하지 않는다.
작품 한 개의 결과만으로 다른 작품에서도 해석 분리가 성공했다고 일반화하지 않는다.

## 재현과 파일

저장소 루트에서:

```bash
.venv/bin/python classicfy-ai/scripts/validate_articulation_examples.py
# 다른 경로/작품이나 non-robust 포함 정책을 쓸 때:
.venv/bin/python classicfy-ai/scripts/validate_articulation_examples.py \\
  --asap-root ../datasets/ASAP --nasap-root ../datasets/nASAP \\
  --work Bach/Fugue/bwv_848 --include-non-robust --out /tmp/articulation_examples
```

기본 경로는 스크립트 위치 기준 `Classicfy/datasets/{{ASAP,nASAP}}`과 이 폴더다.
CSV 5개와 JSON에 곡 전체 값을 남긴다. 그림은 이해를 위한 일부 곡선이다.

| 파일 | 내용 |
|---|---|
| [beat_features.csv](beat_features.csv) | raw/common/relative/standardized(log2, 표준화값은 무차원), 각 mask/support/음 개수/표시 % |
| [note_features.csv](note_features.csv) | 모든 정렬 음의 원본 실제·기대 길이, match 인덱스, 사용 여부·제외 사유·beat 위치 |
| [performance_summary.csv](performance_summary.csv) | 전체 연주 요약, 정렬 품질·제외 수·변화 폭 |
| [pairwise_distances.csv](pairwise_distances.csv) | 공통 제거 전후 쌍별 차이와 수치 오차 |
| [scale.csv](scale.csv) | 선택 scale과 MAD/IQR/SD |
| [stats.json](stats.json) | 선택 규칙·예시·최대 표준화 구간·MIDI 대조·dataset revision·입력/코드 hash |
| [재현 스크립트](../../scripts/validate_articulation_examples.py) | 계산·그림·CSV·JSON·이 안내 재생성 |

ASAP revision: `{stats['asap_revision']}`  
nASAP revision: `{stats['nasap_revision']}`
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asap-root", type=Path, default=ai.parent.parent / "datasets/ASAP")
    parser.add_argument("--nasap-root", type=Path, default=ai.parent.parent / "datasets/nASAP")
    parser.add_argument("--work", default="Bach/Fugue/bwv_848")
    parser.add_argument("--curve-beats", type=int, default=32)
    parser.add_argument("--include-non-robust", action="store_true")
    parser.add_argument("--out", type=Path, default=ai / "analysis/_articulation")
    args = parser.parse_args()
    if args.curve_beats < 2:
        parser.error("--curve-beats must be at least 2")
    root, nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    candidates = collect_samples(root, nasap_root, args.work)
    samples, excluded = [], []
    for sample in candidates:
        reason = "missing_note_alignment" if sample.note_alignment_path is None else (
            "non_robust_or_unknown" if not args.include_non_robust and sample.robust_note_alignment is not True else "")
        if reason:
            excluded.append({"key": sample.performance_key, "reason": reason})
        else:
            samples.append(sample)
    if len(samples) < 3:
        raise ValueError("At least three eligible performances with note alignment are required")
    normalized, rows, beats, notes, pairs, errors = analyze(samples)
    if any(not nf.relative.mask.any() for nf in normalized.values()):
        raise ValueError("Each plotted performer must have valid Articulation residuals")
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    label = "Bach · Fugue BWV 848" if args.work == "Bach/Fugue/bwv_848" else args.work
    selected, example, curve_range = plot_examples(normalized, rows, notes, pairs, args.out, label, args.curve_beats)
    ref = next(iter(normalized.values()))
    for name, contents in (("performance_summary.csv", rows), ("beat_features.csv", beats),
                           ("note_features.csv", notes), ("pairwise_distances.csv", pairs),
                           ("scale.csv", [asdict(ref.scale)])):
        write_csv(args.out / name, contents)
    def revision(path):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    paths = [root / "asap_annotations.json", root / "metadata.csv", nasap_root / "metadata.csv",
             Path(__file__), Path(__file__).with_name("validate_tempo.py"), Path(__file__).with_name("validate_dynamics.py"),
             *sorted((ai / "src/features").glob("*.py")), *sorted((ai / "src/preprocessing").glob("*.py")),
             *[s.note_alignment_path for s in samples], *[s.performance_path for s in samples]]
    pooled = np.concatenate([nf.relative.values[nf.relative.mask] for nf in normalized.values()])
    stats = {"work": args.work, "performances": len(samples), "robust_performances": sum(s.robust_note_alignment is True for s in samples),
             "include_non_robust": args.include_non_robust, "excluded_performances": excluded,
             "intervals_per_performance": len(ref.raw), "valid_raw_values": sum(r["raw_valid"] for r in rows),
             "valid_relative_values": sum(r["relative_valid"] for r in rows), "matched_notes": len(notes),
             "used_note_values": sum(n["used_for_note_feature"] for n in notes),
             "excluded_note_counts": {k: sum(r[f"excluded_{k}"] for r in rows) for k in ("grace", "ornament", "outside_tempo_map", "non_positive")},
             "tempo_map_dropped_positions": sum(r["tempo_map_dropped_positions"] for r in rows),
             "common_support_min": int(ref.common_support.min()), "common_support_max": int(ref.common_support.max()),
             "minimum_support": 2, "scale": asdict(ref.scale), "residual_zero_ratio": float(np.mean(pooled == 0)),
             "standardized_max_abs": float(np.max(np.abs(pooled / ref.scale.value))),
             "raw_representative_percent_range": [min(r["raw_representative_percent"] for r in rows), max(r["raw_representative_percent"] for r in rows)],
             "relative_representative_delta_percent_range": [min(r["relative_representative_delta_percent"] for r in rows), max(r["relative_representative_delta_percent"] for r in rows)],
             "pair_count": len(pairs), "median_pair_mean_abs_log2_difference": float(np.median([p["before_mean_abs_log2_difference"] for p in pairs])),
             "median_pair_multiplicative_gap_percent": float(np.median([p["multiplicative_gap_percent"] for p in pairs])),
             "selected_performances": [{"role": role, "key": row["key"], "raw_representative_percent": row["raw_representative_percent"]} for role, row in zip(ROLES, selected)],
             "curve_selection_rule": "nearest ranks to 25th, 50th and 75th percentiles of mean raw log2 Articulation",
             "curve_display_interval_range": curve_range, "duration_example": example,
             "example_rule": "nearest midpoint valid beat of the median-mean-log2-rank performer; eligible note nearest the beat median; ties by match index; no maximum-difference search",
             "extraction_checks": errors, "asap_revision": revision(root), "nasap_revision": revision(nasap_root),
             "inputs_and_code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest(),
             "clipping": None, "smoothing": None,
             "mask_policy": "positive beat width and eligible matched notes; relative additionally finite raw, positive score width and common support >= 2; no new Tempo masks",
             "validation_scope": "Numerical duration/tempo-map/beat/normalization consistency under the existing alignment and LIS tie policy, not alignment ground truth, acoustic articulation or recommendation performance"}
    extreme = max((b for b in beats if b["standardized_mask"]), key=lambda b: abs(b["standardized"]))
    stats["largest_standardized_beat"] = extreme
    checks = []
    for key, chosen in ((example["key"], [example["note"]]),
                        (extreme["key"], [n for n in notes if n["key"] == extreme["key"] and n["beat_index"] == extreme["beat_index"] and n["used_for_beat_feature"]])):
        sample = next(s for s in samples if s.performance_key == key)
        midi = load_midi(sample.performance_path)
        for event in chosen:
            matching = min((n for n in midi.notes if n.pitch == event["pitch"]), key=lambda n: abs(n.start - event["onset_seconds"]))
            error = max(abs(matching.start - event["onset_seconds"]), abs(matching.end - event["offset_seconds"]))
            if error > MIDI_TIME_TOLERANCE:
                raise ValueError(f"Example match/MIDI note timestamps disagree: {key} / {event['match_index']}")
            checks.append({"key": key, "match_index": event["match_index"], "pitch": event["pitch"], "max_timestamp_error_seconds": error})
    stats["example_and_extreme_midi_checks"] = checks
    stats["midi_time_tolerance_seconds"] = MIDI_TIME_TOLERANCE
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(stats, rows, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
