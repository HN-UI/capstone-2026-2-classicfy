"""Create readable, one-chart-per-file Pedaling analyses of an ASAP piece.

Run from the repository root:
    .venv/bin/python classicfy-ai/scripts/validate_pedaling.py

Existing extraction, common subtraction and channel-specific SD scaling are used
unchanged. Raw CC64 events and all beat values remain available in CSV exports.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from validate_tempo import BLUE, GRAY, ORANGE, chart, collect, plt, save, setup_style, write_csv
from matplotlib.ticker import FuncFormatter, MaxNLocator
from features import PieceFeatureInput, extract_pedaling, separate_piece_feature, standardize_piece_feature
from preprocessing import load_midi

CHANNELS = ("depth", "down_ratio", "changes")
LABELS = {"depth": "페달 깊이", "down_ratio": "페달 on 시간 비율", "changes": "페달 on/off 전환 횟수"}
MULTIPLIERS = {"depth": 100., "down_ratio": 100., "changes": 1.}
UNITS = {"depth": "%", "down_ratio": "%", "changes": "회/beat"}
COLORS = (BLUE, GRAY, ORANGE)
ROLES = ("얕은 편", "중간", "깊은 편")


def reference_pedal(midi, beats):
    """Independent direct segment integration, not the extractor's cumulative integral."""
    times = np.array([p.time for p in midi.pedals])
    values = np.array([p.value for p in midi.pedals])
    previous = np.r_[0, values[:-1] >= 64] if len(values) else np.array([])
    transitions = times[(values >= 64) != previous] if len(values) else np.array([])
    result = {name: np.zeros(len(beats) - 1) for name in CHANNELS}
    for i, (left, right) in enumerate(zip(beats[:-1], beats[1:])):
        if right <= left:
            continue
        inside = times[(times > left) & (times < right)]
        edges = np.r_[left, np.unique(inside), right]
        midpoint = (edges[:-1] + edges[1:]) / 2
        active = np.searchsorted(times, midpoint, side="right") - 1
        signal = np.zeros(len(midpoint))
        valid = active >= 0
        signal[valid] = values[active[valid]]
        weights = np.diff(edges) / (right - left)
        result["depth"][i] = np.sum(signal / 127 * weights)
        result["down_ratio"][i] = np.sum((signal >= 64) * weights)
        result["changes"][i] = np.count_nonzero((transitions >= left) & (transitions < right))
    return result


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    valid = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[valid] - expected[valid]))) if valid.any() else 0.


def analyze(samples, tempo):
    midis, pedals, events, beat_rows, rows, pairs, scale_rows = {}, {}, [], [], [], [], []
    errors = {name: 0. for name in ("raw_depth", "raw_down_ratio", "raw_changes", "common", "relative", "scale", "standardized", "pair_difference", "scaled_pair_difference")}
    original = {}
    for sample in samples:
        key = sample.performance_key
        midi = load_midi(sample.performance_path)
        feature = extract_pedaling(midi, sample.performance_beats)
        midis[key], pedals[key] = midi, feature
        direct = reference_pedal(midi, sample.performance_beats)
        for name in CHANNELS:
            seq = getattr(feature, name)
            original[key, name] = (seq.values.copy(), seq.mask.copy())
            np.testing.assert_array_equal(seq.mask, np.diff(sample.performance_beats) > 0)
            errors[f"raw_{name}"] = max(errors[f"raw_{name}"], check(seq.values, direct[name]))
        events.extend({"key": key, "event_index": i, "time_seconds": event.time,
                       "value_cc64": event.value, "instrument_idx": event.instrument_idx,
                       "on": event.value >= 64} for i, event in enumerate(midi.pedals))

    normalized = {}
    for name in CHANNELS:
        inputs = [PieceFeatureInput.from_asap_sample(sample, getattr(pedals[sample.performance_key], name)) for sample in samples]
        separated = separate_piece_feature(inputs)
        fitted = standardize_piece_feature(list(separated.values()), feature_name=f"pedal_{name}")
        normalized[name] = fitted
        reference = next(iter(fitted.values()))
        matrix = np.array([np.where(item.sequence.mask & np.isfinite(item.sequence.values) & (np.diff(item.score_beats) > 0), item.sequence.values, np.nan) for item in inputs])
        support = np.isfinite(matrix).sum(axis=0)
        common = np.full(matrix.shape[1], np.nan)
        for i in np.flatnonzero(support >= 2):
            common[i] = np.median(matrix[np.isfinite(matrix[:, i]), i])
        residual = matrix - common
        pooled = residual[np.isfinite(residual)]
        sd = float(np.std(pooled, ddof=0))
        expected_scale = sd if sd > 1e-12 else 1.
        errors["scale"] = max(errors["scale"], check(np.array([reference.scale.value]), np.array([expected_scale])))
        np.testing.assert_allclose([reference.scale.mad, reference.scale.iqr, reference.scale.std],
                                   [1.4826 * np.median(np.abs(pooled - np.median(pooled))),
                                    np.diff(np.percentile(pooled, [25, 75]))[0] / 1.3489795003921634, sd], rtol=0, atol=1e-12)
        scale_rows.append({"channel": name, **asdict(reference.scale), "raw_zero_ratio": float(np.mean(matrix[np.isfinite(matrix)] == 0)),
                           "relative_zero_ratio": float(np.mean(pooled == 0)), "common_mean": float(np.mean(common[np.isfinite(common)])),
                           "standardized_max_abs": float(np.max(np.abs(pooled / expected_scale)))})
        for sample in samples:
            key, nf = sample.performance_key, fitted[sample.performance_key]
            raw_values, raw_mask = original[key, name]
            np.testing.assert_array_equal(nf.raw.values, raw_values)
            np.testing.assert_array_equal(nf.raw.mask, raw_mask)
            np.testing.assert_array_equal(nf.common_support, support)
            np.testing.assert_array_equal(nf.common.mask, support >= 2)
            expected_mask = raw_mask & np.isfinite(raw_values) & (np.diff(sample.score_beats) > 0) & (support >= 2)
            np.testing.assert_array_equal(nf.relative.mask, expected_mask)
            np.testing.assert_array_equal(nf.standardized.mask, expected_mask)
            errors["common"] = max(errors["common"], check(nf.common.values, common))
            expected = np.where(expected_mask, raw_values - common, np.nan)
            errors["relative"] = max(errors["relative"], check(nf.relative.values, expected))
            errors["standardized"] = max(errors["standardized"], check(nf.standardized.values, expected / expected_scale))
            rows.append({"channel": name, "key": key, "performance": Path(key).stem,
                         "intervals": len(nf.raw), "raw_valid": int(nf.raw.mask.sum()),
                         "relative_valid": int(nf.relative.mask.sum()), "standardized_valid": int(nf.standardized.mask.sum()),
                         "cc64_events": len(midis[key].pedals), "cc64_unique_values": len({p.value for p in midis[key].pedals}),
                         "cc64_instruments": len({p.instrument_idx for p in midis[key].pedals}),
                         "no_cc64_events": not bool(midis[key].pedals), "unit": UNITS[name],
                         "raw_mean": float(np.mean(nf.raw.values[nf.raw.mask])),
                         "common_mean": float(np.mean(nf.common.values[nf.relative.mask])),
                         "relative_mean": float(np.mean(nf.relative.values[nf.relative.mask])),
                         "relative_mean_abs": float(np.mean(np.abs(nf.relative.values[nf.relative.mask]))),
                         "standardized_mean": float(np.mean(nf.standardized.values[nf.standardized.mask])),
                         "scale": nf.scale.value,
                         "raw_mean_display": float(MULTIPLIERS[name] * np.mean(nf.raw.values[nf.raw.mask]))})
            for i, interval in enumerate(tempo[key].intervals):
                beat_rows.append({"channel": name, "key": key, "beat_index": i, "display_interval": i + 1,
                                  "score_interval_seconds": float(np.diff(sample.score_beats)[i]),
                                  "performance_interval_seconds": float(np.diff(sample.performance_beats)[i]),
                                  "score_beat_type": sample.score_beat_types[i], "performance_beat_type": sample.performance_beat_types[i],
                                  "tempo_status_context": interval.status,
                                  **{stage: float(getattr(nf, stage).values[i]) for stage in ("raw", "common", "relative", "standardized")},
                                  **{f"{stage}_mask": bool(getattr(nf, stage).mask[i]) for stage in ("raw", "common", "relative", "standardized")},
                                  "common_support": int(nf.common_support[i]), "scale": nf.scale.value})
        values = list(fitted.values())
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                shared = a.relative.mask & b.relative.mask
                if not shared.any():
                    continue
                before = (a.raw.values - b.raw.values)[shared]
                after = (a.relative.values - b.relative.values)[shared]
                scaled = (a.standardized.values - b.standardized.values)[shared]
                error = check(after, before)
                scaled_error = check(scaled, before / expected_scale)
                errors["pair_difference"] = max(errors["pair_difference"], error)
                errors["scaled_pair_difference"] = max(errors["scaled_pair_difference"], scaled_error)
                pairs.append({"channel": name, "a": a.performance_key, "b": b.performance_key, "shared_beats": int(shared.sum()),
                              "before_mean_abs_difference": float(np.mean(np.abs(before))),
                              "after_mean_abs_difference": float(np.mean(np.abs(after))),
                              "before_median_abs_difference": float(np.median(np.abs(before))),
                              "different_beat_ratio": float(np.mean(np.abs(before) > 1e-12)),
                              "standardized_mean_abs_difference": float(np.mean(np.abs(scaled))),
                              "max_error": error, "max_scaled_error": scaled_error})
    for (key, name), (values, mask) in original.items():
        np.testing.assert_array_equal(getattr(pedals[key], name).values, values)
        np.testing.assert_array_equal(getattr(pedals[key], name).mask, mask)
    return midis, normalized, rows, beat_rows, pairs, scale_rows, events, errors


def legend(fig, ax):
    fig.legend(*ax.get_legend_handles_labels(), fontsize=10, ncols=2,
               loc="lower center", bbox_to_anchor=(.55, .08))


def plot_signal(sample, midi, feature, out, label):
    beats = np.asarray(sample.performance_beats)
    width = min(4, len(beats) - 1)
    raw = feature["depth"][sample.performance_key].raw
    on = feature["down_ratio"][sample.performance_key].raw
    changes_seq = feature["changes"][sample.performance_key].raw
    # A threshold-crossing beat explains all three quantities; select the nearest
    # one to the midpoint, rather than the most extreme. Retain zero-only cases.
    candidates = np.flatnonzero(raw.mask & (on.values > 0) & (on.values < 1) & (changes_seq.values > 0))
    rule = "nearest midpoint valid beat with partial on time and a threshold crossing"
    if not len(candidates):
        candidates = np.flatnonzero(raw.mask & (raw.values > 0))
        rule = "nearest midpoint valid beat with nonzero depth (no partial-on crossing beat)"
    if not len(candidates):
        candidates = np.flatnonzero(raw.mask)
        rule = "nearest midpoint valid beat (no nonzero pedal depth)"
    example = int(candidates[np.argmin(np.abs(candidates - (len(raw) - 1) / 2))])
    start = max(0, min(example - (width - 1) // 2, len(raw) - width))
    stop = start + width
    left, right = beats[start], beats[stop]
    times, values = np.array([p.time for p in midi.pedals]), np.array([p.value for p in midi.pedals])
    preceding = np.searchsorted(times, left, side="right") - 1
    initial = values[preceding] if preceding >= 0 else 0
    inside = (times > left) & (times < right)
    x, y = np.r_[left, times[inside], right], np.r_[initial, values[inside], values[times < right][-1] if (times < right).any() else 0]
    fig, ax = chart("페달 깊이와 on 시간은 서로 다른 값이다", f"{label} | {Path(sample.performance_key).stem} | 실제 CC64 신호, 예시 주변 {width}구간")
    ax.step(x, y, where="post", color=BLUE, lw=2.3, label="MIDI CC64 값")
    ax.axhline(64, color=ORANGE, ls="--", lw=1.5, label="on 기준: CC64 ≥ 64")
    ax.axvspan(beats[example], beats[example + 1], color="#eef0df", alpha=.8,
               label=f"계산 예시: 구간 {example + 1}")
    for i in range(start, stop + 1):
        ax.axvline(beats[i], color="#c4c8be", lw=.8)
        if i < stop:
            ax.text((beats[i] + beats[i + 1]) / 2, 121, f"구간 {i + 1}", ha="center", fontsize=10)
    ax.set(xlabel="실제 연주 시각(초)", ylabel="페달 CC64 값", xlim=(left, right), ylim=(-4, 131))
    legend(fig, ax)
    depth, down, changes = (feature[name][sample.performance_key].raw.values[example] for name in CHANNELS)
    note = f"음영 구간: 평균 깊이 {100 * depth:.1f}%, on 시간 {100 * down:.1f}%, 전환 {int(changes)}회. Changes는 CC64 이벤트 수가 아니라 64 경계의 on/off 전환 수다."
    save(fig, out / "01_cc64_signal_example.png", note, bottom=.23)
    return {"key": sample.performance_key, "selection_rule": rule, "display_intervals": [start + 1, stop],
            "example_display_interval": example + 1, "depth_percent": float(100 * depth),
            "down_ratio_percent": float(100 * down), "changes": int(changes)}


def plot_channels(samples, midis, normalized, rows, out, label, curve_beats):
    depth_rows = sorted([row for row in rows if row["channel"] == "depth"], key=lambda row: (row["raw_mean"], row["key"]))
    selected = [depth_rows[int(round(q * (len(depth_rows) - 1)))] for q in (.25, .5, .75)]
    keys = [row["key"] for row in selected]
    names = {key: f"{role} · {Path(key).stem}" for key, role in zip(keys, ROLES)}
    sample = next(s for s in samples if s.performance_key == keys[1])
    signal = plot_signal(sample, midis[keys[1]], normalized, out, label)
    reference = normalized["depth"][keys[1]]
    width = min(curve_beats, len(reference.raw))
    start = (len(reference.raw) - width) // 2
    section, x = slice(start, start + width), np.arange(start + 1, start + width + 1)
    for name in CHANNELS:
        folder = out / name
        folder.mkdir(exist_ok=True)
        ref = normalized[name][keys[1]]
        multiplier = MULTIPLIERS[name]
        for stage, filename in (("raw", "01_raw_and_common.png"), ("relative", "02_relative.png"), ("standardized", "03_standardized.png")):
            title = {"raw": f"같은 위치에서도 {LABELS[name]}가 다르다",
                     "relative": f"공통 패턴을 뺀 {LABELS[name]} 편차",
                     "standardized": f"{LABELS[name]} 편차를 작품 내 SD로 나눈다"}[stage]
            subtitle = f"{label} | 가운데 {width}구간 | 세 연주는 전체 평균 깊이 순위 25·50·75%"
            fig, ax = chart(title, subtitle)
            factor = 1. if stage == "standardized" else multiplier
            for key, color in zip(keys, COLORS):
                seq = getattr(normalized[name][key], stage)
                values = np.where(seq.mask, seq.values, np.nan)
                ax.plot(x, factor * values[section], color=color, lw=2, marker="o" if name == "changes" else None,
                        ms=3.5, label=names[key])
            if stage == "raw":
                ax.plot(x, multiplier * ref.common.values[section], color="#252d28", lw=2.3,
                        label=f"작품 공통 · {len(samples)}개 연주의 중앙값")
                if name != "changes":
                    ax.set_ylim(-3, 103)
                else:
                    ax.set_ylim(bottom=-.15)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                ax.axhline(0, color="#252d28", ls="--", lw=1.2, label="위치별 공통값과 같음")
                plotted = np.concatenate([getattr(normalized[name][key], stage).values[section] for key in keys])
                plotted = plotted[np.isfinite(plotted)]
                limit = max(.5 if stage == "standardized" or name == "changes" else 5., float(np.max(np.abs(plotted))) * factor * 1.15) if len(plotted) else 1.
                ax.set_ylim(-limit, limit)
            ylabel = ("표준화 편차(단위 없음)" if stage == "standardized" else
                      ("전환 횟수(회/beat)" if stage == "raw" else "공통 대비 전환 횟수 차이(회/beat)") if name == "changes" else
                      f"{LABELS[name]}(%)" if stage == "raw" else "공통 대비 차이(%p)")
            ax.set(xlabel="악보 beat 구간 번호", ylabel=ylabel, xlim=(x[0], x[-1]))
            if name != "changes" and stage != "standardized":
                suffix = "%" if stage == "raw" else "%p"
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _, suffix=suffix: f"{value:g}{suffix}"))
            legend(fig, ax)
            note = {"raw": "100%는 최대 CC64 깊이 또는 구간 전체 on. 0도 유효한 값이다." if name != "changes" else "0은 전환 없음. 연속 깊이 조절 이벤트는 64 경계를 넘지 않으면 횟수에 포함하지 않는다.",
                    "relative": "양수는 공통보다 더 깊게/더 오래/더 자주, 음수는 그 반대. 퍼센트 지표의 차이는 %p로 표시한다.",
                    "standardized": f"standardized = relative / {ref.scale.value:.6f}. 해당 작품·해당 하위 feature의 모든 유효 residual을 함께 fit; clipping 없음."}[stage]
            save(fig, folder / filename, note, bottom=.23 if stage != "raw" else .26)

        ordered = sorted([row for row in rows if row["channel"] == name], key=lambda row: (row["raw_mean"], row["key"]))
        fig, ax = chart(f"전체 {len(samples)}개 연주의 평균 {LABELS[name]} 비교", f"{label} | 전체 유효 beat 평균 | 공통 제거 전의 원본값", height=max(5.8, 2.5 + .36 * len(samples)))
        values = [multiplier * row["raw_mean"] for row in ordered]
        ax.barh([row["performance"] for row in ordered], values, height=.55,
                color=[COLORS[keys.index(row["key"])] if row["key"] in keys else "#b8bcb2" for row in ordered])
        for i, value in enumerate(values):
            ax.text(value + .025 * max(max(values), .01), i, f"{value:.1f}%" if name != "changes" else f"{value:.3f}회/beat", va="center", fontsize=11)
        ax.set(xlabel=f"유효 beat의 평균 {LABELS[name]} ({UNITS[name]})", xlim=(0, max(max(values), .01) * 1.45))
        save(fig, folder / "04_all_performances.png", "색 막대는 곡선에 나온 세 연주. 모든 beat를 동일 가중으로 평균하며 전체 연주 시간에 대한 가중 평균은 아니다.")
    return selected, signal, [start + 1, start + width]


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asap-root", type=Path, default=ai.parent.parent / "datasets/ASAP")
    parser.add_argument("--nasap-root", type=Path, default=ai.parent.parent / "datasets/nASAP")
    parser.add_argument("--work", default="Bach/Fugue/bwv_848")
    parser.add_argument("--curve-beats", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ai / "analysis/pedaling")
    args = parser.parse_args()
    if args.curve_beats < 2:
        parser.error("--curve-beats must be at least 2")
    root, nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    samples, tempo = collect(root, nasap_root, args.work)
    midis, normalized, rows, beat_rows, pairs, scales, events, errors = analyze(samples, tempo)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    label = "Bach · Fugue BWV 848" if args.work == "Bach/Fugue/bwv_848" else args.work
    selected, signal, curve_range = plot_channels(samples, midis, normalized, rows, args.out, label, args.curve_beats)
    for filename, contents in (("performance_summary.csv", rows), ("beat_features.csv", beat_rows),
                               ("pairwise_distances.csv", pairs), ("scales.csv", scales)):
        write_csv(args.out / filename, contents)
    # Even a piece with no pedal events has a reproducible empty event table.
    if events:
        write_csv(args.out / "cc64_events.csv", events)
    else:
        (args.out / "cc64_events.csv").write_text("key,event_index,time_seconds,value_cc64,instrument_idx,on\n", encoding="utf-8-sig")
    def revision(path):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    paths = [root / "asap_annotations.json", root / "metadata.csv", nasap_root / "metadata.csv",
             Path(__file__), Path(__file__).with_name("validate_tempo.py"),
             *sorted((ai / "src/features").glob("*.py")), *sorted((ai / "src/preprocessing").glob("*.py")),
             *[sample.performance_path for sample in samples]]
    reference = next(iter(normalized["depth"].values()))
    stats = {"work": args.work, "performances": len(samples), "intervals_per_performance": len(reference.raw),
             "curve_display_interval_range": curve_range,
             "curve_selection_rule": "nearest ranks to 25th, 50th and 75th percentiles of mean raw pedal depth; same three performers in every channel",
             "selected_performances": [{"role": role, "key": row["key"], "mean_depth_percent": row["raw_mean_display"]}
                                       for role, row in zip(ROLES, selected)],
             "cc64_signal_example": signal,
             "signal_selection_rule": "median-depth-rank performer; closest midpoint partial-on crossing beat, then nonzero-depth fallback, then valid zero fallback; four adjacent intervals; no search for maximum differences",
             "channels": {name: {"valid_raw_values": sum(row["raw_valid"] for row in rows if row["channel"] == name),
                                 "valid_relative_values": sum(row["relative_valid"] for row in rows if row["channel"] == name),
                                 "valid_standardized_values": sum(row["standardized_valid"] for row in rows if row["channel"] == name),
                                 "mean_raw_display_min": min(row["raw_mean_display"] for row in rows if row["channel"] == name),
                                 "mean_raw_display_max": max(row["raw_mean_display"] for row in rows if row["channel"] == name),
                                 "median_pair_mean_abs_difference": float(np.median([p["before_mean_abs_difference"] for p in pairs if p["channel"] == name])),
                                 "pair_count": sum(p["channel"] == name for p in pairs),
                                 "scale": next(row for row in scales if row["channel"] == name)} for name in CHANNELS},
             "common_support_min": int(reference.common_support.min()),
             "no_cc64_performances": [key for key, midi in midis.items() if not midi.pedals],
             "cc64_event_count": len(events), "minimum_support": 2,
             "tempo_status_context_counts": dict(Counter(i.status for tf in tempo.values() for i in tf.intervals)),
             "mask_policy": "Pedaling retains its positive-width mask; Tempo special/suspicious statuses are exported only as context, not applied as new pedal exclusions",
             "extraction_checks": errors, "asap_revision": revision(root), "nasap_revision": revision(nasap_root),
             "inputs_and_code_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest(),
             "clipping": None, "smoothing": None,
             "validation_scope": "CC64 numerical extraction, piece median subtraction and separate SD scaling, within-piece differences; not independent pedal sensor calibration or recommendation evaluation"}
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
