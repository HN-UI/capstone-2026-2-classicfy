"""Draw readable, one-chart-per-file Dynamics analyses of an ASAP piece.

Run from the repository root:
    .venv/bin/python classicfy-ai/scripts/validate_dynamics.py

Extraction, piece median subtraction and MAD scaling remain unchanged. Curves
show raw/relative values in MIDI velocity units (multiply by 127 for display).
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from validate_tempo import BLUE, GRAY, ORANGE, chart, plt, save, setup_style, write_csv
from features import PieceFeatureInput, extract_dynamics, separate_piece_feature, standardize_piece_feature
from preprocessing import ASAPLoader, load_midi

COLORS = (BLUE, GRAY, ORANGE)
ROLES = ("평균 세기 낮은 편", "평균 세기 중간", "평균 세기 높은 편")


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    valid = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[valid] - expected[valid]))) if valid.any() else 0.


def collect_samples(asap_root, nasap_root, work):
    if Path(work).is_absolute() or ".." in Path(work).parts:
        raise ValueError("--work must be a piece folder relative to ASAP")
    loader = ASAPLoader(asap_root, nasap_root)
    score = (loader.root / work / "midi_score.mid").resolve()
    samples = [s for s in loader.iter_samples(True) if s.score_path == score and (
        s.note_alignment_path is None or s.note_alignment_path.parent.name == s.performance_path.parent.name
    )]
    if len(samples) < 3:
        raise ValueError("At least three aligned performances with the same repeat structure are required")
    return samples


def reference_dynamics(midi, beats):
    """Direct onset filtering per beat, independent of the extractor's binning."""
    onsets = np.array([note.start for note in midi.notes])
    velocities = np.array([note.velocity for note in midi.notes])
    counts = np.zeros(len(beats) - 1, dtype=int)
    values = np.full(len(counts), np.nan)
    for i, (left, right) in enumerate(zip(beats[:-1], beats[1:])):
        inside = (onsets >= left) & (onsets < right)
        counts[i] = inside.sum()
        if counts[i] and right > left:
            values[i] = np.mean(velocities[inside]) / 127
    return counts, values


def value_range(sequence):
    values = sequence.values[sequence.mask]
    return float(np.diff(np.percentile(values, [5, 95]))[0]) if len(values) else float("nan")


def analyze(samples):
    midis, dynamics, inputs = {}, {}, []
    note_rows, beat_rows, rows, pairs = [], [], [], []
    errors = {name: 0. for name in ("raw", "common", "relative", "scale", "standardized", "pair_difference", "scaled_pair_difference")}
    for sample in samples:
        key = sample.performance_key
        performance_beats = np.asarray(sample.performance_beats, dtype=float)
        midi = load_midi(sample.performance_path)
        raw = extract_dynamics(midi, sample.performance_beats)
        counts, direct = reference_dynamics(midi, sample.performance_beats)
        np.testing.assert_array_equal(raw.onset_counts, counts)
        np.testing.assert_array_equal(raw.sequence.mask, (counts > 0) & (np.diff(sample.performance_beats) > 0))
        errors["raw"] = max(errors["raw"], check(raw.sequence.values, direct))
        midis[key], dynamics[key] = midi, raw
        inputs.append(PieceFeatureInput.from_asap_sample(sample, raw.sequence))
        for index, note in enumerate(midi.notes):
            # Determine membership directly, without reusing assign_windows.
            positions = np.flatnonzero((performance_beats[:-1] <= note.start) & (note.start < performance_beats[1:]))
            beat = int(positions[0]) if len(positions) else -1
            note_rows.append({"key": key, "note_index": index, "pitch": note.pitch, "start_seconds": note.start,
                              "end_seconds": note.end, "velocity": note.velocity, "instrument_idx": note.instrument_idx,
                              "beat_index": beat, "display_interval": beat + 1 if beat >= 0 else None})
    separated = separate_piece_feature(inputs)
    normalized = standardize_piece_feature(list(separated.values()), feature_name="dynamics")
    reference = next(iter(normalized.values()))
    matrix = np.array([np.where(item.sequence.mask & np.isfinite(item.sequence.values) & (np.diff(item.score_beats) > 0),
                                item.sequence.values, np.nan) for item in inputs])
    support = np.isfinite(matrix).sum(axis=0)
    common = np.full(matrix.shape[1], np.nan)
    for i in np.flatnonzero(support >= 2):
        common[i] = np.median(matrix[np.isfinite(matrix[:, i]), i])
    residual = matrix - common
    pooled = residual[np.isfinite(residual)]
    if not len(pooled):
        raise ValueError("Piece has no valid Dynamics residuals with support >= 2")
    candidates = {"mad": float(1.4826 * np.median(np.abs(pooled - np.median(pooled)))),
                  "iqr": float(np.diff(np.percentile(pooled, [25, 75]))[0] / 1.3489795003921634),
                  "std": float(np.std(pooled, ddof=0))}
    method = next((name for name in ("mad", "iqr", "std") if candidates[name] > 1e-12), "unit")
    expected_scale = candidates.get(method, 1.)
    assert reference.scale.method == method
    errors["scale"] = check(np.array([reference.scale.value]), np.array([expected_scale]))
    np.testing.assert_allclose([reference.scale.mad, reference.scale.iqr, reference.scale.std], list(candidates.values()), atol=1e-12)
    for sample in samples:
        key, nf = sample.performance_key, normalized[sample.performance_key]
        raw = dynamics[key]
        np.testing.assert_array_equal(nf.raw.values, raw.sequence.values)
        np.testing.assert_array_equal(nf.raw.mask, raw.sequence.mask)
        np.testing.assert_array_equal(nf.common_support, support)
        np.testing.assert_array_equal(nf.common.mask, support >= 2)
        mask = raw.sequence.mask & np.isfinite(raw.sequence.values) & (np.diff(sample.score_beats) > 0) & (support >= 2)
        np.testing.assert_array_equal(nf.relative.mask, mask)
        np.testing.assert_array_equal(nf.standardized.mask, mask)
        expected = np.where(mask, raw.sequence.values - common, np.nan)
        errors["common"] = max(errors["common"], check(nf.common.values, common))
        errors["relative"] = max(errors["relative"], check(nf.relative.values, expected))
        errors["standardized"] = max(errors["standardized"], check(nf.standardized.values, expected / expected_scale))
        rows.append({"key": key, "performance": Path(key).stem, "intervals": len(nf.raw),
                     "raw_valid": int(nf.raw.mask.sum()), "relative_valid": int(mask.sum()),
                     "note_count": len(midis[key].notes), "onsets_in_beat_grid": int(raw.onset_counts.sum()),
                     "empty_beat_count": int((raw.onset_counts == 0).sum()), "scale": nf.scale.value,
                     **{f"{stage}_mean": float(np.mean(getattr(nf, stage).values[getattr(nf, stage).mask]))
                        for stage in ("raw", "common", "relative", "standardized")},
                     "raw_mean_velocity": float(127 * np.mean(nf.raw.values[nf.raw.mask])),
                     "relative_mean_velocity": float(127 * np.mean(nf.relative.values[mask])),
                     "raw_range_velocity": 127 * value_range(nf.raw), "relative_range_velocity": 127 * value_range(nf.relative),
                     "relative_mean_abs_velocity": float(127 * np.mean(np.abs(nf.relative.values[mask])))})
        for i in range(len(nf.raw)):
            beat_rows.append({"key": key, "beat_index": i, "display_interval": i + 1,
                              "score_interval_seconds": float(np.diff(sample.score_beats)[i]),
                              "performance_start_seconds": sample.performance_beats[i],
                              "performance_end_seconds": sample.performance_beats[i + 1],
                              "score_beat_type": sample.score_beat_types[i], "performance_beat_type": sample.performance_beat_types[i],
                              "onset_count": int(raw.onset_counts[i]),
                              **{stage: float(getattr(nf, stage).values[i]) for stage in ("raw", "common", "relative", "standardized")},
                              **{f"{stage}_mask": bool(getattr(nf, stage).mask[i]) for stage in ("raw", "common", "relative", "standardized")},
                              "raw_velocity": float(127 * nf.raw.values[i]), "common_velocity": float(127 * nf.common.values[i]),
                              "relative_velocity": float(127 * nf.relative.values[i]), "common_support": int(nf.common_support[i]), "scale": nf.scale.value})
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
                          "before_mean_abs_velocity_difference": float(127 * np.mean(np.abs(before))),
                          "after_mean_abs_velocity_difference": float(127 * np.mean(np.abs(after))),
                          "standardized_mean_abs_difference": float(np.mean(np.abs(scaled))),
                          "max_error": error, "max_scaled_error": scaled_error})
    return midis, dynamics, normalized, rows, beat_rows, note_rows, pairs, errors


def legend(fig, ax):
    fig.legend(*ax.get_legend_handles_labels(), fontsize=10, ncols=2,
               loc="lower center", bbox_to_anchor=(.55, .08))


def plot_examples(samples, midis, dynamics, normalized, rows, pairs, out, label, curve_beats):
    ordered = sorted(rows, key=lambda row: (row["raw_mean"], row["key"]))
    selected = [ordered[int(round(q * (len(ordered) - 1)))] for q in (.25, .5, .75)]
    keys = [row["key"] for row in selected]
    names = {key: f"{role} · {Path(key).stem}" for key, role in zip(keys, ROLES)}
    ref = normalized[keys[1]]
    sample = next(s for s in samples if s.performance_key == keys[1])
    positions = np.flatnonzero(ref.raw.mask)
    beat = int(positions[np.argmin(np.abs(positions - (len(ref.raw) - 1) / 2))])
    left, right = sample.performance_beats[beat:beat + 2]
    notes = [note for note in midis[keys[1]].notes if left <= note.start < right]
    fig, ax = chart("Dynamics는 구간에서 시작한 음의 평균 세기다",
                    f"{label} | {Path(keys[1]).stem} | 악보 구간 {beat + 1} | 시작 음 {len(notes)}개")
    x = np.arange(len(notes))
    ax.bar(x, [note.velocity for note in notes], color=BLUE, width=.6, label="개별 음의 velocity")
    mean = float(127 * ref.raw.values[beat])
    ax.axhline(mean, color=ORANGE, ls="--", lw=2, label=f"beat 평균 velocity {mean:.2f}")
    for i, note in enumerate(notes):
        ax.text(i, note.velocity + 2, str(note.velocity), ha="center", fontsize=12)
    ax.set(xticks=x, xticklabels=[f"음 {i + 1}\npitch {n.pitch}" for i, n in enumerate(notes)],
           xlabel="이 beat 구간 안에서 시작한 음", ylabel="MIDI velocity (0~127)", ylim=(0, 132))
    ax.legend(loc="upper right", fontsize=10)
    save(fig, out / "01_note_velocity_to_beat.png", f"mean velocity {mean:.2f} / 127 = raw Dynamics {ref.raw.values[beat]:.6f}. 이전 구간에서 시작해 계속 울리는 음은 이 구간 평균에 포함하지 않는다.")

    width = min(curve_beats, len(ref.raw))
    start = (len(ref.raw) - width) // 2
    section, x = slice(start, start + width), np.arange(start + 1, start + width + 1)
    for stage, filename, title in (
        ("raw", "02_raw_and_common.png", "같은 악보 위치에서도 강약 곡선이 다르다"),
        ("relative", "03_relative.png", "작품 공통 강약을 빼도 개인별 편차가 남는다"),
        ("standardized", "04_standardized.png", "개인별 강약 편차를 작품 내 MAD로 나눈다"),
    ):
        fig, ax = chart(title, f"{label} | 가운데 {width}구간 확대 | 전체 평균 세기 순위 25·50·75% 연주")
        factor = 1 if stage == "standardized" else 127
        for key, color in zip(keys, COLORS):
            seq = getattr(normalized[key], stage)
            ax.plot(x, factor * np.where(seq.mask, seq.values, np.nan)[section], color=color, lw=2, label=names[key])
        if stage == "raw":
            ax.plot(x, 127 * ref.common.values[section], color="#252d28", lw=2.3,
                    label=f"작품 공통 · {len(rows)}개 연주의 중앙값")
            ax.set_ylim(0, 127)
        else:
            ax.axhline(0, color="#252d28", ls="--", lw=1.2, label="그 위치의 공통값과 같음")
            values = np.concatenate([getattr(normalized[key], stage).values[section] for key in keys])
            valid = values[np.isfinite(values)]
            limit = max(1., float(np.max(np.abs(valid))) * factor * 1.15) if len(valid) else 1.
            ax.set_ylim(-limit, limit)
        ax.set(xlabel="악보 beat 구간 번호", ylabel={"raw": "beat 평균 velocity (0~127)",
               "relative": "공통 대비 평균 velocity 차이", "standardized": "표준화 편차(단위 없음)"}[stage], xlim=(x[0], x[-1]))
        legend(fig, ax)
        note = {"raw": "9개 모두로 공통 패턴을 계산했다. 평균 세기 순위가 높은 연주도 특정 위치에서는 더 약하게 연주할 수 있다.",
                "relative": "양수는 공통보다 강한 velocity, 음수는 약한 velocity다. 127 × (raw − common)으로 표시하며 평활화 없음.",
                "standardized": f"standardized = relative / {ref.scale.value:.6f}. +1은 공통보다 약 {127 * ref.scale.value:.2f} velocity 높음. clipping 없음."}[stage]
        save(fig, out / filename, note, bottom=.26 if stage == "raw" else .23)

    fig, ax = chart(f"전체 {len(rows)}개 연주의 대표 강약 편차를 비교한다", f"{label} | 모든 유효 beat 평균 | 원본 평균과 공통 제거 후 평균을 함께 표시", height=max(5.8, 2.5 + .36 * len(rows)))
    y, values = np.arange(len(ordered)), [row["relative_mean_velocity"] for row in ordered]
    ax.hlines(y, 0, values, color="#d6d9cf", lw=2)
    ax.scatter(values, y, color=[COLORS[keys.index(row["key"])] if row["key"] in keys else "#b8bcb2" for row in ordered], s=65)
    ax.axvline(0, color="#252d28", ls="--", lw=1.2)
    span = max(max(values) - min(values), 1.)
    for i, row in enumerate(ordered):
        ax.text(max(values) + .06 * span, i, f"원본 {row['raw_mean_velocity']:.1f} → 편차 {row['relative_mean_velocity']:+.1f}", va="center", fontsize=11)
    ax.set(yticks=y, yticklabels=[row["performance"] for row in ordered],
           xlabel="전체 유효 beat의 평균 강약 편차(velocity 차이)", xlim=(min(min(values), 0) - .1 * span, max(values) + 1.35 * span))
    save(fig, out / "05_all_performance_means.png", "곡선에 나온 세 연주를 색으로 표시했다. 점은 평균 편차이며, 공통 패턴은 위치마다 다르다. 전체 재센터링 없음.")

    fig, ax = chart("공통 강약을 빼도 연주 사이의 차이는 유지된다", f"{label} | 모든 {len(pairs)}개 연주 쌍 | 같은 유효 위치의 평균 절대 차이")
    before = np.array([pair["before_mean_abs_velocity_difference"] for pair in pairs])
    after = np.array([pair["after_mean_abs_velocity_difference"] for pair in pairs])
    bound = max(1., float(before.max()) * 1.15)
    ax.plot([0, bound], [0, bound], color="#343b35", ls="--", lw=1.5, label="제거 전후 차이가 같은 위치")
    ax.scatter(before, after, s=52, color=BLUE, edgecolor="white", linewidth=.8, label="연주 쌍 하나")
    ax.set(xlabel="공통 제거 전의 평균 velocity 차이", ylabel="공통 제거 후의 평균 velocity 차이", xlim=(0, bound), ylim=(0, bound))
    ax.legend(loc="upper left", fontsize=10)
    save(fig, out / "06_pairwise_difference_preserved.png", "같은 beat에서 같은 공통값을 빼므로 (raw A − common) − (raw B − common) = raw A − raw B다.")

    fig, ax = chart("평균 세기와 곡 안에서 변하는 폭은 따로 보아야 한다", f"{label} | 전체 유효 beat | 변화 폭 = 95백분위 − 5백분위", height=max(5.8, 2.5 + .36 * len(rows)))
    raw_range, relative_range = [row["raw_range_velocity"] for row in ordered], [row["relative_range_velocity"] for row in ordered]
    for i, a, b in zip(y, raw_range, relative_range):
        ax.plot([a, b], [i, i], color="#c3c6bd", lw=2)
        ax.text(max(raw_range + relative_range) + 1, i, f"{a:.1f} → {b:.1f}", va="center", fontsize=11)
    ax.scatter(raw_range, y, color=GRAY, s=65, label="원본 강약 변화 폭")
    ax.scatter(relative_range, y, color=ORANGE, s=65, label="공통 제거 후 편차의 변화 폭")
    ax.set(yticks=y, yticklabels=[row["performance"] for row in ordered], xlabel="5~95백분위 폭(velocity 차이)",
           xlim=(0, max(raw_range + relative_range) * 1.4))
    legend(fig, ax)
    save(fig, out / "07_all_performance_ranges.png", "원본은 연주 내 강약 범위, 상대값은 공통과 다른 편차의 범위다. 공통 제거 후 폭이 반드시 줄지는 않는다.", bottom=.23)
    return selected, {"key": keys[1], "display_interval": beat + 1, "note_velocities": [n.velocity for n in notes],
                      "mean_velocity": mean, "raw_dynamics": float(ref.raw.values[beat])}, [start + 1, start + width]


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asap-root", type=Path, default=ai.parent.parent / "datasets/ASAP")
    parser.add_argument("--nasap-root", type=Path, default=ai.parent.parent / "datasets/nASAP")
    parser.add_argument("--work", default="Bach/Fugue/bwv_848")
    parser.add_argument("--curve-beats", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ai / "analysis/dynamics")
    args = parser.parse_args()
    if args.curve_beats < 2:
        parser.error("--curve-beats must be at least 2")
    root, nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    samples = collect_samples(root, nasap_root, args.work)
    midis, dynamics, normalized, rows, beat_rows, notes, pairs, errors = analyze(samples)
    if any(not nf.relative.mask.any() for nf in normalized.values()):
        raise ValueError("Each plotted performer must have valid Dynamics residuals")
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    label = "Bach · Fugue BWV 848" if args.work == "Bach/Fugue/bwv_848" else args.work
    selected, example, curve_range = plot_examples(samples, midis, dynamics, normalized, rows, pairs, args.out, label, args.curve_beats)
    for filename, contents in (("performance_summary.csv", rows), ("beat_features.csv", beat_rows),
                               ("note_events.csv", notes), ("pairwise_distances.csv", pairs)):
        write_csv(args.out / filename, contents)
    ref = next(iter(normalized.values()))
    scale = asdict(ref.scale)
    write_csv(args.out / "scale.csv", [scale])
    def revision(path):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    paths = [root / "asap_annotations.json", root / "metadata.csv", nasap_root / "metadata.csv",
             Path(__file__), Path(__file__).with_name("validate_tempo.py"),
             *sorted((ai / "src/features").glob("*.py")), *sorted((ai / "src/preprocessing").glob("*.py")),
             *[sample.performance_path for sample in samples]]
    pooled = np.concatenate([nf.relative.values[nf.relative.mask] for nf in normalized.values()])
    stats = {"work": args.work, "performances": len(rows), "intervals_per_performance": len(ref.raw),
             "valid_raw_values": sum(row["raw_valid"] for row in rows), "valid_relative_values": sum(row["relative_valid"] for row in rows),
             "empty_beat_count": sum(row["empty_beat_count"] for row in rows), "note_event_count": len(notes),
             "common_support_min": int(ref.common_support.min()), "common_support_max": int(ref.common_support.max()),
             "minimum_support": 2, "scale": scale, "scale_in_velocity_units": 127 * ref.scale.value,
             "residual_zero_ratio": float(np.mean(pooled == 0)),
             "standardized_max_abs": float(np.max(np.abs(pooled / ref.scale.value))),
             "mean_raw_velocity_min": min(row["raw_mean_velocity"] for row in rows),
             "mean_raw_velocity_max": max(row["raw_mean_velocity"] for row in rows),
             "mean_relative_velocity_min": min(row["relative_mean_velocity"] for row in rows),
             "mean_relative_velocity_max": max(row["relative_mean_velocity"] for row in rows),
             "pair_count": len(pairs),
             "median_pair_mean_abs_velocity_difference": float(np.median([row["before_mean_abs_velocity_difference"] for row in pairs])),
             "curve_selection_rule": "nearest ranks to 25th, 50th and 75th percentiles of mean raw Dynamics",
             "selected_performances": [{"role": role, "key": row["key"], "raw_mean_velocity": row["raw_mean_velocity"]}
                                       for role, row in zip(ROLES, selected)],
             "curve_display_interval_range": curve_range, "note_velocity_example": example,
             "example_rule": "nearest midpoint valid beat of the median-mean-Dynamics-rank performer; no maximum-difference search",
             "extraction_checks": errors, "asap_revision": revision(root), "nasap_revision": revision(nasap_root),
             "inputs_and_code_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest(),
             "clipping": None, "smoothing": None,
             "mask_policy": "onset_count > 0 and positive performance interval width; common/relative additionally require finite values, positive score width, and support >= 2; no new Tempo masks",
             "validation_scope": "Numerical velocity-to-beat extraction and piece normalization, within-piece differences; not calibrated acoustic loudness, voice-level dynamics or recommendation evaluation"}
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
