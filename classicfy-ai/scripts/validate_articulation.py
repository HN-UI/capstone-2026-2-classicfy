r"""(n)ASAP note 정렬로 계산한 Articulation 특징을 ASAP 전체 연주에서 검증하고 그림과 표로 정리한다.

classicfy-ai 디렉터리에서 실행한다. matplotlib과 pandas가 필요하다.

    PYTHONPATH=src python scripts/validate_articulation.py \
        --asap-root /path/to/ASAP --nasap-root /path/to/nASAP --out analysis/articulation

match 파일을 읽는 데 몇 분이 걸리므로 --cache를 주면 수집 결과를 저장해 두고 그림과 표만 다시 만들 수 있다.

곡을 골라 같은 곡 곡선(01, 02번 그림)만 그리려면 --works를 준다. 이때는 고른 작품의 연주만 읽고,
다른 그림과 표는 다시 만들지 않으며, 그림은 *_custom.png로 저장한다. 작품 이름은 --list-works로 찾는다.
추출 검증 그림(08번)에 쓸 연주는 --check-key로 바꿀 수 있다.
"""

import argparse
import bisect
import difflib
import json
import pickle
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import (  # noqa: E402
    build_tempo_map,
    extract_articulation,
    extract_note_articulation,
    summarize_articulation,
)
from features.articulation import ORNAMENT_ATTRIBUTES  # noqa: E402
from preprocessing import ASAPLoader, load_match, load_midi  # noqa: E402

TIME_TOLERANCE = 0.002  # match 시각과 MIDI 시각이 같다고 볼 오차(초)
MIN_WORK_SIZE = 5
PEDAL_ON_THRESHOLD = 64
LONG_NOTE_SECONDS = 0.5  # 기대 길이가 이보다 긴 음에서 페달 영향을 따로 본다
SMOOTH_BEATS = 8
# 악보 기호별 비교에 쓰는 속성. 앞에 있을수록 우선한다(스타카토+악센트는 스타카토로 센다).
MARKINGS = ["staccatissimo", "staccato", "detachedlegato", "tenuto", "fermata", "accent", "strongaccent"]

# 규칙 기반 점검 기준
TIMING_AGREEMENT_LIMIT = 0.999
DROPPED_TEMPO_LIMIT = 0.05
DELETION_LIMIT = 0.2
USED_NOTE_LIMIT = 0.95
BEAT_COVERAGE_LIMIT = 0.8
# 통계 기반 점검 기준
GLOBAL_Z_LIMIT = 3.5
WITHIN_WORK_Z_LIMIT = 4.0
OUTLIER_MIN_WORK_SIZE = 4

SUMMARY_COLUMNS = ["articulation_mean", "articulation_range"]
SUMMARY_LABELS = {
    "articulation_mean": "Articulation 평균 (log2)",
    "articulation_range": "Articulation 범위 (5–95%)",
}
MARKING_LABELS = {
    "staccatissimo": "staccatissimo",
    "staccato": "staccato",
    "accent": "accent",
    "tenuto": "tenuto",
    "none": "기호 없음",
    "fermata": "fermata",
    "detachedlegato": "detached legato",
    "strongaccent": "strong accent",
}

# 색은 validate_dynamics_pedaling.py와 같은 dataviz 기본 팔레트다. 범주 색은 앞 세 슬롯만 순서대로 쓴다.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CONTEXT = "#c3c2b7"
PEDAL_BAND = "#ececea"
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

_LOADER: ASAPLoader | None = None


def work_of(performance_key: str) -> str:
    return str(Path(performance_key).parent.as_posix())


# ---------------------------------------------------------------- 수집


def _init_worker(asap_root: str, nasap_root: str) -> None:
    global _LOADER
    _LOADER = ASAPLoader(asap_root, nasap_root)


def _timing_agreement(alignment, performance) -> float:
    """정렬된 연주 음 중 같은 음높이·onset·offset의 MIDI 음이 있는 비율."""
    by_pitch = defaultdict(list)
    for note in performance.notes:
        by_pitch[note.pitch].append((note.start, note.end))
    arrays = {pitch: np.array(sorted(times)) for pitch, times in by_pitch.items()}
    found = 0
    for _, note in alignment.matches:
        times = arrays.get(note.pitch)
        if times is None:
            continue
        index = np.searchsorted(times[:, 0], note.onset)
        for candidate in (index - 1, index):
            if 0 <= candidate < len(times) and (
                abs(times[candidate, 0] - note.onset) <= TIME_TOLERANCE
                and abs(times[candidate, 1] - note.offset) <= TIME_TOLERANCE
            ):
                found += 1
                break
    return found / len(alignment.matches) if alignment.matches else float("nan")


def _pedal_down_at(performance, times: np.ndarray) -> np.ndarray:
    """각 시각에 서스테인 페달이 on(CC64 >= 64)이었는지."""
    if not performance.pedals:
        return np.zeros(len(times), dtype=bool)
    pedal_times = np.array([pedal.time for pedal in performance.pedals])
    pedal_values = np.array([pedal.value for pedal in performance.pedals])
    index = np.searchsorted(pedal_times, times, side="right") - 1
    return (index >= 0) & (pedal_values[np.maximum(index, 0)] >= PEDAL_ON_THRESHOLD)


def _marking_of(attributes: tuple[str, ...]) -> str:
    return next((name for name in MARKINGS if name in attributes), "none")


def analyze_sample(performance_key: str) -> dict:
    assert _LOADER is not None
    sample = _LOADER.get_sample(performance_key)
    alignment = load_match(sample.note_alignment_path)
    performance = load_midi(sample.performance_path)
    beats = np.asarray(sample.performance_beats, dtype=float)

    positions, times = build_tempo_map(alignment)
    raw_positions = {
        round(score.onset_beats, 6) for score, _ in alignment.matches if not score.is_grace
    }
    notes = extract_note_articulation(alignment)
    feature = extract_articulation(alignment, beats)

    used = [alignment.matches[index] for index in notes.match_indices]
    expected = np.array([
        np.interp(score.offset_beats, positions, times) - np.interp(score.onset_beats, positions, times)
        for score, _ in used
    ])
    releases = np.array([performed.offset for _, performed in used])

    marking_values = defaultdict(list)
    for index, value in zip(notes.match_indices, notes.values):
        marking_values[_marking_of(alignment.matches[index][0].attributes)].append(float(value))

    return {
        "key": performance_key,
        "work": work_of(performance_key),
        "performer": Path(performance_key).stem,
        "composer": sample.composer,
        "title": sample.title,
        "robust": sample.robust_note_alignment,
        "beat_count": len(beats) - 1,
        "matches": len(alignment.matches),
        "deletions": len(alignment.deletions),
        "insertions": len(alignment.insertions),
        "timing_agreement": _timing_agreement(alignment, performance),
        "tempo_map_points": len(positions),
        "tempo_map_dropped": len(raw_positions) - len(positions),
        "used_notes": len(notes.values),
        **{f"excluded_{name}": count for name, count in notes.excluded_counts.items()},
        "beat_coverage": float(feature.sequence.mask.mean()),
        "note_values": notes.values.astype(np.float32),
        "note_expected_log2": np.log2(expected).astype(np.float32) if len(expected) else np.zeros(0, np.float32),
        "note_pedal_down": _pedal_down_at(performance, releases),
        "marking_values": {name: np.array(values, dtype=np.float32) for name, values in marking_values.items()},
        "beat_values": feature.sequence.values.astype(np.float32),
        "beat_mask": feature.sequence.mask,
        **summarize_articulation(feature),
    }


def target_keys(asap_root: Path, nasap_root: Path) -> list[str]:
    loader = ASAPLoader(asap_root, nasap_root)
    return [
        sample.performance_key
        for sample in loader.iter_samples(aligned_only=True)
        if sample.note_alignment_path is not None
    ]


def work_sizes(asap_root: Path, nasap_root: Path) -> dict[str, int]:
    """작품별 note 정렬이 있는 연주 수."""
    sizes: dict[str, int] = defaultdict(int)
    for key in target_keys(asap_root, nasap_root):
        sizes[work_of(key)] += 1
    return dict(sizes)


def check_works(requested: list[str], sizes: dict[str, int]) -> list[str]:
    """고른 작품이 그림을 그릴 수 있는지(연주 2개 이상) 확인한다."""
    works = list(dict.fromkeys(w.strip().replace("\\", "/").strip("/") for w in requested))
    problems = []
    for work in works:
        if sizes.get(work, 0) >= 2:
            continue
        message = f"'{work}': note 정렬된 연주가 2개 이상인 작품이 아니다."
        close = difflib.get_close_matches(work, [w for w, n in sizes.items() if n >= 2], n=3, cutoff=0.5)
        if close:
            message += " 비슷한 이름: " + ", ".join(close)
        problems.append(message)
    if problems:
        raise ValueError("\n".join(problems) + "\n--list-works로 선택할 수 있는 작품을 볼 수 있다.")
    return works


def load_or_collect(asap_root: Path, nasap_root: Path, cache: Path | None, workers: int,
                    works: list[str] | None = None) -> list[dict]:
    """캐시가 있으면 읽고, 없으면 수집한다. 일부 작품만 수집했을 때는 캐시에 저장하지 않는다."""
    if cache and cache.exists():
        print(f"loading cache: {cache}", flush=True)
        with cache.open("rb") as file:
            results = pickle.load(file)
        return [r for r in results if works is None or r["work"] in works]
    keys = [k for k in target_keys(asap_root, nasap_root) if works is None or work_of(k) in works]
    print(f"performances to analyze: {len(keys)}", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(asap_root), str(nasap_root))
    ) as pool:
        results = list(pool.map(analyze_sample, keys, chunksize=4))
    if cache and works is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as file:
            pickle.dump(results, file)
    return results


# ---------------------------------------------------------------- 표와 통계


def build_table(results: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame([
        {key: value for key, value in result.items() if not isinstance(value, (np.ndarray, dict))}
        for result in results
    ])
    excluded = [column for column in table.columns if column.startswith("excluded_")]
    table["deletion_ratio"] = table.deletions / (table.matches + table.deletions).replace(0, np.nan)
    table["tempo_map_dropped_ratio"] = table.tempo_map_dropped / (
        table.tempo_map_points + table.tempo_map_dropped
    ).replace(0, np.nan)
    table["used_note_ratio"] = table.used_notes / (table.used_notes + table[excluded].sum(axis=1)).replace(0, np.nan)
    return table


def robust_z(values: pd.Series) -> pd.Series:
    """중앙값과 MAD로 잰 z 점수. 극단값이 척도를 흔들지 않는다."""
    median = values.median()
    scale = 1.4826 * (values - median).abs().median()
    return (values - median) / scale if scale > 0 else values * 0.0


def add_flags(table: pd.DataFrame) -> pd.DataFrame:
    """정렬 품질 규칙과 요약값 분포 기준으로 점검 결과를 열로 추가한다."""
    table = table.copy()
    table["flag_no_matches"] = table.matches == 0
    table["flag_timing_mismatch"] = table.timing_agreement < TIMING_AGREEMENT_LIMIT
    table["flag_many_dropped_tempo_points"] = table.tempo_map_dropped_ratio > DROPPED_TEMPO_LIMIT
    table["flag_many_deletions"] = table.deletion_ratio > DELETION_LIMIT
    table["flag_many_excluded_notes"] = table.used_note_ratio < USED_NOTE_LIMIT
    table["flag_low_beat_coverage"] = table.beat_coverage < BEAT_COVERAGE_LIMIT

    global_hit = pd.Series(False, index=table.index)
    within_hit = pd.Series(False, index=table.index)
    eligible = (table.groupby("work")["key"].transform("size") >= OUTLIER_MIN_WORK_SIZE) & table.articulation_mean.notna()
    for column in SUMMARY_COLUMNS:
        z = robust_z(table[column])
        table[f"z_{column}"] = z
        global_hit |= z.abs() > GLOBAL_Z_LIMIT
        residual = table[column] - table.groupby("work")[column].transform("median")
        residual_z = robust_z(residual[eligible]).reindex(table.index)
        table[f"within_z_{column}"] = residual_z
        within_hit |= residual_z.abs() > WITHIN_WORK_Z_LIMIT
    table["flag_global_outlier"] = global_hit
    table["flag_within_work_outlier"] = within_hit

    flag_columns = [c for c in table.columns if c.startswith("flag_")]
    table["flags"] = table[flag_columns].apply(
        lambda row: ",".join(name[5:] for name in flag_columns if row[name]), axis=1
    )
    table["flagged"] = table["flags"] != ""
    table["statistical_outlier"] = table.flag_global_outlier | table.flag_within_work_outlier
    return table


def eta_squared(table: pd.DataFrame, column: str, permutations: int = 300) -> tuple[float, float]:
    """작품 라벨이 설명하는 분산 비율과, 라벨을 섞었을 때의 기대값."""
    values = table[column].to_numpy(dtype=float)
    labels = table["work"].to_numpy()
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]

    def ratio(assign: np.ndarray) -> float:
        frame = pd.DataFrame({"value": values, "work": assign})
        means = frame.groupby("work")["value"].transform("mean")
        return float(((means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum())

    rng = np.random.default_rng(0)
    shuffled = [ratio(rng.permutation(labels)) for _ in range(permutations)]
    return ratio(labels), float(np.mean(shuffled))


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """NaN을 건너뛰는 가운데 정렬 이동 평균."""
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().to_numpy()


def filled_curve(item: dict) -> np.ndarray:
    """결측 구간을 선형 보간으로 채운 beat 곡선."""
    values = np.where(item["beat_mask"], item["beat_values"], np.nan).astype(float)
    return pd.Series(values).interpolate(limit_direction="both").to_numpy()


def same_work_consistency(results: list[dict]) -> pd.DataFrame:
    """같은 작품 연주들의 beat 곡선 쌍별 상관 중앙값과 작품 안 평균의 표준편차."""
    by_work = defaultdict(list)
    for result in results:
        by_work[result["work"]].append(result)

    rows = []
    for work, items in by_work.items():
        lengths = {len(item["beat_values"]) for item in items}
        if len(items) < MIN_WORK_SIZE or len(lengths) != 1:
            continue
        for window in (1, SMOOTH_BEATS):
            curves = [filled_curve(item) for item in items]
            curves = [smooth(curve, window) if window > 1 else curve for curve in curves]
            matrix = np.corrcoef(np.vstack(curves))
            upper = matrix[np.triu_indices(len(curves), k=1)]
            rows.append({
                "work": work,
                "performances": len(items),
                "smooth_beats": window,
                "median_pair_correlation": float(np.nanmedian(upper)),
                "std_of_performance_means": float(np.std([item["articulation_mean"] for item in items])),
            })
    return pd.DataFrame(rows)


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = [5, 25, 50, 75, 95]
    return {f"p{point}": round(float(value), 3) for point, value in zip(points, np.percentile(values, points))}


def binned_medians(expected_log2: np.ndarray, values: np.ndarray, pedal: np.ndarray,
                   edges: np.ndarray, minimum: int = 200) -> pd.DataFrame:
    """기대 길이 구간별·페달 상태별 articulation 중앙값."""
    frame = pd.DataFrame({
        "bin": np.digitize(expected_log2, edges) - 1,
        "value": values,
        "pedal": pedal,
    })
    frame = frame[(frame.bin >= 0) & (frame.bin < len(edges) - 1)]
    grouped = frame.groupby(["bin", "pedal"])["value"].agg(["median", "size"]).reset_index()
    grouped = grouped[grouped["size"] >= minimum]
    grouped["center_seconds"] = 2 ** ((edges[grouped.bin] + edges[grouped.bin + 1]) / 2)
    return grouped


# ---------------------------------------------------------------- 독립 재계산 검증


def _reference_tempo_map(alignment) -> tuple[list[float], list[float]]:
    """추출기와 다른 방식(파이썬 루프, O(n²) 동적 계획법 LIS)으로 tempo map을 다시 만든다.

    순증가 최장 부분열은 길이가 같은 후보가 여럿일 수 있어서, 추출기(O(n log n))와 다른
    후보를 고를 수 있다. 그래서 길이는 같아야 하지만 점 집합은 다를 수 있다.
    """
    groups: dict[float, list[float]] = {}
    for score, performed in alignment.matches:
        if "grace" in score.attributes or score.offset_beats - score.onset_beats <= 0:
            continue
        groups.setdefault(round(score.onset_beats, 6), []).append(performed.onset)
    points = sorted((position, statistics.median(onsets)) for position, onsets in groups.items())

    length = [1] * len(points)
    previous = [-1] * len(points)
    for i in range(len(points)):
        for j in range(i):
            if points[j][1] < points[i][1] and length[j] + 1 > length[i]:
                length[i], previous[i] = length[j] + 1, j
    kept = []
    index = max(range(len(points)), key=lambda k: length[k]) if points else -1
    while index >= 0:
        kept.append(points[index])
        index = previous[index]
    kept.reverse()
    return [p for p, _ in kept], [t for _, t in kept]


def _reference_note_values(alignment, positions: list[float], times: list[float]) -> dict[int, tuple[float, float]]:
    """주어진 tempo map으로 음 단위 값을 파이썬 루프와 bisect 보간으로 다시 계산한다.

    돌려주는 값은 {match 인덱스: (연주 onset, log2 비율)}이다.
    """
    def to_time(position: float) -> float:
        right = bisect.bisect_left(positions, position)
        if right < len(positions) and positions[right] == position:
            return times[right]
        left = right - 1
        share = (position - positions[left]) / (positions[right] - positions[left])
        return times[left] + share * (times[right] - times[left])

    notes = {}
    for index, (score, performed) in enumerate(alignment.matches):
        if "grace" in score.attributes or score.offset_beats - score.onset_beats <= 0:
            continue
        if ORNAMENT_ATTRIBUTES.intersection(score.attributes):
            continue
        if len(positions) < 2 or score.onset_beats < positions[0] or score.offset_beats > positions[-1]:
            continue
        expected = to_time(score.offset_beats) - to_time(score.onset_beats)
        held = performed.offset - performed.onset
        if expected <= 0 or held <= 0:
            continue
        notes[index] = (performed.onset, float(np.log2(held / expected)))
    return notes


def independent_check(asap_root: Path, nasap_root: Path, table: pd.DataFrame, count: int = 6) -> list[dict]:
    """무작위 연주 몇 개를 다른 방식으로 다시 계산해 비교한다.

    1. tempo map: 다른 LIS 알고리즘으로 만든 결과와 길이가 같은지, 순증가하는지.
    2. 음·beat 값: 추출기와 같은 tempo map을 주고 파이썬 루프로 다시 계산해 오차를 잰다.
       LIS 동점 선택과 분리해서 보간·제외 규칙·구간 배정·중앙값을 검증한다.
    3. LIS 동점 민감도: 다른 LIS 후보로 계산했을 때 값이 바뀌는 음의 비율.
    """
    loader = ASAPLoader(asap_root, nasap_root)
    rng = np.random.default_rng(1)
    candidates = table[table.matches > 0]
    picks = candidates.iloc[rng.choice(len(candidates), size=count, replace=False)]
    report = []
    for key in picks.key:
        sample = loader.get_sample(key)
        alignment = load_match(sample.note_alignment_path)
        beats = list(sample.performance_beats)

        positions, times = build_tempo_map(alignment)
        ref_positions, ref_times = _reference_tempo_map(alignment)
        notes = extract_note_articulation(alignment)
        feature = extract_articulation(alignment, beats)

        same_map_ref = _reference_note_values(alignment, list(positions), list(times))
        extracted = dict(zip(notes.match_indices.tolist(), notes.values.tolist()))
        same_indices = set(same_map_ref) == set(extracted)
        note_error = max((abs(extracted[i] - same_map_ref[i][1]) for i in extracted if i in same_map_ref),
                         default=0.0)

        beat_error = 0.0
        mask_mismatch = 0
        ordered = sorted(same_map_ref.values())
        onsets = [onset for onset, _ in ordered]
        for i in range(len(beats) - 1):
            low = bisect.bisect_left(onsets, beats[i])
            high = bisect.bisect_left(onsets, beats[i + 1])
            inside = [value for _, value in ordered[low:high]]
            has_value = bool(inside) and beats[i + 1] > beats[i]
            if has_value != bool(feature.sequence.mask[i]):
                mask_mismatch += 1
                continue
            if has_value:
                beat_error = max(beat_error, abs(statistics.median(inside) - feature.sequence.values[i]))

        other_map = _reference_note_values(alignment, ref_positions, ref_times)
        shared = [i for i in extracted if i in other_map]
        changed = [abs(extracted[i] - other_map[i][1]) for i in shared]
        changed_share = float(np.mean(np.array(changed) > 1e-6)) if changed else 0.0

        report.append({
            "key": key,
            "tempo_map_points": len(positions),
            "reference_tempo_map_points": len(ref_positions),
            "tempo_map_same_length": len(positions) == len(ref_positions),
            "tempo_map_same_points": len(positions) == len(ref_positions) and bool(
                np.allclose(positions, ref_positions) and np.allclose(times, ref_times)),
            "tempo_map_strictly_increasing": bool(np.all(np.diff(times) > 0)),
            "notes": len(extracted),
            "same_note_set": same_indices,
            "note_max_abs_error": float(note_error),
            "beat_max_abs_error": float(beat_error),
            "beat_mask_mismatch": mask_mismatch,
            "lis_tie_changed_note_share": round(changed_share, 5),
            "lis_tie_max_abs_change": round(float(max(changed, default=0.0)), 3),
        })
        print(f"  check {key}: map length {'same' if len(positions) == len(ref_positions) else 'DIFF'}, "
              f"note {note_error:.1e}, beat {beat_error:.1e}, mask mismatch {mask_mismatch}, "
              f"LIS tie changes {changed_share:.2%}", flush=True)
    return report


# ---------------------------------------------------------------- 그림


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_SECONDARY,
    })


def blue_ramp() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("blue_ramp", SEQUENTIAL)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def zero_line(ax, vertical: bool = False) -> None:
    """log2 비율 0(악보 음가만큼 누름) 기준선."""
    (ax.axvline if vertical else ax.axhline)(0, color=AXIS, lw=1, zorder=1)


def pick_works(table: pd.DataFrame, count: int = 3) -> list[str]:
    """연주가 많은 작품을 작곡가가 겹치지 않게 고른다."""
    sizes = table.groupby("work").agg(n=("key", "size"), composer=("composer", "first"))
    chosen, used = [], set()
    for work, row in sizes.sort_values("n", ascending=False).iterrows():
        if row.composer in used:
            continue
        chosen.append(work)
        used.add(row.composer)
        if len(chosen) == count:
            break
    return chosen


def work_members(results: list[dict], work: str) -> tuple[list[dict], np.ndarray]:
    """작품의 연주들과 결측을 보간한 (연주, beat) 행렬."""
    members = [r for r in results if r["work"] == work and r["matches"] > 0]
    lengths = {len(m["beat_values"]) for m in members}
    assert len(lengths) == 1, f"{work}: beat 수가 다르다 {lengths}"
    return members, np.stack([filled_curve(m) for m in members])


def plot_same_work_curves(results, works, path: Path) -> None:
    fig, axes = plt.subplots(1, len(works), figsize=(5.6 * len(works), 4.2), squeeze=False)
    for ax, work in zip(axes[0], works):
        members, matrix = work_members(results, work)
        means = np.array([m["articulation_mean"] for m in members])
        high, low = int(np.nanargmax(means)), int(np.nanargmin(means))
        curves = np.stack([smooth(line, SMOOTH_BEATS) for line in matrix])
        zero_line(ax)
        for line in curves:
            ax.plot(line, color=CONTEXT, lw=0.8, alpha=0.9)
        ax.plot(np.nanmedian(curves, axis=0), color=BLUE, lw=1.8)
        ax.plot(curves[high], color=ORANGE, lw=1.5)
        ax.plot(curves[low], color=AQUA, lw=1.5)
        ax.set_title(f"{members[0]['composer']} · {members[0]['title']}  (연주 {len(members)}개)")
        ax.set_xlabel(f"beat 번호 ({SMOOTH_BEATS}-beat 이동 평균)")
        ax.set_ylabel("Articulation (log2 누른 시간 / 기대 길이)")
        ax.legend(
            handles=[
                Line2D([], [], color=CONTEXT, lw=1, label="개별 연주"),
                Line2D([], [], color=BLUE, lw=1.8, label="작품 중앙값"),
                Line2D([], [], color=ORANGE, lw=1.5, label=f"평균 최고(길게): {members[high]['performer']}"),
                Line2D([], [], color=AQUA, lw=1.5, label=f"평균 최저(짧게): {members[low]['performer']}"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2,
        )
    fig.suptitle("같은 곡 여러 연주의 Articulation 곡선 (0 = 악보 음가만큼 누름, 아래로 갈수록 짧게 끊음)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_same_work_heatmaps(results, works, path: Path) -> None:
    fig, axes = plt.subplots(1, len(works), figsize=(5.6 * len(works), 4.4), squeeze=False)
    cmap = blue_ramp()
    for ax, work in zip(axes[0], works):
        members, matrix = work_members(results, work)
        order = np.argsort([m["articulation_mean"] for m in members])
        shown = np.stack([smooth(line, 4) for line in matrix[order]])
        low, high = np.nanpercentile(shown, [2, 98])
        image = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        ax.grid(False)
        ax.set_yticks([])
        ax.set_ylabel("연주 (평균 오름차순, 위 = 짧게)")
        ax.set_xlabel("beat 번호")
        ax.set_title(f"{members[0]['composer']} · {members[0]['title']}")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Articulation (2–98% 범위)")
    fig.suptitle("연주 × beat 히트맵 (색이 진할수록 길게 누름)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def histogram(ax, values: np.ndarray, bins, xlabel: str, title: str) -> None:
    ax.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
            color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    median = float(np.median(values))
    ax.axvline(median, color=INK_SECONDARY, lw=1)
    ax.annotate(f"중앙값 {median:.2f}", (median, ax.get_ylim()[1]), xytext=(6, -12),
                textcoords="offset points", fontsize=8, color=INK_SECONDARY,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_xlabel(xlabel)
    ax.set_ylabel("비율 (%)")
    ax.set_title(title)


def plot_distributions(results, table, path: Path) -> None:
    notes = np.concatenate([r["note_values"] for r in results])
    beats = np.concatenate([r["beat_values"][r["beat_mask"]] for r in results])
    valid = table.dropna(subset=SUMMARY_COLUMNS)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    bins = np.linspace(-5, 3, 81)
    histogram(axes[0, 0], np.clip(notes, -5, 3), bins, "음 단위 Articulation (±범위 밖은 끝에 모음)",
              f"음 단위 (음 {len(notes):,}개)")
    histogram(axes[0, 1], np.clip(beats, -5, 3), bins, "beat 구간 중앙값", f"beat 단위 (구간 {len(beats):,}개)")
    histogram(axes[1, 0], valid.articulation_mean, 30, SUMMARY_LABELS["articulation_mean"],
              f"연주별 평균 (연주 {len(valid):,}개)")
    histogram(axes[1, 1], valid.articulation_range, 30, SUMMARY_LABELS["articulation_range"], "연주별 범위")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        zero_line(ax, vertical=True)
    fig.suptitle("Articulation 분포 (0 = 악보 음가만큼, -1 = 절반, -2 = 1/4)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def box_stats(values: np.ndarray, label: str) -> dict:
    """5–95% 수염의 상자 통계. 수백만 개 값을 그대로 넘기지 않으려고 직접 계산한다."""
    p5, q1, median, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {"label": label, "whislo": p5, "q1": q1, "med": median, "q3": q3, "whishi": p95, "fliers": []}


def draw_boxes(ax, stats: list[dict]) -> None:
    box = ax.bxp(stats, orientation="horizontal", patch_artist=True, widths=0.55, showfliers=False)
    for patch in box["boxes"]:
        patch.set(facecolor=SEQUENTIAL[1], edgecolor=BLUE, linewidth=1)
    for name in ("whiskers", "caps"):
        for line in box[name]:
            line.set(color=BLUE, linewidth=1)
    for line in box["medians"]:
        line.set(color=BLUE, linewidth=2)
    ax.grid(axis="y", visible=False)
    zero_line(ax, vertical=True)


def plot_markings_and_pedal(results, path: Path) -> None:
    marking_notes = defaultdict(list)
    for result in results:
        for name, values in result["marking_values"].items():
            marking_notes[name].append(values)
    marking_notes = {name: np.concatenate(chunks) for name, chunks in marking_notes.items()}
    order = sorted(marking_notes, key=lambda name: np.median(marking_notes[name]))

    values = np.concatenate([r["note_values"] for r in results])
    expected = np.concatenate([r["note_expected_log2"] for r in results])
    pedal = np.concatenate([r["note_pedal_down"] for r in results])
    long_notes = expected >= np.log2(LONG_NOTE_SECONDS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})

    ax = axes[0]
    draw_boxes(ax, [box_stats(marking_notes[name], f"{MARKING_LABELS[name]} ({len(marking_notes[name]):,})")
                    for name in order])
    ax.set_xlabel("음 단위 Articulation")
    ax.set_title("악보 기호별 (상자 = 25–75%, 수염 = 5–95%)")

    ax = axes[1]
    groups = [
        (~long_notes & ~pedal, "짧은 음 · 페달 off"),
        (~long_notes & pedal, "짧은 음 · 페달 on"),
        (long_notes & ~pedal, "긴 음 · 페달 off"),
        (long_notes & pedal, "긴 음 · 페달 on"),
    ]
    draw_boxes(ax, [box_stats(values[mask], f"{label} ({mask.sum():,})") for mask, label in groups])
    ax.set_xlabel("음 단위 Articulation")
    ax.set_title(f"건반을 뗄 때 페달 상태 (긴 음 = 기대 길이 {LONG_NOTE_SECONDS}초 이상)")

    ax = axes[2]
    edges = np.linspace(-5, 2, 29)
    binned = binned_medians(expected, values, pedal, edges)
    for is_down, color, label in ((False, AQUA, "뗄 때 페달 off"), (True, ORANGE, "뗄 때 페달 on")):
        part = binned[binned.pedal == is_down]
        ax.plot(part.center_seconds, part["median"], color=color, lw=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, label=label)
    ax.set_xscale("log", base=2)
    # 기본 log 눈금은 mathtext의 유니코드 마이너스를 써서 한글 글꼴에서 깨지므로 초를 그대로 쓴다.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.axvline(LONG_NOTE_SECONDS, color=AXIS, lw=1, ls="--")
    zero_line(ax)
    ax.set_xlabel("기대 길이 (초, log 눈금)")
    ax.set_ylabel("음 단위 Articulation 중앙값")
    ax.set_title("기대 길이별 중앙값 (점선 = 0.5초, 음 200개 이상인 구간만)")
    ax.legend(loc="lower left")

    fig.suptitle("Articulation이 재는 것: 악보 기호와 순서가 맞고, 긴 음에서는 페달이 섞인다",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_by_composer(table, path: Path, minimum: int = 20) -> None:
    valid = table.dropna(subset=SUMMARY_COLUMNS)
    counts = valid.composer.value_counts()
    composers = [c for c in counts.index if counts[c] >= minimum]
    order = (valid[valid.composer.isin(composers)].groupby("composer")["articulation_mean"].median()
             .sort_values().index.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, column in zip(axes, SUMMARY_COLUMNS):
        groups = [valid.loc[valid.composer == c, column].to_numpy() for c in order]
        box = ax.boxplot(groups, orientation="horizontal", whis=(5, 95), showfliers=True, patch_artist=True,
                         widths=0.55, flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                                                  "markeredgecolor": SURFACE, "alpha": 0.7})
        for patch in box["boxes"]:
            patch.set(facecolor=SEQUENTIAL[1], edgecolor=BLUE, linewidth=1)
        for name in ("whiskers", "caps"):
            for line in box[name]:
                line.set(color=BLUE, linewidth=1)
        for line in box["medians"]:
            line.set(color=BLUE, linewidth=2)
        ax.set_yticks(range(1, len(order) + 1), [f"{c} ({counts[c]})" for c in order])
        ax.set_title(SUMMARY_LABELS[column])
        ax.grid(axis="y", visible=False)
    zero_line(axes[0], vertical=True)
    fig.suptitle(f"작곡가별 연주 요약 (연주 {minimum}개 이상, 상자 = 25–75%, 수염 = 5–95%)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_eta(eta: dict[str, tuple[float, float]], path: Path) -> None:
    columns = sorted(eta, key=lambda c: eta[c][0])
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    positions = np.arange(len(columns))
    ax.barh(positions + 0.19, [eta[c][0] for c in columns], height=0.34, color=BLUE)
    ax.barh(positions - 0.19, [eta[c][1] for c in columns], height=0.34, color=CONTEXT)
    for position, column in zip(positions, columns):
        ax.text(eta[column][0] + 0.01, position + 0.19, f"{eta[column][0]:.2f}", va="center",
                fontsize=8, color=INK_SECONDARY)
        ax.text(eta[column][1] + 0.01, position - 0.19, f"{eta[column][1]:.2f}", va="center",
                fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(positions, [SUMMARY_LABELS[c] for c in columns])
    ax.set_xlim(0, 1)
    ax.set_xlabel("작품 정체성이 설명하는 분산 비율 (η²)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="실제 작품 라벨"),
                       Line2D([], [], color=CONTEXT, lw=6, label="작품 라벨을 섞은 기대값")],
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncols=2)
    ax.set_title("연주 요약 특징 중 작품이 좌우하는 부분")
    fig.tight_layout()
    save(fig, path)


def plot_outliers(table, path: Path) -> None:
    valid = table.dropna(subset=SUMMARY_COLUMNS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    normal, flagged = valid[~valid.statistical_outlier], valid[valid.statistical_outlier]
    ax.scatter(normal.articulation_mean, normal.articulation_range, s=16, color=CONTEXT, edgecolor=SURFACE,
               linewidth=0.6)
    ax.scatter(flagged.articulation_mean, flagged.articulation_range, s=26, color=ORANGE, edgecolor=SURFACE,
               linewidth=0.8)
    middle = valid.articulation_mean.median()
    strength = pd.concat([valid[f"{prefix}{c}"].abs() for prefix in ("z_", "within_z_") for c in SUMMARY_COLUMNS],
                         axis=1).max(axis=1)
    extreme = strength[valid.statistical_outlier].nlargest(4).index
    for index in extreme:
        row = valid.loc[index]
        side = -1 if row.articulation_mean > middle else 1
        ax.annotate(f"{row.composer} {row.performer}", (row.articulation_mean, row.articulation_range),
                    xytext=(8 * side, 6), textcoords="offset points", fontsize=7, color=INK_SECONDARY,
                    ha="left" if side > 0 else "right")
    ax.set_xlabel(SUMMARY_LABELS["articulation_mean"])
    ax.set_ylabel(SUMMARY_LABELS["articulation_range"])
    ax.set_title("요약 특징 (주황 = 전체 분포나 같은 작품 안에서 크게 벗어남)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CONTEXT, label=f"기준 이내 ({len(normal)})"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label=f"통계적 이상치 ({len(flagged)})")],
              loc="lower right", bbox_to_anchor=(1, 1.06), ncols=2)

    ax = axes[1]
    robust = table.robust == True  # noqa: E712
    for mask, color, label in ((robust, CONTEXT, "robust 정렬"), (~robust, BLUE, "non-robust 정렬")):
        part = table[mask]
        ax.scatter(part.tempo_map_dropped_ratio * 100, part.deletion_ratio * 100, s=16, color=color,
                   edgecolor=SURFACE, linewidth=0.6, label=f"{label} ({mask.sum()})")
    ax.axvline(DROPPED_TEMPO_LIMIT * 100, color=AXIS, lw=1, ls="--")
    ax.axhline(DELETION_LIMIT * 100, color=AXIS, lw=1, ls="--")
    ax.set_xlabel("tempo map에서 버린 위치 (%)")
    ax.set_ylabel("연주에서 빠진 악보 음 (%)")
    ax.set_title("정렬 품질 (점선 = 규칙 점검 기준)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.06), ncols=2)

    fig.suptitle("연주별 이상치 점검", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def pick_check_key(results: list[dict]) -> str:
    """추출 검증 그림용 연주: 볼 거리가 모두 있는 robust 연주.

    스타카토 음, 페달을 밟은 채 뗀 음과 뗀 음, tempo map에서 버린 위치가 모두 있어야 한다.
    그중 스타카토 음이 가장 많은 연주를 고른다.
    """
    candidates = [
        r for r in results
        if r["robust"] is True and r["tempo_map_dropped"] >= 3
        and 0.2 <= float(np.mean(r["note_pedal_down"])) <= 0.8
    ]
    best = max(candidates, key=lambda r: len(r["marking_values"].get("staccato", ())))
    return str(best["key"])


def plot_extraction_check(asap_root: Path, nasap_root: Path, key: str, path: Path, span: float = 6.0) -> None:
    """tempo map, 음마다의 누른 시간 vs 기대 길이, beat 구간 중앙값을 원본 위에 겹쳐 본다."""
    sample = ASAPLoader(asap_root, nasap_root).get_sample(key)
    alignment = load_match(sample.note_alignment_path)
    performance = load_midi(sample.performance_path)
    beats = np.asarray(sample.performance_beats, dtype=float)
    positions, times = build_tempo_map(alignment)
    notes = extract_note_articulation(alignment)
    feature = extract_articulation(alignment, beats)

    # 스타카토 음이 가장 많이 들어가는 구간을 보여 준다. 없으면 곡의 1/3 지점.
    staccato_onsets = np.sort([
        performed.onset for score, performed in (alignment.matches[i] for i in notes.match_indices)
        if _marking_of(score.attributes) in ("staccato", "staccatissimo")
    ])
    if len(staccato_onsets):
        counts = np.searchsorted(staccato_onsets, staccato_onsets + span * 0.8) - np.arange(len(staccato_onsets))
        start = float(staccato_onsets[int(np.argmax(counts))]) - span * 0.1
    else:
        start = float(np.interp(positions[0] + (positions[-1] - positions[0]) / 3, positions, times))
    end = start + span

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1.6, 1])
    ax_map, ax_roll, ax_value = (fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[2]))
    ax_value.sharex(ax_roll)

    # 1. tempo map: 곡 전체 평균 빠르기 직선을 빼서 rubato와 버린 위치(주황)가 보이게 한다.
    raw = defaultdict(list)
    for score, performed in alignment.matches:
        if not score.is_grace:
            raw[round(score.onset_beats, 6)].append(performed.onset)
    raw_positions = np.array(sorted(raw))
    raw_times = np.array([np.median(raw[p]) for p in raw_positions])
    dropped = ~np.isin(raw_positions, positions)
    slope, intercept = np.polyfit(positions, times, 1)
    ax_map.plot(positions, times - (slope * positions + intercept), color=BLUE, lw=1.5,
                label=f"tempo map ({len(positions):,}개 위치)")
    ax_map.scatter(raw_positions[dropped], raw_times[dropped] - (slope * raw_positions[dropped] + intercept),
                   s=26, color=ORANGE, edgecolor=SURFACE, linewidth=0.8, zorder=3,
                   label=f"역행해서 버린 위치 ({dropped.sum()}개)")
    window = (times >= start) & (times < end)
    if window.any():
        ax_map.axvspan(positions[window][0], positions[window][-1], color=PEDAL_BAND, zorder=0,
                       label="아래 두 그림의 구간")
    ax_map.set_xlabel("악보 위치 (beat)")
    ax_map.set_ylabel("연주 시각 - 평균 빠르기 직선 (초)")
    ax_map.set_title("1. tempo map: 악보 위치마다 연주 onset 중앙값, 순증가 최장 부분열만 남김 (위 = 늦어짐)")
    ax_map.legend(loc="upper left")

    # 2. 피아노 롤: 회색 테 = 기대 길이, 채운 막대 = 실제로 누른 시간.
    shown = [(i, onset, value) for i, onset, value in zip(notes.match_indices, notes.onsets, notes.values)
             if start <= onset < end]
    for match_index, onset, value in shown:
        score, performed = alignment.matches[match_index]
        expected = np.interp(score.offset_beats, positions, times) - np.interp(score.onset_beats, positions, times)
        staccato = _marking_of(score.attributes) in ("staccato", "staccatissimo")
        ax_roll.add_patch(plt.Rectangle((onset, performed.pitch - 0.4), expected, 0.8, facecolor="none",
                                        edgecolor=MUTED, linewidth=0.9, zorder=2))
        ax_roll.add_patch(plt.Rectangle((onset, performed.pitch - 0.3), performed.duration, 0.6,
                                        facecolor=ORANGE if staccato else BLUE, edgecolor="none", zorder=3))
    pitches = [alignment.matches[i][1].pitch for i, _, _ in shown] or [60]
    pedal_times = np.array([p.time for p in performance.pedals])
    pedal_values = np.array([p.value for p in performance.pedals])
    for index in range(len(pedal_times)):
        if pedal_values[index] >= PEDAL_ON_THRESHOLD:
            until = pedal_times[index + 1] if index + 1 < len(pedal_times) else end
            if until > start and pedal_times[index] < end:
                ax_roll.axvspan(max(pedal_times[index], start), min(until, end), color=PEDAL_BAND, zorder=0)
    ax_roll.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax_roll.set_ylabel("MIDI 음높이")
    ax_roll.set_title("2. 음마다 실제로 누른 시간(채움) vs tempo map으로 환산한 악보 음가(회색 테)")
    ax_roll.legend(handles=[Patch(facecolor=BLUE, label="누른 시간"),
                            Patch(facecolor=ORANGE, label="누른 시간 (staccato 기호)"),
                            Patch(facecolor="none", edgecolor=MUTED, label="기대 길이"),
                            Patch(facecolor=PEDAL_BAND, label="서스테인 페달 on")],
                   loc="lower right", bbox_to_anchor=(1, 1.08), ncols=4)
    ax_roll.tick_params(labelbottom=False)

    # 3. 음 단위 값과 beat 구간 중앙값.
    ax_value.scatter([o for _, o, _ in shown], [v for _, _, v in shown], s=14, color=CONTEXT,
                     edgecolor=SURFACE, linewidth=0.5, zorder=2)
    window_beats = np.flatnonzero((beats[:-1] < end) & (beats[1:] > start))
    for i in window_beats:
        if feature.sequence.mask[i]:
            ax_value.hlines(feature.sequence.values[i], beats[i], beats[i + 1], color=BLUE, lw=2, zorder=3)
    zero_line(ax_value)
    inside_beats = beats[(beats >= start) & (beats < end)]
    ax_value.plot(inside_beats, np.full(len(inside_beats), ax_value.get_ylim()[0]), "|", color=MUTED,
                  markersize=8, clip_on=False)
    ax_value.set_xlim(start, end)
    ax_value.set_xlabel("연주 시각 (초)  ·  아래 눈금 = beat")
    ax_value.set_ylabel("log2(누른 시간 / 기대 길이)")
    ax_value.set_title("3. 음 단위 값과 beat 구간 중앙값")
    ax_value.legend(handles=[Line2D([], [], marker="o", ls="", color=CONTEXT, label="음 단위 값"),
                             Line2D([], [], color=BLUE, lw=2, label="beat 구간 중앙값 (특징)")],
                    loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2)

    fig.suptitle(f"추출 검증: {key}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------- 실행


def build_stats(results, table, consistency, eta, check, figure_works, check_key) -> dict:
    all_notes = np.concatenate([result["note_values"] for result in results])
    marking_notes = defaultdict(list)
    for result in results:
        for name, values in result["marking_values"].items():
            marking_notes[name].append(values)
    excluded_columns = [column for column in table.columns if column.startswith("excluded_")]
    total_candidates = table["used_notes"].sum() + table[excluded_columns].to_numpy().sum()

    all_expected = np.concatenate([result["note_expected_log2"] for result in results])
    all_pedal = np.concatenate([result["note_pedal_down"] for result in results])
    long_notes = all_expected >= np.log2(LONG_NOTE_SECONDS)
    robust = table["robust"] == True  # noqa: E712
    flag_columns = [c for c in table.columns if c.startswith("flag_")]

    return {
        "performances": len(table),
        "works": int(table["work"].nunique()),
        "robust_performances": int(robust.sum()),
        "matched_notes": int(table["matches"].sum()),
        "deleted_notes": int(table["deletions"].sum()),
        "inserted_notes": int(table["insertions"].sum()),
        "timing_agreement_min": round(float(table["timing_agreement"].min()), 5),
        "timing_agreement_mean": round(float(table["timing_agreement"].mean()), 5),
        "tempo_map_dropped_ratio": round(
            float(table["tempo_map_dropped"].sum() / (table["tempo_map_points"] + table["tempo_map_dropped"]).sum()), 5
        ),
        "used_note_ratio": round(float(table["used_notes"].sum() / total_candidates), 4),
        "excluded_ratio": {
            column.removeprefix("excluded_"): round(float(table[column].sum() / total_candidates), 4)
            for column in excluded_columns
        },
        "beat_coverage_median": round(float(table.beat_coverage.median()), 4),
        "beat_coverage_min": round(float(table.beat_coverage.min()), 4),
        "note_quantiles": quantiles(all_notes),
        "note_share_below_-3": round(float((all_notes < -3).mean()), 4),
        "note_share_above_2": round(float((all_notes > 2).mean()), 4),
        "corr_note_value_vs_log2_expected_seconds": round(float(np.corrcoef(all_notes, all_expected)[0, 1]), 3),
        "pedal_at_release": {
            f"{'long' if is_long else 'short'}_{'down' if is_down else 'up'}": {
                "notes": int((mask := (long_notes == is_long) & (all_pedal == is_down)).sum()),
                "median": round(float(np.median(all_notes[mask])), 3),
            }
            for is_long in (False, True)
            for is_down in (False, True)
        },
        "marking_medians": {
            name: {"notes": int(sum(len(values) for values in chunks)),
                   "median": round(float(np.median(np.concatenate(chunks))), 3)}
            for name, chunks in sorted(marking_notes.items())
        },
        "robust_vs_not": {
            str(flag): {
                "performances": int(len(group)),
                "tempo_map_dropped_ratio": round(float(group["tempo_map_dropped"].sum() / (group["tempo_map_points"] + group["tempo_map_dropped"]).sum()), 5),
                "deletion_ratio": round(float(group["deletions"].sum() / (group["matches"] + group["deletions"]).sum()), 4),
                "articulation_mean_median": round(float(group["articulation_mean"].median()), 3),
            }
            for flag, group in table.groupby(table["robust"].astype(str))
        },
        "summary_quantiles": {
            column: quantiles(table[column].dropna().to_numpy()) for column in SUMMARY_COLUMNS
        },
        "composer_articulation_mean_median": {
            composer: round(float(value), 3)
            for composer, value in table.groupby("composer")["articulation_mean"].median().sort_values().items()
        },
        "eta_squared": {
            column: {"actual": round(values[0], 3), "shuffled": round(values[1], 3)}
            for column, values in eta.items()
        },
        "same_work": {
            str(window): {
                "works": int(len(group)),
                "median_pair_correlation": round(float(group["median_pair_correlation"].median()), 3),
                "std_of_performance_means_median": round(float(group["std_of_performance_means"].median()), 3),
            }
            for window, group in consistency.groupby("smooth_beats")
        },
        "overall_std_of_performance_means": round(float(table["articulation_mean"].std()), 3),
        "flag_counts": {c[5:]: int(table[c].sum()) for c in flag_columns},
        "flagged_performances": int(table.flagged.sum()),
        "statistical_outliers": int(table.statistical_outlier.sum()),
        "statistical_outliers_by_composer": table[table.statistical_outlier].composer.value_counts().to_dict(),
        "independent_check": check,
        "figure_works": figure_works,
        "extraction_check_key": check_key,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asap-root", type=Path, required=True)
    parser.add_argument("--nasap-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "analysis/articulation")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--works", nargs="+", metavar="WORK",
                        help="같은 곡 곡선을 그릴 작품 (예: Chopin/Etudes_op_10/1). 주면 01, 02번 그림만 *_custom.png로 만든다")
    parser.add_argument("--list-works", action="store_true", help="선택할 수 있는 작품과 연주 수를 출력하고 끝낸다")
    parser.add_argument("--check-key", default=None, help="08번 추출 검증 그림에 쓸 연주 키 (기본: 자동 선택)")
    args = parser.parse_args()

    if args.list_works:
        for work, size in sorted(work_sizes(args.asap_root, args.nasap_root).items(),
                                 key=lambda item: (-item[1], item[0])):
            if size >= 2:
                print(f"{size:3d}  {work}")
        return

    works = None
    if args.works:
        try:
            works = check_works(args.works, work_sizes(args.asap_root, args.nasap_root))
        except ValueError as error:
            parser.error(str(error))

    results = load_or_collect(args.asap_root, args.nasap_root, args.cache, args.workers, works)
    args.out.mkdir(parents=True, exist_ok=True)

    if works is not None:
        apply_style()
        print("figures")
        plot_same_work_curves(results, works, args.out / "01_same_work_curves_custom.png")
        plot_same_work_heatmaps(results, works, args.out / "02_same_work_heatmaps_custom.png")
        return

    table = add_flags(build_table(results))
    table.to_csv(args.out / "performance_summary.csv", index=False, encoding="utf-8-sig")
    table[table.flagged].sort_values("flags").to_csv(args.out / "outliers.csv", index=False, encoding="utf-8-sig")

    consistency = same_work_consistency(results)
    consistency.to_csv(args.out / "same_work_consistency.csv", index=False, encoding="utf-8-sig")
    eta = {column: eta_squared(table, column) for column in SUMMARY_COLUMNS}

    print("independent re-computation check")
    check = independent_check(args.asap_root, args.nasap_root, table)

    figure_works = pick_works(table)
    check_key = args.check_key or pick_check_key(results)
    apply_style()
    print("figures")
    plot_same_work_curves(results, figure_works, args.out / "01_same_work_curves.png")
    plot_same_work_heatmaps(results, figure_works, args.out / "02_same_work_heatmaps.png")
    plot_distributions(results, table, args.out / "03_distributions.png")
    plot_markings_and_pedal(results, args.out / "04_markings_and_pedal.png")
    plot_by_composer(table, args.out / "05_by_composer.png")
    plot_eta(eta, args.out / "06_work_vs_performance_variance.png")
    plot_outliers(table, args.out / "07_outliers.png")
    plot_extraction_check(args.asap_root, args.nasap_root, check_key, args.out / "08_extraction_check.png")

    stats = build_stats(results, table, consistency, eta, check, figure_works, check_key)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats["flag_counts"], ensure_ascii=False), "flagged:", stats["flagged_performances"])


if __name__ == "__main__":
    main()
