r"""ASAP 전체 연주에 Dynamics·Pedaling 특징을 적용하고 검증 결과를 그림과 표로 정리한다.

classicfy-ai 디렉터리에서 실행한다. matplotlib과 pandas가 필요하다.

    PYTHONPATH=src python scripts/validate_dynamics_pedaling.py \
        --asap-root /path/to/ASAP --out reports/dynamics_pedaling

MIDI를 읽는 데 시간이 걸리므로 --cache를 주면 수집 결과를 저장해 두고 그림만 다시 그릴 수 있다.

곡을 골라 같은 곡 곡선(01, 02번 그림)만 그리려면 --works를 준다. 이때는 고른 작품의 연주만 읽고,
다른 그림과 표는 다시 만들지 않으며, 그림은 *_custom.png로 저장한다. 작품 이름은 --list-works로 찾는다.

    python scripts/validate_dynamics_pedaling.py --asap-root /path/to/ASAP --list-works
    python scripts/validate_dynamics_pedaling.py --asap-root /path/to/ASAP \
        --works Chopin/Etudes_op_10/1 Liszt/Gran_Etudes_de_Paganini/2_La_campanella --out reports/custom
"""

import argparse
import difflib
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import ASAPLoader, load_midi  # noqa: E402
from preprocessing.features import (  # noqa: E402
    extract_dynamics,
    extract_pedaling,
    summarize_dynamics,
    summarize_pedaling,
)

SHORT_INTERVAL_RATIO = 0.05  # 중앙값 beat 간격 대비 이보다 짧으면 비정상적으로 짧은 beat로 본다
LONG_INTERVAL_RATIO = 20.0  # 중앙값 beat 간격 대비 이보다 길면 비정상적으로 긴 beat로 본다
LOW_VELOCITY = 10
EMPTY_BEAT_LIMIT = 0.3
BINARY_PEDAL_MAX_VALUES = 5
GLOBAL_Z_LIMIT = 3.5
WITHIN_WORK_Z_LIMIT = 4.0
MIN_WORK_SIZE = 4
SMOOTH_BEATS = 8

SUMMARY_COLUMNS = [
    "dynamics_mean",
    "dynamics_range",
    "pedal_depth_mean",
    "pedal_usage",
    "pedal_change_rate",
]
SUMMARY_LABELS = {
    "dynamics_mean": "Dynamics 평균",
    "dynamics_range": "Dynamics 범위 (5–95%)",
    "pedal_depth_mean": "페달 깊이 평균",
    "pedal_usage": "페달 on 비율",
    "pedal_change_rate": "페달 전환 횟수 / beat",
}

# 색은 dataviz 기본 팔레트를 그대로 쓴다. 범주 색은 앞 세 슬롯만 쓰고 순서를 바꾸지 않는다.
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
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

_loader: ASAPLoader | None = None


def work_of(performance_key: str) -> str:
    """연주 키에서 작품(폴더) 이름을 구한다. 예: Bach/Fugue/bwv_846"""
    return str(Path(performance_key).parent).replace("\\", "/")


# ---------------------------------------------------------------- 수집


def _init_worker(asap_root: str) -> None:
    global _loader
    _loader = ASAPLoader(asap_root)


def analyze_sample(performance_key: str) -> dict:
    """연주 하나에서 특징 시퀀스, 곡 요약, 원본 MIDI·beat 통계를 모은다."""
    sample = _loader.get_sample(performance_key)
    performance = load_midi(sample.performance_path)
    beats = np.asarray(sample.performance_beats)

    dynamics = extract_dynamics(performance, beats)
    pedaling = extract_pedaling(performance, beats)

    velocities = np.array([note.velocity for note in performance.notes])
    onsets = np.array([note.start for note in performance.notes])
    pedal_values = np.array([pedal.value for pedal in performance.pedals])
    inside = (onsets >= beats[0]) & (onsets < beats[-1])

    stats = {
        "key": performance_key,
        "work": work_of(performance_key),
        "performer": Path(performance_key).stem,
        "composer": sample.composer,
        "title": sample.title,
        "beat_count": len(beats) - 1,
        "note_count": len(velocities),
        "velocity_min": int(velocities.min()),
        "velocity_max": int(velocities.max()),
        "velocity_unique": len(set(velocities.tolist())),
        "velocity_low_ratio": float((velocities <= LOW_VELOCITY).mean()),
        "pedal_event_count": len(pedal_values),
        "pedal_unique": len(set(pedal_values.tolist())),
        "onset_inside_ratio": float(inside.mean()),
        "empty_beat_ratio": float((~dynamics.mask).mean()),
        "duration": float(beats[-1] - beats[0]),
        **summarize_dynamics(dynamics),
        **summarize_pedaling(pedaling),
    }
    sequences = {
        "beats": beats,
        "dynamics": dynamics.values.astype(np.float32),
        "dynamics_mask": dynamics.mask,
        "pedal_depth": pedaling.depth.astype(np.float32),
        "pedal_down": pedaling.down_ratio.astype(np.float32),
        "pedal_changes": pedaling.changes.astype(np.int16),
    }
    return {"stats": stats, "sequences": sequences}


def aligned_keys(asap_root: Path) -> list[str]:
    return [s.performance_key for s in ASAPLoader(asap_root).iter_samples(aligned_only=True)]


def work_sizes(asap_root: Path) -> dict[str, int]:
    """작품별 정렬된 연주 수."""
    sizes: dict[str, int] = {}
    for key in aligned_keys(asap_root):
        sizes[work_of(key)] = sizes.get(work_of(key), 0) + 1
    return sizes


def check_works(requested: list[str], sizes: dict[str, int]) -> list[str]:
    """고른 작품이 그림을 그릴 수 있는지(정렬된 연주 2개 이상) 확인한다."""
    works = list(dict.fromkeys(w.strip().replace("\\", "/").strip("/") for w in requested))
    problems = []
    for work in works:
        if sizes.get(work, 0) >= 2:
            continue
        message = f"'{work}': 정렬된 연주가 2개 이상인 작품이 아니다."
        close = difflib.get_close_matches(work, [w for w, n in sizes.items() if n >= 2], n=3, cutoff=0.5)
        if close:
            message += " 비슷한 이름: " + ", ".join(close)
        problems.append(message)
    if problems:
        raise ValueError("\n".join(problems) + "\n--list-works로 선택할 수 있는 작품을 볼 수 있다.")
    return works


def collect(asap_root: Path, workers: int, keys: list[str]) -> list[dict]:
    """주어진 연주를 병렬로 분석한다."""
    print(f"performances to analyze: {len(keys)}", flush=True)
    results = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(asap_root),)) as pool:
        for done, result in enumerate(pool.map(analyze_sample, keys, chunksize=4), start=1):
            results.append(result)
            if done % 100 == 0:
                print(f"  {done}/{len(keys)}", flush=True)
    return results


def load_or_collect(asap_root: Path, cache: Path | None, workers: int,
                    works: list[str] | None = None) -> list[dict]:
    """캐시가 있으면 읽고, 없으면 수집한다. 일부 작품만 수집했을 때는 캐시에 저장하지 않는다."""
    if cache is not None and cache.is_file():
        print(f"loading cache: {cache}", flush=True)
        with cache.open("rb") as file:
            results = pickle.load(file)
        return [r for r in results if works is None or r["stats"]["work"] in works]
    keys = [k for k in aligned_keys(asap_root) if works is None or work_of(k) in works]
    results = collect(asap_root, workers, keys)
    if cache is not None and works is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as file:
            pickle.dump(results, file)
    return results


# ---------------------------------------------------------------- 이상치 점검


def build_table(results: list[dict]) -> pd.DataFrame:
    """연주별 통계 표. beat 간격 통계는 beat 시각에서 바로 계산한다."""
    table = pd.DataFrame([r["stats"] for r in results])
    intervals = [np.diff(r["sequences"]["beats"]) for r in results]
    median = np.array([np.median(x) for x in intervals])
    table["interval_min"] = [x.min() for x in intervals]
    table["interval_median"] = median
    table["interval_max"] = [x.max() for x in intervals]
    table["short_interval_count"] = [int((x < SHORT_INTERVAL_RATIO * m).sum()) for x, m in zip(intervals, median)]
    table["long_interval_count"] = [int((x > LONG_INTERVAL_RATIO * m).sum()) for x, m in zip(intervals, median)]
    return table


def robust_z(values: pd.Series) -> pd.Series:
    """중앙값과 MAD로 잰 z 점수. 극단값이 척도를 흔들지 않는다."""
    median = values.median()
    scale = 1.4826 * (values - median).abs().median()
    return (values - median) / scale if scale > 0 else values * 0.0


def add_flags(table: pd.DataFrame) -> pd.DataFrame:
    """규칙 기반 점검과 통계 기반 점검 결과를 열로 추가한다."""
    table = table.copy()
    table["flag_no_pedal_events"] = table.pedal_event_count == 0
    table["flag_binary_pedal"] = (table.pedal_event_count > 0) & (
        table.pedal_unique <= BINARY_PEDAL_MAX_VALUES
    )
    table["flag_short_beat_interval"] = table.short_interval_count > 0
    table["flag_long_beat_interval"] = table.long_interval_count > 0
    table["flag_many_empty_beats"] = table.empty_beat_ratio > EMPTY_BEAT_LIMIT

    global_hit = pd.Series(False, index=table.index)
    within_hit = pd.Series(False, index=table.index)
    work_size = table.groupby("work")["key"].transform("size")
    eligible = work_size >= MIN_WORK_SIZE
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
    """작품 정체성이 설명하는 분산 비율과, 작품 라벨을 섞었을 때의 기대값."""
    grouped = table[table.groupby("work")["key"].transform("size") >= 2]
    values = grouped[column].to_numpy()
    labels = pd.factorize(grouped["work"])[0]

    def ratio(assign: np.ndarray) -> float:
        counts = np.bincount(assign)
        means = np.bincount(assign, weights=values) / counts
        between = (counts * (means - values.mean()) ** 2).sum()
        return float(between / ((values - values.mean()) ** 2).sum())

    rng = np.random.default_rng(0)
    null = np.mean([ratio(rng.permutation(labels)) for _ in range(permutations)])
    return ratio(labels), float(null)


def same_work_consistency(results: list[dict], minimum: int = 5) -> pd.DataFrame:
    """같은 작품의 연주끼리 beat별 곡선이 얼마나 닮았는지(평균 쌍별 상관)와 평균 수준이 얼마나 다른지."""
    rows = []
    for work in sorted({r["stats"]["work"] for r in results}):
        members, dynamics, pedal = work_matrices(results, work)
        if len(members) < minimum:
            continue
        row = {"work": work, "performances": len(members)}
        for name, matrix in (("dynamics", dynamics), ("pedal", pedal)):
            for suffix, curves in (("raw", matrix), ("smooth", np.stack([smooth(x, SMOOTH_BEATS) for x in matrix]))):
                corr = pd.DataFrame(curves.T).corr().to_numpy()
                row[f"{name}_corr_{suffix}"] = float(np.nanmean(corr[np.triu_indices(len(members), 1)]))
            row[f"{name}_level_std"] = float(np.nanstd(np.nanmean(matrix, axis=1)))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 독립 재계산 검증


def independent_check(asap_root: Path, results: list[dict], count: int = 5) -> list[dict]:
    """특징 추출기와 다른 방식(파이썬 루프, 0.1ms 샘플링)으로 몇 곡을 다시 계산해 비교한다."""
    loader = ASAPLoader(asap_root)
    rng = np.random.default_rng(1)
    picks = rng.choice(len(results), size=count, replace=False)
    report = []
    for index in picks:
        record = results[index]
        key = record["stats"]["key"]
        performance = load_midi(loader.get_sample(key).performance_path)
        beats = record["sequences"]["beats"]

        dynamics_error = 0.0
        for i in range(len(beats) - 1):
            inside = [n.velocity for n in performance.notes if beats[i] <= n.start < beats[i + 1]]
            value = record["sequences"]["dynamics"][i]
            if not inside:
                assert not record["sequences"]["dynamics_mask"][i]
                continue
            dynamics_error = max(dynamics_error, abs(np.mean(inside) / 127 - value))

        times = np.array([p.time for p in performance.pedals])
        values = np.array([p.value for p in performance.pedals], dtype=float) / 127
        grid = np.arange(beats[0], beats[-1], 1e-4)
        last = np.searchsorted(times, grid, side="right") - 1
        signal = np.where(last >= 0, values[np.maximum(last, 0)], 0.0)
        window = np.searchsorted(beats, grid, side="right") - 1
        sums = np.bincount(window, weights=signal, minlength=len(beats) - 1)
        counts = np.bincount(window, minlength=len(beats) - 1)
        wide = np.diff(beats) >= 0.05
        sampled = sums[wide] / counts[wide]
        pedal_error = float(np.abs(sampled - record["sequences"]["pedal_depth"][wide]).max())

        report.append({
            "key": key,
            "beats": len(beats) - 1,
            "dynamics_max_abs_error": float(dynamics_error),
            "pedal_depth_max_abs_error": pedal_error,
        })
        print(f"  check {key}: dynamics {dynamics_error:.2e}, pedal {pedal_error:.2e}", flush=True)
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


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """NaN을 건너뛰는 가운데 정렬 이동 평균."""
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}", flush=True)


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


def work_matrices(results: list[dict], work: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """작품의 연주들을 (연주, beat) 행렬로 쌓는다. aligned 연주는 beat 수가 같다."""
    members = [r for r in results if r["stats"]["work"] == work]
    lengths = {len(m["sequences"]["dynamics"]) for m in members}
    assert len(lengths) == 1, f"{work}: beat 수가 다르다 {lengths}"
    dynamics = np.stack([m["sequences"]["dynamics"] for m in members]).astype(float)
    pedal = np.stack([m["sequences"]["pedal_depth"] for m in members]).astype(float)
    return members, dynamics, pedal


def plot_same_work_curves(results, works, path: Path) -> None:
    fig, axes = plt.subplots(2, len(works), figsize=(5.6 * len(works), 6.6), sharex="col", squeeze=False)
    for col, work in enumerate(works):
        members, dynamics, pedal = work_matrices(results, work)
        mean_dyn = np.nanmean(dynamics, axis=1)
        high, low = int(np.argmax(mean_dyn)), int(np.argmin(mean_dyn))
        names = [m["stats"]["performer"] for m in members]

        for row, (matrix, ylabel) in enumerate(((dynamics, "Dynamics (velocity / 127)"),
                                                 (pedal, "페달 깊이 (CC64 / 127)"))):
            ax = axes[row, col]
            curves = np.stack([smooth(line, SMOOTH_BEATS) for line in matrix])
            for line in curves:
                ax.plot(line, color=CONTEXT, lw=0.8, alpha=0.9)
            ax.plot(np.nanmedian(curves, axis=0), color=BLUE, lw=1.8)
            ax.plot(curves[high], color=ORANGE, lw=1.5)
            ax.plot(curves[low], color=AQUA, lw=1.5)
            ax.set_ylabel(ylabel)
            if row == 0:
                composer = members[0]["stats"]["composer"]
                ax.set_title(f"{composer} · {members[0]['stats']['title']}  (연주 {len(members)}개)")
            else:
                ax.set_xlabel(f"beat 번호 ({SMOOTH_BEATS}-beat 이동 평균)")
                ax.legend(
                    handles=[
                        Line2D([], [], color=CONTEXT, lw=1, label="개별 연주"),
                        Line2D([], [], color=BLUE, lw=1.8, label="작품 중앙값"),
                        Line2D([], [], color=ORANGE, lw=1.5, label=f"Dynamics 평균 최고: {names[high]}"),
                        Line2D([], [], color=AQUA, lw=1.5, label=f"Dynamics 평균 최저: {names[low]}"),
                    ],
                    loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2,
                )
    fig.suptitle("같은 곡 여러 연주의 Dynamics·페달 곡선", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_same_work_heatmaps(results, works, path: Path) -> None:
    fig, axes = plt.subplots(2, len(works), figsize=(5.6 * len(works), 7.2), squeeze=False)
    cmap = blue_ramp()
    cmap.set_bad("#ececea")
    for col, work in enumerate(works):
        members, dynamics, pedal = work_matrices(results, work)
        for row, (matrix, label) in enumerate(((dynamics, "Dynamics"), (pedal, "페달 깊이"))):
            ax = axes[row, col]
            order = np.argsort(np.nanmean(matrix, axis=1))
            shown = np.stack([smooth(line, 4) for line in matrix[order]])
            low, high = (0.0, 1.0) if label == "페달 깊이" else np.nanpercentile(shown, [2, 98])
            image = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
            ax.grid(False)
            ax.set_yticks([])
            ax.set_ylabel(f"연주 ({label} 평균 오름차순)")
            if row == 0:
                composer = members[0]["stats"]["composer"]
                ax.set_title(f"{composer} · {members[0]['stats']['title']}")
            else:
                ax.set_xlabel("beat 번호")
            fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02,
                         label=label if label == "페달 깊이" else f"{label} (2–98% 범위)")
    fig.suptitle("연주 × beat 히트맵 (행 = 연주, 색이 진할수록 값이 큼)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def histogram(ax, values: np.ndarray, bins, xlabel: str, title: str, unit_note: str | None = None) -> None:
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
    beat_dynamics = np.concatenate([r["sequences"]["dynamics"][r["sequences"]["dynamics_mask"]] for r in results])
    beat_pedal = np.concatenate([r["sequences"]["pedal_depth"] for r in results])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    histogram(axes[0, 0], beat_dynamics, np.linspace(0, 1, 51), "beat별 Dynamics",
              f"beat 단위 Dynamics (beat {len(beat_dynamics):,}개)")
    histogram(axes[0, 1], table.dynamics_mean, 30, "연주별 Dynamics 평균", f"연주별 평균 (연주 {len(table):,}개)")
    histogram(axes[0, 2], table.dynamics_range, 30, "연주별 Dynamics 범위 (5–95%)", "연주별 범위")
    histogram(axes[1, 0], beat_pedal, np.linspace(0, 1, 51), "beat별 페달 깊이",
              f"beat 단위 페달 깊이 (beat {len(beat_pedal):,}개)")
    histogram(axes[1, 1], table.pedal_usage, 30, "연주별 페달 on 비율", "연주별 페달 on 비율")
    histogram(axes[1, 2], table.pedal_change_rate, 30, "연주별 페달 전환 횟수 / beat", "연주별 페달 전환 빈도")
    fig.suptitle("특징별 분포 (ASAP 정렬 연주 전체)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_by_composer(table, path: Path, minimum: int = 20) -> None:
    counts = table.composer.value_counts()
    composers = [c for c in counts.index if counts[c] >= minimum]
    order = (table[table.composer.isin(composers)].groupby("composer")["dynamics_mean"].median()
             .sort_values().index.tolist())
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, column in zip(axes, ["dynamics_mean", "dynamics_range", "pedal_usage", "pedal_change_rate"]):
        groups = [table.loc[table.composer == c, column].to_numpy() for c in order]
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
    fig.suptitle(f"작곡가별 연주 요약 특징 (연주 {minimum}개 이상, 상자 = 25–75%, 수염 = 5–95%)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_eta(eta: dict[str, tuple[float, float]], path: Path) -> None:
    columns = sorted(eta, key=lambda c: eta[c][0])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    positions = np.arange(len(columns))
    ax.barh(positions + 0.19, [eta[c][0] for c in columns], height=0.34, color=BLUE)
    ax.barh(positions - 0.19, [eta[c][1] for c in columns], height=0.34, color=CONTEXT)
    for position, column in zip(positions, columns):
        ax.text(eta[column][0] + 0.01, position + 0.19, f"{eta[column][0]:.2f}", va="center",
                fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(positions, [SUMMARY_LABELS[c] for c in columns])
    ax.set_xlim(0, 1)
    ax.set_xlabel("작품 정체성이 설명하는 분산 비율 (η²)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="실제 작품 라벨"),
                       Line2D([], [], color=CONTEXT, lw=6, label="작품 라벨을 섞은 기대값")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    ax.set_title("연주 요약 특징 중 작품이 좌우하는 부분")
    fig.tight_layout()
    save(fig, path)


def plot_outliers(table, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (x, y) in zip(axes, [("dynamics_mean", "dynamics_range"), ("pedal_usage", "pedal_change_rate")]):
        normal, flagged = table[~table.statistical_outlier], table[table.statistical_outlier]
        ax.scatter(normal[x], normal[y], s=16, color=CONTEXT, edgecolor=SURFACE, linewidth=0.6)
        ax.scatter(flagged[x], flagged[y], s=26, color=ORANGE, edgecolor=SURFACE, linewidth=0.8)
        ax.set_xlabel(SUMMARY_LABELS[x])
        ax.set_ylabel(SUMMARY_LABELS[y])
        middle = table[x].median()
        extreme = (table[f"z_{x}"].abs() + table[f"z_{y}"].abs()).nlargest(3).index
        for index in extreme:
            row = table.loc[index]
            side = -1 if row[x] > middle else 1
            ax.annotate(row.performer, (row[x], row[y]), xytext=(8 * side, 6), textcoords="offset points",
                        fontsize=7, color=INK_SECONDARY, ha="left" if side > 0 else "right")
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CONTEXT, label=f"기준 이내 ({len(normal)})"),
                           Line2D([], [], marker="o", ls="", color=ORANGE, label=f"통계적 이상치 ({len(flagged)})")],
                  loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2)
    fig.suptitle("연주 요약 특징 산점도 (주황 = 전체 분포 또는 같은 작품 안에서 크게 벗어난 연주)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_extraction_check(asap_root: Path, results, key: str, path: Path, span: float = 16.0) -> None:
    """원본 note velocity와 CC64 위에 beat별 특징을 겹쳐 추출이 맞는지 눈으로 확인한다."""
    record = next(r for r in results if r["stats"]["key"] == key)
    performance = load_midi(ASAPLoader(asap_root).get_sample(key).performance_path)
    beats = record["sequences"]["beats"]
    start = beats[len(beats) // 3]
    end = start + span

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    onsets = np.array([n.start for n in performance.notes])
    velocities = np.array([n.velocity for n in performance.notes])
    shown = (onsets >= start) & (onsets < end)
    ax = axes[0]
    ax.scatter(onsets[shown], velocities[shown], s=14, color=CONTEXT, edgecolor=SURFACE, linewidth=0.5)
    for i in np.flatnonzero((beats[:-1] >= start) & (beats[:-1] < end)):
        if record["sequences"]["dynamics_mask"][i]:
            ax.hlines(record["sequences"]["dynamics"][i] * 127, beats[i], beats[i + 1], color=BLUE, lw=2)
    ax.set_ylabel("velocity")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CONTEXT, label="음 onset의 velocity"),
                       Line2D([], [], color=BLUE, lw=2, label="beat 구간 평균 (Dynamics × 127)")],
              loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2)
    ax.set_title("Dynamics: 원본 음 vs beat별 특징")

    times = np.array([p.time for p in performance.pedals])
    values = np.array([p.value for p in performance.pedals])
    ax = axes[1]
    inside = (times >= start - 2) & (times < end + 2)
    ax.step(times[inside], values[inside], where="post", color=CONTEXT, lw=1.2)
    for i in np.flatnonzero((beats[:-1] >= start) & (beats[:-1] < end)):
        ax.hlines(record["sequences"]["pedal_depth"][i] * 127, beats[i], beats[i + 1], color=BLUE, lw=2)
    ax.plot(beats[(beats >= start) & (beats < end)], np.full(((beats >= start) & (beats < end)).sum(), -4),
            "|", color=MUTED, markersize=6)
    ax.set_ylabel("CC64")
    ax.set_xlabel("연주 시각 (초)  ·  아래 눈금 = beat")
    ax.set_xlim(start, end)
    ax.legend(handles=[Line2D([], [], color=CONTEXT, lw=1.2, label="원본 CC64 (다음 이벤트까지 유지)"),
                       Line2D([], [], color=BLUE, lw=2, label="beat 구간 평균 (페달 깊이 × 127)")],
              loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2)
    ax.set_title("Pedaling: 원본 CC64 vs beat별 특징")
    fig.suptitle(f"추출 검증: {key}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------- 실행


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asap-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/dynamics_pedaling"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--works", nargs="+", metavar="WORK",
                        help="같은 곡 곡선을 그릴 작품 (예: Chopin/Etudes_op_10/1). 주면 01, 02번 그림만 *_custom.png로 만든다")
    parser.add_argument("--list-works", action="store_true", help="선택할 수 있는 작품과 연주 수를 출력하고 끝낸다")
    args = parser.parse_args()

    if args.list_works:
        for work, size in sorted(work_sizes(args.asap_root).items(), key=lambda item: (-item[1], item[0])):
            if size >= 2:
                print(f"{size:3d}  {work}")
        return

    works = None
    if args.works:
        try:
            works = check_works(args.works, work_sizes(args.asap_root))
        except ValueError as error:
            parser.error(str(error))

    results = load_or_collect(args.asap_root, args.cache, args.workers, works)
    args.out.mkdir(parents=True, exist_ok=True)

    if works is not None:
        apply_style()
        print("figures")
        plot_same_work_curves(results, works, args.out / "01_same_work_curves_custom.png")
        plot_same_work_heatmaps(results, works, args.out / "02_same_work_heatmaps_custom.png")
        return

    table = add_flags(build_table(results))

    table.to_csv(args.out / "performance_summary.csv", index=False, encoding="utf-8-sig")
    table[table.flagged].sort_values("flags").to_csv(
        args.out / "outliers.csv", index=False, encoding="utf-8-sig"
    )

    print("independent re-computation check")
    check = independent_check(args.asap_root, results)

    eta = {column: eta_squared(table, column) for column in SUMMARY_COLUMNS}
    consistency = same_work_consistency(results)
    consistency.to_csv(args.out / "same_work_consistency.csv", index=False, encoding="utf-8-sig")
    figure_works = pick_works(table)

    apply_style()
    print("figures")
    plot_same_work_curves(results, figure_works, args.out / "01_same_work_curves.png")
    plot_same_work_heatmaps(results, figure_works, args.out / "02_same_work_heatmaps.png")
    plot_distributions(results, table, args.out / "03_distributions.png")
    plot_by_composer(table, args.out / "04_by_composer.png")
    plot_eta(eta, args.out / "05_work_vs_performance_variance.png")
    plot_outliers(table, args.out / "06_outliers.png")
    plot_extraction_check(args.asap_root, results, table.key.iloc[0], args.out / "07_extraction_check.png")

    flag_columns = [c for c in table.columns if c.startswith("flag_")]
    stats = {
        "performances": len(table),
        "works": int(table.work.nunique()),
        "beats_total": int(table.beat_count.sum()),
        "flag_counts": {c[5:]: int(table[c].sum()) for c in flag_columns},
        "flagged_performances": int(table.flagged.sum()),
        "statistical_outliers": int(table.statistical_outlier.sum()),
        "pedal_usage_below_5pct": {
            "count": int((table.pedal_usage < 0.05).sum()),
            "by_composer": table[table.pedal_usage < 0.05].composer.value_counts().to_dict(),
        },
        "empty_beat_ratio_max": round(float(table.empty_beat_ratio.max()), 4),
        "velocity_low_ratio_max": round(float(table.velocity_low_ratio.max()), 4),
        "onset_inside_ratio_min": round(float(table.onset_inside_ratio.min()), 4),
        "summary_describe": table[SUMMARY_COLUMNS].describe().round(4).to_dict(),
        "same_work_consistency": {
            "works": len(consistency),
            "median": consistency.drop(columns="work").median().round(4).to_dict(),
            "level_std_across_all_performances": {
                "dynamics_mean": round(float(table.dynamics_mean.std()), 4),
                "pedal_depth_mean": round(float(table.pedal_depth_mean.std()), 4),
            },
        },
        "eta_squared": {c: {"observed": round(v[0], 4), "shuffled": round(v[1], 4)} for c, v in eta.items()},
        "independent_check": check,
        "figure_works": figure_works,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats["flag_counts"], ensure_ascii=False), "flagged:", stats["flagged_performances"])


if __name__ == "__main__":
    main()
