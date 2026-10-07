"""Select four recordings by feature summaries, then inspect saved CNN distances."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from validate_tempo import plt, setup_style


CHANNELS = ("tempo", "rubato", "dynamics", "articulation", "pedal_depth",
            "pedal_down_ratio", "pedal_changes")
BLOCKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9, 10, 11, 12, 13))
IDS = ("B1", "J1", "B2", "J2")
GREEN, GRAY, CORAL, BLUE = "#148766", "#87919d", "#d76b47", "#2878c8"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def work_family(key):
    """Treat movements/repeat variants of one Beethoven sonata as one work."""
    parts = key.split("/")
    if parts[:2] == ["Beethoven", "Piano_Sonatas"]:
        return f"Beethoven/Piano_Sonatas/{parts[2].split('-')[0]}"
    return "/".join(parts[:-1])


def performer_id(key):
    """Dataset filename identifier only; no inferred full performer names."""
    return re.split(r"\d", Path(key).stem)[0].casefold()


def select_pairs(features, coordinates):
    """Greedy nearest pairs using no CNN values; four works and performer IDs."""
    be = np.flatnonzero(features.key.str.startswith("Beethoven/"))
    ba = np.flatnonzero(features.key.str.startswith("Bach/"))
    squared = sum(np.mean((coordinates[be, None, :][:, :, block]
                           - coordinates[None, ba, :][:, :, block]) ** 2, axis=2)
                  for block in BLOCKS)
    distances = np.sqrt(squared / 5)
    keys = features.key.tolist()
    candidates = sorted((float(distances[i, j]), keys[a], keys[b], int(a), int(b))
                        for i, a in enumerate(be) for j, b in enumerate(ba)
                        if performer_id(keys[a]) != performer_id(keys[b]))
    chosen, used_works, used_people = [], set(), set()
    for rank, (distance, key_a, key_b, a, b) in enumerate(candidates, 1):
        works = {work_family(key_a), work_family(key_b)}
        people = {performer_id(key_a), performer_id(key_b)}
        if works & used_works or people & used_people:
            continue
        chosen.append((a, b, distance, rank))
        used_works.update(works)
        used_people.update(people)
        if len(chosen) == 2:
            return chosen, {"beethoven": len(be), "bach": len(ba),
                            "cross_pairs": len(be) * len(ba),
                            "distinct_performer_cross_pairs": len(candidates)}
    raise ValueError("Cannot find two pairs with distinct work/performer identifiers")


def work_label(key):
    parts = key.split("/")
    if parts[0] == "Beethoven":
        number, movement = parts[2].split("-", 1)
        return f"소나타 {number}번 · {movement.split('_')[0]}악장"
    return f"푸가 BWV {parts[2].removeprefix('bwv_')}"


def draw_figure(selected, xy, variance, distances, pairs, out):
    setup_style()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 16, "axes.labelsize": 11,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 9), facecolor="#fcfcfb")
    fig.text(.045, .947, "작곡가보다 연주 스타일이 가까운가?", fontsize=25, weight="bold")
    fig.text(.045, .903, "5개 feature 요약으로 비슷한 연주를 먼저 선정 → 현재 CNN 임베딩에서 거리 확인",
             fontsize=13, color="#555953")
    nearest = distances + np.eye(4) * 10
    hit_count = sum(nearest[i].argmin() == (i ^ 1) for i in range(4))
    fig.text(.045, .852, f"선정한 4연주 중 {hit_count}연주: 다른 작곡가의 스타일 유사 연주가 최근접",
             fontsize=17, weight="bold", color=GREEN)

    ax = fig.add_axes((.07, .365, .43, .40))
    ax.set_title("A. 우리 임베딩의 2차원 배치", loc="left", pad=25, weight="bold")
    for a, b in ((0, 2), (1, 3)):
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], ls="--", lw=1.6,
                color=GRAY, alpha=.6, zorder=1)
    for a, b in ((0, 1), (2, 3)):
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], lw=3.2, color=GREEN, zorder=2)
    for i, code in enumerate(IDS):
        color, marker = (CORAL, "o") if i % 2 == 0 else (BLUE, "^")
        ax.scatter(*xy[i], s=240, marker=marker, color=color, edgecolor="white", lw=1.8, zorder=3)
        ax.annotate(code, xy[i], xytext=(0, 16), textcoords="offset points",
                    ha="center", weight="bold", fontsize=13, color=color)
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.3)
    ax.grid(alpha=.4)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", color=CORAL, label="베토벤"),
               Line2D([], [], marker="^", ls="", color=BLUE, label="바흐"),
               Line2D([], [], color=GREEN, lw=3, label="스타일 유사 쌍"),
               Line2D([], [], color=GRAY, ls="--", label="같은 작곡가")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.285, .280),
               ncols=4, frameon=False, fontsize=10)

    ax = fig.add_axes((.655, .365, .285, .40))
    ax.set_title("B. 원래 128차원에서의 실제 거리", loc="left", pad=25, weight="bold")
    comparisons = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (2, 1)]
    labels = ["B1–J1  스타일 유사", "B2–J2  스타일 유사", "B1–B2  같은 작곡가",
              "J1–J2  같은 작곡가", "B1–J2  나머지 쌍", "B2–J1  나머지 쌍"]
    values = [distances[a, b] for a, b in comparisons]
    colors = [GREEN, GREEN, GRAY, GRAY, "#c8cdd3", "#c8cdd3"]
    ax.barh(np.arange(6), values, color=colors, height=.60)
    ax.set_yticks(np.arange(6), labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Cosine distance = 1 − cosine similarity\n작을수록 비슷함")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.4)
    ax.set_axisbelow(True)
    for y, value in enumerate(values):
        ax.text(value + .012, y, f"{value:.3f}", va="center", fontsize=11,
                weight="bold" if y < 2 else "normal")
    fig.text(.045, .231, "선정한 연주  ·  B = Beethoven / J = J. S. Bach", weight="bold", fontsize=12)
    for i, row in selected.iterrows():
        x, y = (.045 if i < 2 else .515), (.196 if i % 2 == 0 else .162)
        composer = "베토벤" if i % 2 == 0 else "바흐"
        fig.text(x, y, f"{IDS[i]}  {composer} · {work_label(row.key)} · {Path(row.key).stem}", fontsize=11)
    fig.text(.045, .112, f"선정 기준: 대표 성향 + 변화 폭의 14D 요약, 5개 feature 동일 비중. 쌍별 거리 {pairs[0][2]:.3f} / {pairs[1][2]:.3f}.",
             fontsize=10, color="#555953")
    fig.text(.045, .084, f"PCA는 선정한 4개의 L2 정규화 벡터에 적용 (분산 {sum(variance) * 100:.1f}% 표시). 거리 판정은 오른쪽 원차원 값 기준.",
             fontsize=10, color="#555953")
    fig.text(.045, .056, "Feature 기준으로 고른 탐색 사례입니다. 학습·검증·test 연주가 섞여 있으며, 청취 유사성이나 전체 검색 성능의 입증은 아닙니다.",
             fontsize=10, color="#555953")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"composer_vs_style.{suffix}", dpi=200)
    plt.close(fig)


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ai.parent.parent / "datasets/temporal_embedding_run01")
    parser.add_argument("--features", type=Path, default=ai / "analysis/_feature_search_roles/embeddings.csv")
    parser.add_argument("--audit", type=Path, default=ai / "analysis/temporal_neighbors")
    parser.add_argument("--out", type=Path, default=ai / "analysis/composer_style_example")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory to preserve existing figures")
    features = pd.read_csv(args.features)
    embeddings = pd.read_csv(args.run / "embeddings.csv").set_index("key")
    if not features.key.is_unique or not embeddings.index.is_unique:
        raise ValueError("Duplicate performance keys")
    if set(features.key) != set(embeddings.index):
        raise ValueError("Feature and CNN cohorts differ")
    audit = json.loads((args.audit / "stats.json").read_text())
    for field, path in (("checkpoint_sha256", args.run / "best.pt"),
                        ("embedding_sha256", args.run / "embeddings.npz")):
        if sha256(path) != audit[field] or not audit["reencoded_vectors_match"]:
            raise ValueError("Saved embeddings do not match the verified checkpoint audit")
    with np.load(args.run / "embeddings.npz", allow_pickle=False) as cache:
        # CSV is the human-readable export; check its numeric values below.
        cached_vectors = cache["vectors"]
    if not np.allclose(embeddings[[f"z{i}" for i in range(128)]].to_numpy(), cached_vectors,
                       atol=1e-12, rtol=0):
        raise ValueError("CSV vectors differ from audited NPZ")
    embeddings = embeddings.loc[features.key]
    coordinates = features[[f"coordinate_{i}" for i in range(14)]].to_numpy()
    summaries = pd.read_csv(args.audit / "feature_summaries.csv").set_index(["key", "channel"])
    expected = np.array([[value for channel in CHANNELS for value in
                          summaries.loc[(key, channel), ["representative", "width"]]]
                         for key in features.key])
    if not np.allclose(coordinates, expected, atol=1e-12, rtol=0):
        raise ValueError("Feature selection summaries differ from current CNN input summaries")
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite feature coordinates")
    pairs, pool = select_pairs(features, coordinates)
    indices = [index for a, b, _, _ in pairs for index in (a, b)]
    selected = features.iloc[indices].reset_index(drop=True).copy()
    selected.insert(0, "id", IDS)
    selected["split"] = embeddings.iloc[indices].split.to_numpy()
    z = embeddings.iloc[indices][[f"z{i}" for i in range(128)]].to_numpy()
    if not np.isfinite(z).all() or np.any(np.linalg.norm(z, axis=1) == 0):
        raise ValueError("Invalid CNN vectors")
    normalized = z / np.linalg.norm(z, axis=1, keepdims=True)
    distances = np.clip(1 - normalized @ normalized.T, 0, 2)
    centered = normalized - normalized.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ axes[:2].T
    variance = singular[:2] ** 2 / np.sum(singular ** 2)
    selected["pc1"], selected["pc2"] = xy.T
    rows = []
    for a, b in combinations(range(4), 2):
        feature_distance = np.sqrt(sum(np.mean((coordinates[indices[a], block]
                                                 - coordinates[indices[b], block]) ** 2)
                                       for block in BLOCKS) / 5)
        rows.append({"a": IDS[a], "b": IDS[b], "feature_distance": feature_distance,
                     "cnn_cosine_distance": distances[a, b],
                     "pca_distance": np.linalg.norm(xy[a] - xy[b]),
                     "same_composer": a % 2 == b % 2,
                     "selected_style_pair": a // 2 == b // 2})
    nearest = distances + np.eye(4) * 10
    hits = [int(nearest[i].argmin()) == (i ^ 1) for i in range(4)]
    stats = {"pool": pool, "best_epoch": audit["best_epoch"],
             "selection": "Greedy smallest existing 14D five-block feature distance; different composers, sonata families, work folders, and filename performer IDs. CNN values not used in selection.",
             "pair_ranks": [pair[3] for pair in pairs],
             "nearest_style_partner": dict(zip(IDS, hits)),
             "pca": "Centered PCA fitted only to four L2-normalized 128D CNN vectors; no axis scaling",
             "pca_variance_ratio": variance.tolist(),
             "hashes": {str(p.resolve()): sha256(p) for p in
                        (args.features, args.audit / "feature_summaries.csv", args.audit / "stats.json",
                         args.run / "embeddings.csv", args.run / "embeddings.npz",
                         args.run / "best.pt", Path(__file__))}}
    args.out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out / "selected_performances.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(args.out / "pairwise_distances.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(z, index=IDS, columns=[f"z{i}" for i in range(128)]).to_csv(args.out / "selected_cnn_embeddings.csv", encoding="utf-8-sig")
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    draw_figure(selected, xy, variance, distances, pairs, args.out)
    table = "\n".join(f"| {r['a']}–{r['b']} | {r['feature_distance']:.6f} | {r['cnn_cosine_distance']:.6f} |" for r in rows)
    recordings = "\n".join(f"| {row.id} | `{row.key}` | {row.split} |" for row in selected.itertuples())
    (args.out / "README.md").write_text(f"""# 작곡가와 연주 스타일: 실제 네 연주의 CNN 배치

선정한 네 연주 중 **{sum(hits)}/4**가 같은 작곡가보다 다른 작곡가의 스타일 유사 파트너에 가깝다.

![Figure](composer_vs_style.png)

[발표용 PDF](composer_vs_style.pdf)

| ID | 실제 연주 키 | 모델 split |
|---|---|---|
{recordings}

## 선정과 해석

베토벤 {pool['beethoven']}연주·바흐 {pool['bach']}연주, cross-composer {pool['cross_pairs']}쌍에서
기존 작품 공통 제거/표준화된 feature 요약 거리가 작은 순서로 탐색했다.
파일명 연주자 ID가 다른 쌍 {pool['distinct_performer_cross_pairs']}개를 거리·파일명으로 정렬하고,
첫 쌍을 고른 후 사용한 작품/연주자 ID가 겹치지 않는 첫 번째 쌍을 고른다.
순위는 각각 {pairs[0][3]}위·{pairs[1][3]}위다. 전역 두 쌍 거리 합 최소화는 아니다.
베토벤의 같은 소나타 다른 악장/반복 변형은 같은 작품으로 취급했다.
연주자 ID는 파일명의 첫 숫자 이전 문자열이며 별도의 실명 확인은 수행하지 않았다.

선정에는 CNN 거리를 사용하지 않았다. 14D 요약은 7채널 각각 대표값과 p95−p5 폭이다.
대표값은 평균이고 Rubato만 절댓값 중앙값이다. Tempo/Rubato/Dynamics/Articulation은
각 2좌표의 평균 제곱차, Pedaling은 6좌표의 평균 제곱차를 사용한 후 다섯 블록을 평균하고
제곱근을 취한다. 원 feature/학습 모델/가중치는 변경하지 않았다.
같은 입력 feature에서 나온 요약과 CNN의 일치 사례이며 독립적인 스타일 정답은 아니다.

## 실제 거리

| 비교 | 선정용 14D feature 거리 | 현재 CNN 128D cosine 거리 |
|---|---:|---:|
{table}

두 거리 열은 서로 다른 단위다. 작은 값일수록 해당 표현에서 가깝다.
네 점 PCA는 L2 정규화한 현재 best epoch {audit['best_epoch']}의 128D 전곡 벡터에만 적합했고
두 축의 표시 분산은 {variance.sum()*100:.2f}%다. 축의 비율은 동일하게 유지했다.
PCA 거리 자체를 cosine 거리로 해석하지 않고 원차원 거리 6쌍을 모두 함께 보여준다.
작곡가 정보는 점 색/기호와 같은 작곡가 연결선에만 사용했다. 별도 metadata embedding은 계산하지 않았다.

## 범위

Feature 기반으로 가까운 쌍을 골라 구성한 탐색 사례로, 전체 성능 추정이 아니다.
Train/validation/test가 섞이고 현재 split에는 Bach test 작품이 없으므로 독립 test 4연주 실험도 아니다.
청취자가 느끼는 스타일/선호는 평가하지 않았다. 대표값과 폭으로 선택했으므로
세밀한 구절/시간 변화 패턴의 유사성까지 입증하지 않는다.
작품별 reference에 후보 연주가 포함되는 기존 정규화 조건을 유지했다.
현재 CNN 검증 보고서의 checkpoint/NPZ hash를 확인하고, CSV 벡터가 NPZ와 일치하며
선정용 feature 요약이 현재 CNN 분석 요약과 일치함을 확인했다. 이번 실행에서 재학습하지 않았다.

## 재현

저장소 루트에서 새 출력 폴더를 지정한다.

```bash
.venv/bin/python classicfy-ai/scripts/plot_composer_style_example.py --out /tmp/classicfy-composer-style
```

[선정 연주·14D·PCA 좌표](selected_performances.csv) · [원차원 6쌍 거리](pairwise_distances.csv) ·
[선정 연주의 CNN 원벡터](selected_cnn_embeddings.csv) · [조건·source hash](stats.json) ·
[스크립트](../../scripts/plot_composer_style_example.py)
""", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "style_partner_nearest": sum(hits),
                      "selected": selected[["id", "key", "split"]].to_dict("records"),
                      "pca_variance": float(variance.sum())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
