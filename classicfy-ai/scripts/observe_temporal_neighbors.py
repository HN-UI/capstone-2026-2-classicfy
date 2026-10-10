"""Inspect cross-work CNN neighbors with explicit feature summaries, not style labels."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.data import CHANNELS, WindowDataset, restore_sequences
from embedding.training import extract_embeddings, load_model
from validate_embedding import collect_groups, summarize, weighted_distances
from validate_tempo import BLUE, GREEN, GRAY, chart, save, setup_style, write_csv


FEATURE_GROUPS = ((0,), (1,), (2,), (3,), (4, 5, 6))
CHANNEL_NAMES = ("템포", "루바토", "강약", "아티큘레이션", "페달 깊이", "페달 사용 비율", "페달 전환")
METRIC_NAMES = {"representative": "대표 성향", "width": "변화 폭", "step_rms": "beat별 변화 크기",
                "combined": "대표 성향 + 변화 폭"}


def block_distances(values):
    """Seven summary coordinates; each of the five musical features weighs equally."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected seven finite summary coordinates per performance")
    squared = sum(np.mean((values[:, None, ix] - values[None, :, ix]) ** 2, axis=2)
                  for ix in FEATURE_GROUPS)
    return np.sqrt(squared / 5)


def step_summary(sequence):
    """RMS of true adjacent valid beat differences; never bridge a missing beat."""
    valid = sequence.valid[:, 1:] & sequence.valid[:, :-1]
    count = valid.sum(axis=1)
    if np.any(count == 0):
        raise ValueError(f"No adjacent valid beats: {sequence.key}")
    differences = np.diff(sequence.values.astype(float), axis=1)
    return np.sqrt(np.where(valid, differences ** 2, 0).sum(axis=1) / count), count


def compare_candidate(distances, pieces, query, candidate):
    """Exact peer expectation excludes the selected candidate; ties receive half credit."""
    pieces = np.asarray(pieces)
    if pieces[query] == pieces[candidate]:
        raise ValueError("Candidate must belong to another work")
    pool = np.flatnonzero(pieces != pieces[query])
    peers = np.flatnonzero(pieces == pieces[candidate])
    peers = peers[peers != candidate]
    if not len(peers):
        raise ValueError("Need other performances of candidate work")
    own = float(distances[query, candidate])

    def closer_share(others):
        equal = np.isclose(others, own, atol=1e-12, rtol=0)
        return float(np.mean(((others > own) & ~equal) + .5 * equal))

    return {"candidate_distance": own,
            "other_work_random_mean_distance": float(distances[query, pool].mean()),
            "candidate_work_peer_mean_distance": float(distances[query, peers].mean()),
            "closer_than_peer_share": closer_share(distances[query, peers]),
            "closer_than_other_work_share": closer_share(distances[query, pool]),
            "candidate_work_peers": len(peers), "cross_work_candidates": len(pool)}


def select_fixed_queries(rows):
    """First/middle/last alphabetic test work, then its median key; independent of outcomes."""
    works = sorted({row["piece"] for row in rows if row["split"] == "test"})
    if not works:
        raise ValueError("No held-out test works")
    selected = []
    for position in sorted({0, len(works) // 2, len(works) - 1}):
        indices = sorted((i for i, row in enumerate(rows) if row["piece"] == works[position]),
                         key=lambda i: rows[i]["key"])
        selected.append(indices[len(indices) // 2])
    return selected


def plot_observations(summary, examples, descriptors, rows, out):
    setup_style()
    test_rows = [row for row in rows if row["split"] == "test"]
    works = len({row["piece"] for row in test_rows})
    fig, ax = chart("CNN 검색은 어떤 연주 특징을 닮게 찾을까?", f"학습에 쓰지 않은 {works}작품 · {len(test_rows)}연주 | CNN 최근접과 같은 후보 작품의 다른 연주 비교")
    metrics = list(METRIC_NAMES)
    percentages = [100 * summary[name]["closer_than_peer_share"] for name in metrics]
    ax.bar(np.arange(4), percentages, color=[BLUE, GRAY, GRAY, GREEN], width=.65)
    ax.axhline(50, color="black", ls=":", label="후보 작품에서 무작위 선택 기대값 50%")
    ax.set(xticks=np.arange(4), xticklabels=[METRIC_NAMES[name] for name in metrics],
           ylabel="다른 연주보다 feature 거리가 가까운 비율 (%)", ylim=(0, 110))
    for i, value in enumerate(percentages):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center", va="bottom")
    ax.legend(fontsize=9)
    save(fig, out / "01_feature_agreement.png", "연주별 비교 비율을 평균한 기술통계입니다. 청취 정답·검색 정확도가 아닙니다. 같은 거리에는 0.5점을 줍니다.")
    directory = out / "examples"
    directory.mkdir()
    for number, example in enumerate(examples, 1):
        ids = [example[role] for role in ("query_index", "near_index", "contrast_index")]
        for name in ("representative", "width", "step_rms"):
            fig, ax = chart(f"사례 {number} · {METRIC_NAMES[name]}", "A 기준 연주 · B CNN 최근접 · C B와 같은 작품의 CNN 최저 유사도 연주")
            x, width = np.arange(7), .25
            for offset, index, role, color in zip((-1, 0, 1), ids, ("A", "B", "C"), (BLUE, GREEN, GRAY)):
                ax.bar(x + offset * width, descriptors[name][index], width, color=color,
                       label=f"{role} · {rows[index]['key'].rsplit('/', 1)[-1].removesuffix('.mid')}")
            ax.set_xticks(x, CHANNEL_NAMES, rotation=15)
            ax.set_ylabel("표준화된 상대 feature 단위")
            ax.axhline(0, color="black", lw=.6)
            ax.legend(fontsize=8)
            note = {"representative": "대표 성향은 채널별 평균입니다. Rubato만 절댓값 중앙값입니다. Pedaling은 세 채널을 모두 표시합니다.",
                    "width": "변화 폭은 전체 유효 beat의 p95-p5입니다. C는 CNN으로 고른 대조이며, 모든 feature에서 더 멀다고 보장하지 않습니다.",
                    "step_rms": "원래 시간축에서 이웃한 유효 beat 변화량의 RMS입니다. 변화 순서·방향·음악적 구절의 유사성까지 측정하지 않습니다."}[name]
            save(fig, directory / f"{number:02d}_{name}.png", note)


def write_report(out, stats, rows, example_rows):
    summaries = stats["test_summary"]
    count = stats["test_queries"]
    table = []
    for name, label in METRIC_NAMES.items():
        s = summaries[name]
        table.append(f"| {label} | {100*s['closer_than_peer_share']:.1f}% | {s['closer_than_peer_mean_count']}/{count} |")
    examples = []
    for number, example in enumerate(stats["examples"], 1):
        examples.append(f"### 사례 {number}\n\n" + "\n".join(
            f"- {role}: `{rows[example[field]]['key']}`" for role, field in
            (("A 기준", "query_index"), ("B 최근접", "near_index"), ("C 같은 후보 작품의 대조", "contrast_index")))
            + f"\n\nCNN cosine: A–B **{example['near_cosine']:.3f}**, A–C **{example['contrast_cosine']:.3f}**.\n\n"
            + f"[대표 성향](examples/{number:02d}_representative.png) · [변화 폭](examples/{number:02d}_width.png) · [beat별 변화 크기](examples/{number:02d}_step_rms.png)\n")
        a_b, a_c = [row for row in example_rows if row["example"] == number]
        examples[-1] += "\n| 비교 기준 | A–B 거리 | A–C 거리 | 더 가까운 후보 |\n|---|---:|---:|---|\n" + "\n".join(
            f"| {METRIC_NAMES[name]} | {a_b[name]:.3f} | {a_c[name]:.3f} | {'B' if a_b[name] < a_c[name] else 'C'} |"
            for name in ("representative", "width", "step_rms")) + "\n"
    representative = 100 * summaries["representative"]["closer_than_peer_share"]
    width = 100 * summaries["width"]["closer_than_peer_share"]
    step = 100 * summaries["step_rms"]["closer_than_peer_share"]
    if representative > max(width, step):
        interpretation = "현재는 **대표적인 연주 성향을 중심으로 검색하는 경향**이 있다. 변화 폭과 beat별 변화 크기의 일치는 그보다 약하다."
    else:
        interpretation = "대표 성향·변화 폭·beat별 변화 크기의 일치는 위 표처럼 서로 다르다. 어떤 표현을 주로 검색하는지 구분해 읽어야 한다."
    macro = " / ".join(f"{METRIC_NAMES[name]} {100*stats['test_work_macro'][name]['closer_than_peer_share']:.1f}%"
                       for name in ("representative", "width", "step_rms"))
    (out / "README.md").write_text("""# 현재 1D CNN의 다른 작품 검색 관찰

""" + f"**같은 후보 작품의 다른 연주보다 가까운 비율: 대표 성향 {representative:.1f}%, 변화 폭 {width:.1f}%, beat별 변화 크기 {step:.1f}%.**\n\n"
        + f"{stats['best_epoch']} epoch의 저장 모델을 그대로 사용했다. 모델·학습 목표·검색 가중치를 수정하지 않았다.\n" + """

사람의 유사도 응답은 아직 없으며 아래 숫자는 feature 기반 관찰이다.

## 먼저 볼 결과

[어떤 특징을 닮게 찾는가](01_feature_agreement.png)

""" + f"학습에 쓰지 않은 test {stats['test_works']}작품·{count}연주를 query로 사용했다. 전체 {stats['performances']}연주 중 query와 같은 작품의\n" + """
모든 연주를 제외하고, 현재 128차원 임베딩의 cosine 최근접 B를 선택했다.
B의 작품을 고정한 상태에서 그 작품의 나머지 연주들과 비교했다.
예를 들어 B 작품의 다른 연주 10개 중 9개보다 feature 거리가 가까우면 90%다.
""" + f"이 비율을 {count} query에 동일 비중으로 평균했다. 동점은 0.5점이다.\n" + """
후보 작품의 연주를 무작위로 고르면 이 비교 비율의 기대값은 50%다.

| 관찰 기준 | 같은 후보 작품의 다른 연주보다 가까운 비율 | 다른 연주들의 평균 거리보다 가까운 query |
|---|---:|---:|
""" + "\n".join(table) + """

이 비율은 **검색 정확도나 사람이 비슷하다고 평가한 비율이 아니다.**
CNN 후보를 입력 feature의 별도 요약으로 대조한 기술통계다.
""" + interpretation + """
이 지표만으로 연주의 변화 패턴까지 유사하게 찾는다고 설명할 수는 없다.

""" + f"작품마다 같은 비중으로 평균하면 **{macro}**다. 같은 작품의 여러 query를 독립된 사람 평가로 해석하지 않는다.\n" + """

## 고정한 세 사례

test 작품을 이름순으로 정렬해 첫·중간·마지막 작품의 중앙 순번 연주를 A로 선택했다.
성공한 사례를 탐색해 고르지 않았다. B는 다른 작품 전체의 CNN 최근접이다.
C는 B와 같은 작품의 연주 중 A와 CNN cosine이 가장 낮은 연주다.
이 대조는 CNN상 차이를 보여주는 사례이며 feature상 모든 축이 멀어야 하는 정답은 아니다.

""" + "\n".join(examples) + """
각 표의 거리는 같은 행 안에서만 비교한다. 대표 성향이 가까워도 변화 폭과 인접 변화가 어긋날 수 있다.
곡이 다르므로 같은 beat 번호의 값을 직접 맞춰 비교하지 않았다.

## 비교값의 의미

- **대표 성향:** 7채널 평균, Rubato만 절댓값 중앙값. 부호가 있는 값은 자기 작품의 공통 패턴 대비 경향이다.
  서로 다른 작품의 절대 BPM·음량이 같다는 뜻은 아니다.
- **변화 폭:** 7채널 각각 `p95-p5`. 시간 순서는 사용하지 않는다.
- **beat별 변화 크기:** 원래 시간축에서 인접한 두 유효 beat 차이의 RMS. 결측을 건너 이어 붙이지 않는다.
  이 지표 하나로 변화의 방향·순서·음악적 구절의 유사성을 측정할 수는 없다.
- **대표 성향 + 변화 폭:** 기존 14차원 통계 요약의 거리. 이 조합이 변화 순서를 담는 것은 아니다.

각 거리에서 Tempo/Rubato/Dynamics/Articulation/Pedaling 다섯 feature의 비중을 동일하게 두었다.
Pedaling의 세 채널은 한 묶음의 평균이다. 이 거리로 검색 후보를 다시 선택하지 않았다.
전체 다른 작품 후보 평균과 후보 작품 내부 평균을 모두 CSV에 기록했다.
작품마다 같은 비중으로 집계한 결과는 `stats.json`의 `test_work_macro`에 있다.

## 조건과 한계

- 입력 reference·정규화는 [기존 구현](../../src/embedding/README.md) 그대로다. 작품별 후보 cohort가 common/scale에 포함된다.
- test 작품은 모델 가중치 학습·checkpoint 선택에 사용하지 않았다. gallery에는 train/validation/test가 모두 포함된다.
- 전곡은 64beat 구간 벡터의 평균·표준편차 128차원이다. 전곡의 구간 배치 순서는 보존하지 않는다.
- 저장 checkpoint에서 570연주 벡터를 다시 추출해 기존 벡터와 일치하는지 확인했다.
- feature상 유사성이 귀로 느끼는 연주 성향·선호 유사성의 정답은 아니다. 같은 연주자 식별 평가도 아니다.
- [기존 청취 자료](../_listening_evaluation/README.md)는 통계 임베딩으로 선정한 다른 문항이며 실제 응답은 아직 없다.
  그 자료의 B/C를 이번 CNN 검색의 청취 정답으로 사용하지 않았다.

## 산출물과 재현

- [query_metrics.csv](query_metrics.csv): test연주 × 4관찰 기준의 후보/거리/비교비율
- [work_metrics.csv](work_metrics.csv): test작품별 동일 지표 평균
- [feature_summaries.csv](feature_summaries.csv): 전체연주·7채널 대표값/폭/인접 변화 RMS/지원 수
- [examples.csv](examples.csv): A/B/C 임베딩 유사도와 feature별 거리
- [top5.csv](top5.csv): test연주의 다른 작품 Top5
- [stats.json](stats.json): 조건·source/cache/checkpoint/code hash·집계·선정 기록
- [실행 스크립트](../../scripts/observe_temporal_neighbors.py)

저장소 루트에서 기존 checkpoint를 읽어 실행한다. 결과를 덮어쓰지 않도록 새 출력 폴더를 지정한다.

```bash
.venv/bin/python classicfy-ai/scripts/observe_temporal_neighbors.py \\
  --run ../datasets/temporal_embedding_run01 \\
  --out /tmp/classicfy-temporal-neighbors
```
""", encoding="utf-8")


def main():
    ai = Path(__file__).resolve().parents[1]
    datasets = ai.parent.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=datasets / "temporal_embedding_run01")
    parser.add_argument("--out", type=Path, default=ai / "analysis/temporal_neighbors")
    parser.add_argument("--asap-root", type=Path, default=datasets / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=datasets / "nASAP")
    parser.add_argument("--cache", type=Path, default=datasets / "feature_normalization_raw.npz")
    args = parser.parse_args()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("Choose a new/empty output directory")
    torch.set_num_threads(2)
    stored = json.loads((args.run / "dataset.json").read_text())
    training = json.loads((args.run / "training.json").read_text())
    cache_hash = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if cache_hash != stored["cache_sha256"]:
        parser.error("Raw feature cache differs from the saved run")
    groups, audits, _, provenance = collect_groups(SimpleNamespace(asap_root=args.asap_root.resolve(),
        nasap_root=args.nasap_root.resolve(), cache=args.cache, **stored["cohort_settings"]))
    if provenance != stored["source_provenance"]:
        parser.error("Source provenance differs from the saved run")
    frame = pd.read_csv(args.run / "embeddings.csv")
    rows = frame[["key", "piece", "grid", "windows", "split"]].to_dict("records")
    sequences = restore_sequences(groups, audits)
    dataset = WindowDataset(sequences, window_size=training["model_config"]["window_size"], stride=stored["stride"])
    model, checkpoint = load_model(args.run / "best.pt")
    fresh_rows, fresh_vectors, _ = extract_embeddings(model, dataset)
    with np.load(args.run / "embeddings.npz", allow_pickle=False) as saved:
        vectors = saved["vectors"].copy()
        assert [row["key"] for row in rows] == saved["keys"].tolist() == [row["key"] for row in fresh_rows]
    np.testing.assert_allclose(fresh_vectors, vectors, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(frame[[f"z{i}" for i in range(128)]].to_numpy(), vectors, rtol=1e-6, atol=1e-6)
    assert rows == [{**row, "split": stored["split"][row["piece"]]} for row in fresh_rows]
    stat_by_key = {key: summarize(values) for group in groups for key, values in zip(group["keys"], group["values"])}
    stat = np.array([stat_by_key[row["key"]] for row in rows])
    step_by_key = {s.key: step_summary(s) for s in sequences}
    descriptors = {"representative": stat[:, ::2], "width": stat[:, 1::2],
                   "step_rms": np.array([step_by_key[row["key"]][0] for row in rows])}
    distances = {name: block_distances(values) for name, values in descriptors.items()}
    distances["combined"] = weighted_distances(stat, stat, "all")
    pieces = np.array([row["piece"] for row in rows])
    unit = vectors.astype(float)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    similarities = unit @ unit.T
    similarities[pieces[:, None] == pieces[None, :]] = -np.inf
    neighbor = np.argmax(similarities, axis=1)
    queries = [i for i, row in enumerate(rows) if row["split"] == "test"]
    query_rows = []
    for q in queries:
        b = int(neighbor[q])
        for name, distance in distances.items():
            query_rows.append({"query": rows[q]["key"], "query_piece": pieces[q], "candidate": rows[b]["key"],
                               "candidate_piece": pieces[b], "candidate_split": rows[b]["split"],
                               "cosine_similarity": float(similarities[q, b]), "metric": name,
                               **compare_candidate(distance, pieces, q, b)})
    table = pd.DataFrame(query_rows)
    work = table.groupby(["query_piece", "metric"], sort=True).agg(
        queries=("query", "count"), candidate_distance=("candidate_distance", "mean"),
        candidate_work_peer_mean_distance=("candidate_work_peer_mean_distance", "mean"),
        other_work_random_mean_distance=("other_work_random_mean_distance", "mean"),
        closer_than_peer_share=("closer_than_peer_share", "mean"),
        closer_than_other_work_share=("closer_than_other_work_share", "mean")).reset_index()
    summary = {}
    for name in METRIC_NAMES:
        subset = table[table.metric == name]
        summary[name] = {column: float(subset[column].mean()) for column in
                        ("candidate_distance", "candidate_work_peer_mean_distance", "other_work_random_mean_distance",
                         "closer_than_peer_share", "closer_than_other_work_share")}
        summary[name]["closer_than_peer_mean_count"] = int((subset.candidate_distance < subset.candidate_work_peer_mean_distance).sum())
    examples, example_rows = [], []
    for number, q in enumerate(select_fixed_queries(rows), 1):
        b = int(neighbor[q])
        peers = np.flatnonzero(pieces == pieces[b])
        c = int(peers[np.argmin(similarities[q, peers])])
        if c == b:
            c = int(next(i for i in peers if i != b))
        example = {"query_index": q, "near_index": b, "contrast_index": c,
                   "near_cosine": float(similarities[q, b]), "contrast_cosine": float(similarities[q, c])}
        examples.append(example)
        for role, index in (("B", b), ("C", c)):
            example_rows.append({"example": number, "query": rows[q]["key"], "role": role,
                                 "candidate": rows[index]["key"], "cosine_similarity": float(similarities[q, index]),
                                 **{name: float(distance[q, index]) for name, distance in distances.items()}})
    feature_rows = [{"key": row["key"], "piece": row["piece"], "split": row["split"], "channel": channel,
                     **{name: float(values[i, ch]) for name, values in descriptors.items()},
                     "adjacent_valid_pairs": int(step_by_key[row["key"]][1][ch])}
                    for i, row in enumerate(rows) for ch, channel in enumerate(CHANNELS)]
    top5 = pd.read_csv(args.run / "neighbors.csv")
    top5 = top5[top5.query_split == "test"]
    for q in queries:
        first = top5[(top5["query"] == rows[q]["key"]) & (top5["rank"] == 1)].iloc[0]
        assert first.candidate == rows[int(neighbor[q])]["key"]
    stats = {"run": str(args.run.resolve()), "best_epoch": checkpoint["epoch"], "performances": len(rows),
             "test_queries": len(queries), "test_works": len({pieces[q] for q in queries}),
             "candidate_policy": "All corpus splits, excluding every performance of query work",
             "control_policy": "Exact mean and tie-aware comparison against every other performance of selected candidate work",
             "selection_policy": "First/middle/last alphabetical test work; median performance key; fixed before inspecting outcomes",
             "cache_sha256": cache_hash, "checkpoint_sha256": hashlib.sha256((args.run / "best.pt").read_bytes()).hexdigest(),
             "embedding_sha256": hashlib.sha256((args.run / "embeddings.npz").read_bytes()).hexdigest(),
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_provenance": provenance,
             "reencoded_vectors_match": True, "test_summary": summary,
             "test_work_macro": {name: {column: float(work[work.metric == name][column].mean()) for column in
                                       ("candidate_distance", "candidate_work_peer_mean_distance", "closer_than_peer_share")}
                                 for name in METRIC_NAMES}, "examples": examples, "human_style_labels": False}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "query_metrics.csv", query_rows)
    work.to_csv(args.out / "work_metrics.csv", index=False)
    write_csv(args.out / "feature_summaries.csv", feature_rows)
    write_csv(args.out / "examples.csv", example_rows)
    top5.to_csv(args.out / "top5.csv", index=False)
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_observations(summary, examples, descriptors, rows, args.out)
    write_report(args.out, stats, rows, example_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
