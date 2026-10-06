"""Compare independent listening responses with whole-piece and excerpt embeddings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from listening_study import FIELDS, validate_response, write_csv
from validate_embedding import BLOCKS


def closest(vectors, field):
    """B/C are comparisons, not musical ground truth. Preserve ties."""
    values = np.asarray(vectors)
    components = [np.mean((values[0, indices] - values[1:, indices]) ** 2, axis=1)
                  for name, indices in BLOCKS.items() if field == "overall" or name.lower() == field]
    distances = np.sqrt(np.mean(components, axis=0))
    if np.isclose(distances[0], distances[1], rtol=1e-9, atol=1e-12): return "tie"
    return "B" if distances[0] < distances[1] else "C"


def collect(responses, organizer, allow_partial=False):
    rows, participants = [], set()
    for data in responses:
        valid = validate_response(data, organizer, allow_partial=allow_partial)
        participant = data["participant"]
        if participant in participants: raise ValueError(f"Duplicate participant: {participant}")
        participants.add(participant); rows.extend(valid)
    if not allow_partial and participants != set(organizer["public"]["participants"]):
        raise ValueError("Three complete independent P1/P2/P3 responses are required; use --allow-partial for an explicitly partial report")
    return rows, sorted(participants)


def summarize_responses(rows, organizer):
    pairs, cross, checks = [], [], []
    for task in organizer["tasks"]:
        answers = [r for r in rows if r["task_id"] == task["id"]]
        for field in FIELDS:
            if task["kind"] != "cross":
                numeric = [int(r[field]) for r in answers if r[field] != "unclear"]
                pairs.append(dict(task_id=task["id"], kind=task["kind"], focus=task["focus"], field=field,
                    n_answers=len(answers), n_numeric=len(numeric), n_unclear=len(answers)-len(numeric),
                    median_rating=float(np.median(numeric)) if numeric else None,
                    whole_distance=task["whole_distances"][0], excerpt_distance=task["excerpt_distances"][0]))
            else:
                whole = closest(task["whole_vectors"], field); excerpt = closest(task["excerpt_vectors"], field)
                counts = dict(B=0, C=0, tie=0, unclear=0)
                for row in answers:
                    schedule = next(s for s in organizer["schedules"][row["participant"]] if s["task_id"] == task["id"])
                    choice = row[field]
                    if choice in {"X", "Y"}:
                        key = schedule["x_key" if choice == "X" else "y_key"]
                        choice = "B" if key == task["keys"][1] else "C"
                    counts[choice] += 1
                decisive = counts["B"] + counts["C"]
                cross.append(dict(task_id=task["id"], field=field, n_answers=len(answers), **counts,
                    n_decisive=decisive, whole_closest=whole, excerpt_closest=excerpt,
                    whole_agreement=counts[whole]/decisive if decisive and whole in {"B", "C"} else None,
                    excerpt_agreement=counts[excerpt]/decisive if decisive and excerpt in {"B", "C"} else None))
    participants = sorted({r["participant"] for r in rows})
    lookup = {(r["participant"], r["task_id"]):r for r in rows}
    for participant in participants:
        for field in FIELDS:
            same, first, again = [lookup.get((participant, t), {}).get(field) for t in ("S08", "S01", "S09")]
            gap = abs(int(first)-int(again)) if first not in {None,"unclear"} and again not in {None,"unclear"} else None
            checks.append(dict(participant=participant,field=field,identical_rating=same,
                               first_rating=first,repeat_rating=again,absolute_repeat_difference=gap))
    return pairs, cross, checks


def save_figure(fig, path):
    if len(fig.axes) != 1: raise ValueError("One chart per image")
    fig.tight_layout(); fig.savefig(path, dpi=170, facecolor="white"); plt.close(fig)


def figures(rows, pairs, cross, out):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.spines.top":False,
                         "axes.spines.right":False, "figure.figsize":(9,5.5)})
    def rating_chart(task_ids, field_for, filename, title, labels=None):
        fig, ax = plt.subplots(); colors = ["#197b6a", "#c87645", "#5d76b5"]
        for x, tid in enumerate(task_ids):
            field = field_for(tid)
            observations = [int(r[field]) for r in rows if r["task_id"] == tid and r[field] != "unclear"]
            if observations:
                offsets = np.linspace(-.13,.13,len(observations)) if len(observations)>1 else [0]
                for k,(offset,value) in enumerate(zip(offsets,observations)):
                    ax.scatter(x+offset,value,s=65,c=colors[k%3],zorder=3)
                median = np.median(observations)
                ax.plot([x-.22,x+.22],[median,median],color="#263932",lw=2.3,zorder=4)
            else: ax.text(x,1.1,"No numeric\nrating",ha="center",color="#666")
            count = next(p for p in pairs if p["task_id"] == tid and p["field"] == field)
            ax.text(x,5.4,f'n={count["n_numeric"]}; unclear={count["n_unclear"]}',ha="center",fontsize=9)
        ax.set_xticks(range(len(task_ids)), labels or task_ids);ax.set_ylim(.7,5.7);ax.set_yticks(range(1,6))
        ax.set_ylabel("Heard difference: 1 almost none → 5 very large")
        ax.set_title(title+"\nDots = independent listeners; line = median",loc="left")
        ax.grid(axis="y",alpha=.2);save_figure(fig,out/filename)
    rating_chart([f"S{i:02d}" for i in range(1,6)],
        lambda tid:next(p["focus"].lower() for p in pairs if p["task_id"]==tid),
        "01_feature_contrasts.png","Can listeners hear the selected feature contrasts?",
        ["Tempo", "Rubato", "Dynamics", "Articulation", "Pedaling"])
    nearfar = [next(p for p in pairs if p["task_id"]==tid and p["field"]=="overall") for tid in ("S06","S07")]
    labels = [f'{p["task_id"]} · {p["kind"]}\nwhole {p["whole_distance"]:.3f} / excerpt {p["excerpt_distance"]:.3f}' for p in nearfar]
    rating_chart(["S06","S07"],lambda _:"overall","02_near_far.png",
                 "Same work and anchor: does a farther candidate sound more different?",labels)
    fig, ax = plt.subplots()
    for p in pairs:
        if p["field"] != "overall" or p["task_id"] not in {f"S{i:02d}" for i in range(1,8)}:continue
        if p["median_rating"] is None:continue
        ax.scatter(p["excerpt_distance"],p["median_rating"],s=70,c="#197b6a")
        ax.annotate(p["task_id"],(p["excerpt_distance"],p["median_rating"]),xytext=(5,7),textcoords="offset points")
    ax.set_xlabel("Embedding distance of the two excerpts actually heard")
    ax.set_ylabel("Median heard difference (numeric ratings only)");ax.set_ylim(.7,5.7);ax.set_yticks(range(1,6))
    ax.set_title("Selected examples: embedding distance vs heard difference\nSeven selected pairs; no population-level accuracy claim",loc="left")
    ax.grid(alpha=.2);save_figure(fig,out/"03_distance_vs_hearing.png")
    fig,ax=plt.subplots();categories=["B","C","tie","unclear"]; colors=["#197b6a","#c87645","#aaa","#dedede"]
    cases=[p for p in cross if p["field"]=="overall"]
    for j,cat in enumerate(categories):
        ax.bar(np.arange(len(cases))+(j-1.5)*.18,[p[cat] for p in cases],width=.18,label=cat,color=colors[j])
    ax.set_xticks(range(len(cases)),[f'{p["task_id"]}\nwhole nearest: {p["whole_closest"]}; excerpt nearest: {p["excerpt_closest"]}' for p in cases])
    ax.set_ylabel("Independent listener count");ax.set_yticks(range(4));ax.set_ylim(0,3.6)
    ax.set_title("Across works: which candidate sounds closer to A?\nB/C are candidate identities, not correct answers",loc="left")
    ax.legend(frameon=False,ncol=4);save_figure(fig,out/"04_cross_work_choices.png")


def analyze(args):
    organizer=json.loads(args.organizer.read_text())
    rows,participants=collect([json.loads(p.read_text()) for p in args.responses],organizer,args.allow_partial)
    if not rows: raise ValueError("No completed task responses to analyze")
    pairs,cross,checks=summarize_responses(rows,organizer)
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/"responses.csv",rows);write_csv(args.out/"pair_summary.csv",pairs)
    write_csv(args.out/"cross_choices.csv",cross);write_csv(args.out/"consistency.csv",checks)
    figures(rows,pairs,cross,args.out)
    summary=dict(study_id=organizer["study_id"],participants=participants,
                 partial=args.allow_partial,completed_task_responses=len(rows),pair_summary=pairs,cross_choices=cross,consistency=checks)
    (args.out/"results.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    readme=f"""# 청취 평가 결과

- Study: `{organizer['study_id']}` / 참가자: {', '.join(participants)} / 완료 문항 응답: {len(rows)}
- {'부분 응답 분석입니다. 누락 문항은 제외되며 세 명의 완료 결과로 해석하지 않습니다.' if args.allow_partial else 'P1·P2·P3의 모든 문항 응답을 검증했습니다.'}
- 척도는 1 거의 없음–5 매우 큰 차이입니다. 개인 점과 중앙값을 함께 표시합니다.

| 그림 | 읽는 방법 |
|---|---|
| [feature 차이](01_feature_contrasts.png) | 각 feature를 중심으로 고른 쌍에서 그 차이가 들리는가? 점이 낮거나 흩어지면 근거가 약합니다. |
| [가까운/먼 후보](02_near_far.png) | 같은 기준 연주의 가까운·먼 후보를 비교합니다. 먼 후보의 청취 차이가 더 크면 이 사례에서 방향이 맞습니다. 전체·발췌 거리는 따로 읽습니다. |
| [거리와 청취](03_distance_vs_hearing.png) | 실제 들은 구간의 거리가 커질 때 차이 점수도 커지는지 봅니다. 선별된 7쌍을 일반 정확도로 해석하지 않습니다. |
| [다른 작품 선택](04_cross_work_choices.png) | 사람이 고른 B/C와 전체·발췌 최근접을 비교합니다. B는 음악적인 정답이 아닙니다. |

`pair_summary.csv`에 numeric/unclear 수를, `cross_choices.csv`에 B/C/tie/unclear와 확정 선택 분모를,
`consistency.csv`에 동일 음원(S08) 점수·반복(S01/S09) 차이를 보존합니다. 확인 문항으로 응답자를 자동 제외하지 않습니다.
동점·판단 어려움을 오답으로 바꾸지 않습니다. 후보를 고른 feature 외 다른 feature의 평가는 `responses.csv`에 남습니다.

3인·소수 선별 사례의 탐색적 결과입니다. 추천 성능·선호 만족도·모든 feature의 필수성은 검증하지 않습니다.
작품 내 상대적 MIDI 특징과 귀로 들리는 절대적 빠르기·음량·녹음의 음색/잔향은 다를 수 있습니다.
녹음 gain을 보정하지 않았으므로 Dynamics·Pedaling 판단에는 녹음 조건이 섞일 수 있습니다.
전체 임베딩과 발췌 임베딩의 후보 순서가 바뀐 경우 양쪽을 따로 보고합니다.
"""
    (args.out/"README.md").write_text(readme)
    print(f"Analyzed {len(participants)} participants / {len(rows)} completed task responses: {args.out}")


def main():
    ai=Path(__file__).resolve().parents[1];workspace=ai.parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses",nargs="+",type=Path,help="Independent participant JSON exports")
    parser.add_argument("--organizer",type=Path,default=ai/"analysis/_listening_evaluation/organizer.json")
    parser.add_argument("--out",type=Path,default=workspace/"datasets/listening_evaluation/results")
    parser.add_argument("--allow-partial",action="store_true",help="Explicitly report partial participants/tasks")
    args=parser.parse_args()
    try:analyze(args)
    except (ValueError,FileNotFoundError) as error:parser.exit(2,str(error)+"\n")


if __name__=="__main__":main()
